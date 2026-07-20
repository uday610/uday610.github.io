Since 2025, Mixture of experts became a popular architecture of the latest LLM. MoE packs the intelligent by keeping a steady compute budget. An MoE layer replace the traditional dense feedforward layer with multiple specialized “expert” layers. When each token is propagated, only a subset of the MoE layers are activated. Each input token is processed by only the top-k most relevant experts (typically k=1-8), as determined by a learned router. This selection mechanism allows models to pack more intelligence by billions of parameters but computing only a fraction of them. This essentially breaks the linear relationship between model size and the compute costs. Due to this computational benefits, the MoE architecture become very popular and adopted significantly accross the industry. latest GLM, Deepseek, MinMax, Kimi, Gemma-4, Nemotron-3 they all have MoE variants.

However, even if MoE address the compute, it puts challenge in terms of memory. For every token the sheer amount of weights have to be loaded in the GPU memory. Today most of the MoE Models are very large that cannot be fit into a single GPU. Hence there are parallelism and other inference optimization techniques are getting adopted specifically to address MoE model scaling. 

Many of the current industry documentations are specific to certain inference engines or GPU vendors with implementation details. In this post, I will attempt to generalize these MoE inference challenges and mitigation technique without going into framework specific knobs or benchmarking data. 


### Common architectural pattern of the MoE Models.

Before diving into of MoE inference challenges and mitigation practice, lets recap common modules in the MoE models.

MoE can have generally 3 parts

1.  Router module: Choose the experts
2.  Routed experts: MLP which only process their assigned tokens
3.  Shared experts: Optional MLPs which process all the tokens, independent of routing

Moreover, modern MoE models often combine sparse expert layers with attention variants such as GQA and MLA to reduce the KV-cache footprint during inference. GQA, which uses fewer key and value heads than query heads, has been widely adopted since 2023. More recent leading MoE models, including DeepSeek-V2/V3, GLM-5/5.2, and Kimi-K2/Kimi-K2-Thinking, use MLA, where key and value vectors are projected into a compressed latent representation for caching. This combination requires inference engineers to consider attention and MoE parallelism together when designing an efficient end-to-end serving strategy.

### Expert Parallelism

Traditionally for large model, tensor parallelism (TP) and pipeline parallelism (PP) are two main mechanism to deal with large models in inference. MoE brings a new parallelism dimension, expert parallelism or EP. Simple TP relay on parallelizing the big matrices so TP on MoE model means every experts' matrix multiplication operations are shruds in every GPU (or rank) which cause more communication bandwidth as even if a few experts selected for every token, all rank has to take part of the communication.

Expert parallelism on the other hand parallelize the expert structure itself. Here different experts are placed on different GPUs. Tokens are routed to the GPUs holding their selected experts, and those GPUs execute the experts locally. Compared to TP, EP allows experts GEMM larger and more efficient, reduces synchronization inside the experts. Expert layers now scale independently on large fleet of GPUs whereas dense layer or attention can scale as per their TP on smaller number of GPUs. But like everything else, it introduces some tradeoff. As tokens only goes to few experts, it creates sparse dispatch to selected experts and then combine (when aggregating) communication pattern which is sensitive of network topology and routing efficiency. Depending on the workload it is possible it can create an load imbalance when some specific expets gets more traffic and cause network bottleneck where other experts remain idle.

One common intuition could be for small number of experts like <32 TP can be often better as it can provide a good load balancing without the concerns of expert parallelism. On the other hand for large number of experts often EP is better strategy to reduce network traffic over large number of GPUs. It can depend on the intermediate size of the model too, large intermediate size often gets benefit from TP as big matrix multiplication can be shard across GPUs. On the other hand, smaller intermediate size TP become less useful and EP can be preferred.

However, a hybrid strategy TP+EP is often very common where a moderate TP (2-8) combined with EP often give more balanced result.

Router and shared-expert parallelism: EP distributes only the routed experts. The router is a relatively small dense layer, so its weights are commonly replicated across ranks, with each rank computing expert assignments for its local tokens. The shared expert processes every token, so it is generally parallelized like a regular dense mlp layer, replicated when tokens are divided using DP or sequence parallelism, or sharded with TP when its weight size are large. 

### Load imbalance problem

Some experts receive more tokens than others depending on the workload. The GPUs hosting popular experts become overloaded with token communication and expert computation, while other GPUs remain underutilized. The MoE layer cannot finish until the busiest GPU finishes processing its tokens. So one overloaded expert GPU becomes the straggler, while other GPUs wait idle. This increases layer latency and reduces overall throughput

DeepSeek’s Expert Parallel Load Balancing, EPLB addresses this imbalance by replicating popular experts and placing the copies across GPUs. The idea is simple, adding redundancy and use them to replicate the busy experts. It gets historical load statistics of every experts, and experts with higher load are replicated by coping the expert weights to the redundant GPU slots, balancing the loads.

In the global policy EPLB looks accross all the GPUs accross the nodes, and replicate an expert to any GPU on any node accross the network. This is a highly flexible approach but as cluster size grows it can create high cross node network traffic and bottleneck.

In the hierarchical policy the load balance in hierarchical structural manner. It is specially designed for expert group model such as Deepseek V3. At the cluster level the algorithm balances the workload by assigning the expert groups accross the nodes. Then in the node level heavily loaded expert replicated and then offloaded to the redundant GPU slots in the same node. This mechanism gurantees the expert replicas are within their assigned hardware boundary and reduce the cross-node traffic.

EPLB adds replica for the busy experts, but during dispatch time the incoming tokens still needed to send to the replica which are less busy, otherwise even distribution can overload GPU which is already busy in expert compute. So dispatch time load balancing is a technique which decides how the current batch will be distributed accross all the replica to keep the load even. The implementation could be framework dependent, for example, SGLang use their LPLB method for dispatch time load balancing for routed experts and Waterfill method for shared experts.

### Fast expert communication libraries

MoE inference has two communication steps per layer: expert dispatch (send tokens to the GPU holding the assigned expert) and expert combine (send results back to the original GPUs holding dispatched tokens). Generic collectives do not directly handle MoE-specific token routing and irregular all-to-all exchange.

Specialized libraries such as DeepEP and MoRI-EP provide MoE-specific communication APIs backed by custom GPU kernels. These kernels use GPU-level communication primitives, such as NVSHMEM, MORI-SHMEM, or other libraries depending on the GPU vendor and their implementation, to perform P2P transfers within a node and GPUDirect RDMA (on Nvidia GPU) or PeerDirect RDMA (on AMD GPUs) transfers across nodes. The underlying interconnects are typically NVLink or xGMI within a node, and InfiniBand or RoCE across nodes. DeepEP or MoRI-EP use the router output to group tokens by destination expert, exchange variable-length tokens across GPUs, and return the expert outputs to the original GPUs in the correct token order. The token activations themselves can be quantized like FP8/FP4 before dispatch, reducing the amount of data exchanged during all-to-all communication. 

They provide high-throughput and low-latency communication paths and expose asynchronous operations so the inference engine can control synchronization.

### Communication and Compute overlap

DeepEP and MoRI-EP provides low-latency, high-througput expert dispatch and combine communication primitive. The asyncrhnoys nature of these primitives are exploited by inference engine to hide communication delay under computation. Inference engines like VLLM or SgLang uses dual-batch-overlap or two-batch overlap. These mechanisms are designed to solve the problem in MoE serving by hiding the latency of the all-to-all network communication required for routing tokens to experts. They achieve this by splitting the incoming global batch of requests into two smaller micro-batches. While one micro-batch is transmitting its tokens across the network to remote GPUs (the communication phase), the active GPU is kept busy executing the dense matrix multiplications (the computation phase) for the other micro-batch. Routed-expert dispatch/combine communication can also be overlapped with shared-expert computation because the two paths are independent until their outputs are merged.

### Optimized kernels for MoE

After dispatched tokens reached to their respective experts, each experts perform its own feedforward GEMM. Since different GEMM has irregular shape and they are often very small during decode phase. Running these GEMM kernels separately with standard GEMM kernel cause poor GPU utilization and high-kernel launch overhead. So specillized MoE specific compute kernels solve this problem by mutiple variants of GEMM kernel

1.  Grouped GEMM: Execute the matrix multiplication for many experts with different token counts.
    
2.  Fused MoE kernel: combine communication, GEMMs, activation and other steps such as router top-k selection, token permutation into a larger fused kernel.
    
3.  Low precision GEMM kernel: Low precision GEMM kernels such FP8/FP4, are also used to reduce compute and memory cost.
    

In LLM’s Prefill and decode phase very different numbers of tokens arrive to each expert. This requires distinct token layouts for the expert GEMM in each phase.

In the prefill phase there are generally many tokens resulting large expert GEMM and throughput oriented dispatch. Because of high token volume, they are packed in contiguous layouts, which improve the throughput.

On the other hand, decode generally results few tokens resulting small expert GEMMs which become sensetive to latency. To alliviate tokens are padded to form fixed size masked layouts, helping to create fixed CUDA graph which can be replayed to reduce kernel launch overhead, improving latency.

DeepGEMM for Nvidia GPU or MoE and GEMM kernels from AMD AITER provides such optimized kernels.

### Combining DP Attention with EP

Last year, DeepSeek-V3 became a popular MoE model for inference benchmarking. It has 256 routed experts with large expert-weight footprint which motivated large-scale expert parallelism across many GPUs. At the same time, its attention layers use MLA, where Tensor parallelism provides limitted benifit. Newer open-source models such as GLM-5 and Kimi-K2 continued this MLA+MoE architecture pattern. This made Data Parallel Attention (DPA) for the attention layers combined with EP for the expert layers a common serving pattern that needs to be considered end to end.

For MLA model, TP for the attention layer duplicates the KV cache for all the ranks, which reduce the benifit of MLA architecture whose main goal was reducing the KV cache footprint. So in DP Attention, data parallelism applies only to the attention layers. Attention layers replicated accross GPUs, and each GPU processes a different set of requests and maintains the KV cache for those requests. This prevents the same request’s MLA KV cache from being duplicated across TP ranks. It is especially useful for MLA models, where TP provides limited benefit because the compressed KV representation is not naturally divided across many attention ranks.

#### DP Attention combined with EP

The attention from each DP rank then propagated to the MoE layers. In dispatch phase, EP distributed the tokens to the selected experts picked from the EP ranks, and then during combine phase results are combined into the original DP Attention rank.

The DP attention for the MLA layer and EP for the expert layers become a suitable parallelism strategy for each part of the model.

### MoE weight offloading and disaggregated serving

Expert weights can take significant spaces in the GPU HBM can cause less space for KV cache specially for the long context serving. Similar to KV-cache offloading, expert weights can be kept host memory instead of GPU HBM. Before the corresponding MoE layer executes, the required weights are prefetched back to the GPU. Asynchronous prefetching can overlap this transfer with computation from earlier layers. This help large batch/long context situation for more space for KV cache but tradeoff is additional CPU-to-GPU transfer, which can increase latency when the transfer cannot be fully hidden.

MoE serving can also benefit from prefill-decode disaggregation. Prefill routes many tokens, creating large all-to-all communication and large expert GEMMs, while decode routes fewer tokens and is more sensitive to latency. When both phases share the same EP workers, a large prefill can occupy the communication fabric and expert GPUs, delaying decode can end up waiting, which delays the all-to-all communication and creates pipeline bubbles. Disaggregation avoids this by separating prefill and decode into different GPU pools, allowing each pool to use different EP scales, communication modes, and kernels.

### Wrapping up

As the MoE models are getting popular, it poses a multidimensional challenge for inference. MoE's sparse activations allows to pack huge total parameter
cause **memory pressure**, and we have seen how expert parallelism helps to aliviate the problem. This in turn gives a new challenge which is **communication pressure**. And for that how optimized libraries like DeepEP and MoRI-EP and advanced scheduling such as DBO helps. Finally as routing is data dependent, it cause **imbalance pressure** for popular experts where DPLB, dispatch time load balancing, and pushes for disaggregation. 

MoE inference strategies are rapidly moving. In this post I mainly talked about MoE inference techniques is being done in the steady-state MoE execution. However, in the production setup, additional requirements come to ensure production operation and resilience. Newer technique such as Elastic Expert parallelism are being developed to address these issue by dynamic EP configuration changing when the server is running, or emerging development to mitigate partial-fault tolerance by redistributing expert weights across the new GPU sets. Very curious to see how more development in the MoE inference and serving space will shape up. 






