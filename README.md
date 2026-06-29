# uday610.github.io

Personal technical blog built with Quarto, deployed via GitHub Actions to GitHub Pages.

## Directory Structure

```
uday610.github.io/
  _quarto.yml              # site config (title, theme, navbar)
  index.qmd                # home page (auto-lists all posts)
  about.qmd                # about page
  styles.css               # custom CSS
  posts/                   # all blog posts go here
    hello-world/
      index.qmd            # one post = one folder + index.qmd
    my-next-post/
      index.qmd
      image.png            # images live next to the post
  .github/workflows/
    publish.yml            # auto-builds and deploys on push to main
```

## Creating a New Post

1. Create a folder under `posts/` with a slug name:

```bash
mkdir posts/my-new-post
```

2. Create `posts/my-new-post/index.qmd` with this template:

```markdown
---
title: "Your Post Title"
date: "2026-07-01"
categories: [topic1, topic2]
---

Your content here. Standard Markdown works.

## Code blocks

```python
print("hello")
```

## Images (place image files in the same folder)

![description](image.png)
```

3. Push to publish:

```bash
cd ~/uday610.github.io
git add posts/my-new-post/
git commit -m "Add post: my-new-post"
git push
```

GitHub Actions builds and deploys automatically. Live in ~1-2 minutes.

## Preview Locally Before Publishing (optional)

Requires Quarto installed locally: https://quarto.org/docs/get-started/

```bash
cd ~/uday610.github.io
quarto preview
```

Opens a local server at http://localhost:4444 with live reload. This is your staging area -- review here before pushing.

Without Quarto installed, you can still preview the raw Markdown since `.qmd` is just Markdown with YAML front matter.

## Draft Posts (not published)

Add `draft: true` to the front matter:

```markdown
---
title: "Work in Progress"
date: "2026-07-01"
draft: true
---
```

Draft posts show up in local preview but are excluded from the published site.

## Quick Reference

| Task | Command |
|---|---|
| New post | `mkdir posts/slug && edit posts/slug/index.qmd` |
| Preview locally | `quarto preview` |
| Publish | `git add . && git commit -m "msg" && git push` |
| Check deploy status | https://github.com/uday610/uday610.github.io/actions |
| Live site | https://uday610.github.io/ |
