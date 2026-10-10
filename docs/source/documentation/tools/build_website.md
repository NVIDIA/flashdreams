---
title: 'Build Website'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

FlashDreams uses Zensical to render the Markdown under
`docs/source/`. Run all commands from the repository root.

## Build the website

```bash

uv run --only-group docs zensical build --clean -f docs/zensical.toml

```

The rendered website is written to `docs/dist/`. `--clean` removes stale output
before rendering.

## Preview changes

```bash

uv run --only-group docs zensical serve -f docs/zensical.toml

```

Open `http://localhost:8000`. The development server watches the documentation
sources and rebuilds changed pages. Stop it with `Ctrl+C`.

The site configuration lives in `docs/zensical.toml`, and the sidebar is
defined by `.nav.yml` files below `docs/source/`. The documentation
workflow builds the same site for pull requests and publishes updates from
`main` and releases to GitHub Pages.
