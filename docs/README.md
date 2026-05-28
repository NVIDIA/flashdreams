<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# FlashDreams documentation

This directory hosts the Sphinx sources for the FlashDreams documentation site.

## Build locally

Doc dependencies are declared in the workspace-root `pyproject.toml`
under `[dependency-groups] docs`. The workspace `uv sync` already
installs `flashdreams` (needed by autodoc), so building is a single
command:

```bash
# from the repo root
uv run --group docs sphinx-build -b html docs/source docs/_build/html
```

The rendered site lands in `docs/_build/html/index.html`. Open it with
any browser, e.g. `xdg-open docs/_build/html/index.html`.

## Live preview (auto rebuild)

Use Sphinx live-reload to avoid rerunning build + HTTP server commands on every edit:

```bash
# from the repo root
uv run --group docs sphinx-autobuild -E docs/source docs/_build/html --port 8000
```

Then keep this process running and open:
`http://127.0.0.1:8000`

Any changes under `docs/source/` rebuild automatically and refresh the page.

## Layout

```
docs/
└── source/
    ├── conf.py             # Sphinx configuration (theme + extensions)
    ├── index.rst           # overview landing page + top-level toctrees
    ├── quickstart/
    │   ├── index.rst
    │   ├── installation.rst
    │   └── first_world_model.rst
    ├── developer_guides/
    │   ├── offline_vs_online.rst
    │   ├── index.rst
    │   ├── new_integration.rst
    │   ├── system_overview.rst
    │   ├── usage_patterns.rst
    │   ├── configs.rst
    │   └── interactive_serving.rst
    ├── api/
    │   ├── index.rst
    │   ├── cli.rst
    │   ├── core.rst        # flashdreams.core (attention, distributed, …)
    │   ├── infra.rst       # flashdreams.infra (pipeline, diffusion, …)
    │   ├── integrations.rst     # flashdreams.recipes (wan, cosmos, …)
    │   └── serving.rst     # serving architecture and launch patterns
    └── models/
        ├── index.rst
        ├── omnidreams.rst
        ├── self_forcing.rst
        ├── causal_forcing.rst
        ├── causal_wan22.rst
        ├── lingbot_world.rst
        ├── flashvsr.rst
        ├── cosmos_predict2.rst
        └── wan21.rst
```

Benchmark data now follows a JS + Markdown pipeline:

- Per-model benchmark tables live in
  ``docs/source/_static/performance/<model>/perf-*.md``.
- Model pages declare chart metadata with ``data-benchmark-*`` attributes.
- ``docs/source/_static/js/benchmark_chart.js`` loads those markdown tables at
  runtime and renders SVG charts in the browser.

The ``docs/benchmarks/`` directory still hosts the benchmark JSON corpus and
schema used for tracking raw measurements.

## Hosting on GitHub Pages

`.github/workflows/doc.yml` builds the docs on every push / PR /
release and pushes the rendered HTML to the `gh-pages` branch
(layout cribbed from
[`gsplat`](https://github.com/nerfstudio-project/gsplat/blob/main/.github/workflows/doc.yml)):

| Trigger                | Deployed under                  | Banner shows |
| ---------------------- | ------------------------------- | ------------ |
| `push` to `main`       | `gh-pages:/main/`               | `main`       |
| `release` (tag)        | `gh-pages:/versions/<ver>/`     | `<ver>`      |
| `pull_request`         | (build only, no deploy)         | n/a          |
| `workflow_dispatch`    | `gh-pages:/versions/<ver>/`     | `<ver>`      |

One-time GitHub setup after the first run:

1. **Settings → Pages** → set *Source* to **Deploy from a branch**,
   branch = `gh-pages`, folder = `/ (root)`.
2. (Optional) point a custom domain at it and uncomment the
   `cname:` line in `doc.yml`.
3. Each release also appends its version to
   `gh-pages:/versions/index.txt`, useful for a future version-picker
   widget on the site.

### CI doc build (CPU-only)

The CI workflow uses `uv sync --only-group docs` to install Sphinx
tooling, then manually installs CPU-only PyTorch and the lightweight
subset of flashdreams runtime deps. The heavy GPU packages
(`transformer-engine`, `pynvml`, `boto3`, `mediapy`, `cv2`) are mocked
via `autodoc_mock_imports` in `docs/source/conf.py` so they never need
to be present.

## Adding new content

- **A new model integration** — append a section to `source/apis/integrations.rst`
  using `.. automodule:: flashdreams.recipes.<name>`, and add a launcher
  walk-through to `source/models/<name>.rst`. Wire the new file into
  the models toctree in `source/models/index.rst`.
- **A new infra component** — re-export the public symbols from the
  package `__init__.py`, then add an `.. autoclass::` block to the
  relevant section of `source/apis/infra.rst`.
- **A new API category** — drop a new `source/apis/<topic>.rst`, add it
  to `index.rst`, and (optionally) introduce a new captioned toctree.
