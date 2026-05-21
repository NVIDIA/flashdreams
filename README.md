# FlashDreams

FlashDreams is a high-performance streaming inference stack for world and video
models, with a unified CLI and multi-GPU support for long-rollout generation.

## Documentation

Start with the docs site sections:

- **Overview / Getting Started**
  - Installation
  - Launch your first world model
  - Supported models
- **Developer Guides**
  - Adding a new method (recipe)
  - System overview
  - Configs
  - Interactive serving
- **Reference**
  - CLI
  - API
  - Evaluation and benchmarks

If you are browsing this repository directly, the Sphinx sources live under
`docs/source/`.

## Quickstart (repo workflow)

```bash
git clone https://github.com/NVIDIA/flashdreams.git
cd flashdreams
uv sync --extra dev --extra runners
export HF_TOKEN=<your-hf-token>
uv run flashdreams-run --help
```

Run a first model:

```bash
uv run flashdreams-run self-forcing-wan2.1-t2v-1.3b-flash --total-blocks 7
```

## Integrations

Model integrations are organized under `integrations/` (for example
`self_forcing`, `causal_forcing`, `fastvideo_causal_wan22`, `lingbot`,
`omnidreams`).

## Development

```bash
uv run pre-commit run -a
uv run pytest -m "not manual"
```
