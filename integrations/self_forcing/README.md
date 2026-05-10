# flashdreams-self-forcing

Self-Forcing distilled streaming T2V inference for Wan 2.1 1.3B,
packaged as a [`flashdreams`](../..) plugin, in a standalone repo.

This is a worked example of the
[Adding a new recipe](../../docs/source/developer_guides/new_recipes.rst)
developer-guide flow.

## Shipped slugs

| slug | description |
| --- | --- |
| `self-forcing-wan2.1-t2v-1.3b` | Self-Forcing distilled Wan 2.1 1.3B T2V (Wan VAE decoder, 4-step). |
| `self-forcing-wan2.1-t2v-1.3b-flash` | Same DiT, swapped to the TAEHV (LightTAE) decoder for faster decoding. |

## Install

The plugin is registered as a `uv` workspace member in the repo-root
`pyproject.toml`, so a single `uv sync` from the repo root pulls it in:

```bash
uv sync
```

Standalone (outside the workspace) also works:

```bash
uv pip install -e integrations/self_forcing
```

## HuggingFace setup

Checkpoints are auto-downloaded from HuggingFace at first run. Set an
auth token first.

```bash
# huggingface token.
export HF_TOKEN=<your-hf-token>

# (optional) override the cache location.
export HF_HOME=~/.cache/huggingface  # default
```

## Run

Once installed, the slugs are discovered automatically by `flashdreams-run`:

```bash
# List every registered runner (this plugin's slugs appear under "self-forcing-*").
uv run flashdreams-run --help

# Per-runner help: every overridable field is a CLI flag.
uv run flashdreams-run self-forcing-wan2.1-t2v-1.3b --help

# Single-GPU run with the bundled demo prompt (assets/prompt.txt).
uv run flashdreams-run self-forcing-wan2.1-t2v-1.3b --total-blocks 7

# Inline prompt override.
uv run flashdreams-run self-forcing-wan2.1-t2v-1.3b \
    --prompt "A cat surfing." --total-blocks 7

# Path override (any .txt; first non-empty line is used as the prompt).
uv run flashdreams-run self-forcing-wan2.1-t2v-1.3b \
    --prompt /path/to/my_prompt.txt --total-blocks 7
```

Multi-GPU via context-parallelism:

```bash
# e.g. 4GPUs
uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
    self-forcing-wan2.1-t2v-1.3b --total-blocks 7
```

## Programmatic access

Access via runner.
```python
from self_forcing.config import RUNNER_WAN21_T2V_1PT3B as runner_config
from flashdreams.infra.config import derive_config

# set a new prompt
cfg = derive_config(runner_config, prompt="This is a new prompt")
runner = cfg.setup()
runner.run()
```

Access via pipeline.
```python
import torch
from self_forcing.config import PIPELINE_WAN21_T2V_1PT3B as pipeline_config

pipeline = pipeline_config.setup().to("cuda").eval()

sp = pipeline.decoder.spatial_compression_ratio
cache = pipeline.initialize_cache(
    text=["This is a new prompt"], # set a new prompt
    height=480 // sp, # latent height for DiT
    width=832 // sp, # latent width for DiT
)

total_blocks: int = 7
generated_chunks: list[torch.Tensor] = []
for i in range(total_blocks):
    video_chunk = pipeline.generate(autoregressive_index=i, cache=cache)
    pipeline.finalize(autoregressive_index=i, cache=cache) # update KV cache
    generated_chunks.append(video_chunk.cpu()) # each chunk is [T, C, H, W]
```

## Tests

```bash
uv run --extra dev pytest integrations/self_forcing/tests
```
