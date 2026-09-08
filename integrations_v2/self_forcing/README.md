# flashdreams-self-forcing

Self-Forcing distilled streaming T2V inference for Wan 2.1 1.3B,
packaged as a [`flashdreams`](../..) plugin, in a standalone repo.

This is a worked example of the
[Add a new method](https://nvidia.github.io/flashdreams/main/developer_guides/new_integration.html)
developer-guide flow.

## Shipped pipeline configs

| config name | description |
| --- | --- |
| `self-forcing-wan2.1-t2v-1.3b` | Self-Forcing distilled Wan 2.1 1.3B T2V (Wan VAE decoder, 4-step). |
| `self-forcing-wan2.1-t2v-1.3b-taehv` | Same DiT, swapped to the TAEHV (LightTAE) decoder for faster decoding. |
| `self-forcing-wan2.1-t2v-1.3b-sink5-window7-rerope` | Long-rollout preset with static sink=5 + window=7 + KVCache-relative RoPE. |

## Application integrations

| application slug | pipeline config |
| --- | --- |
| `t2v-self-forcing-wan2.1-t2v-1.3b` | `self-forcing-wan2.1-t2v-1.3b` |
| `t2v-self-forcing-wan2.1-t2v-1.3b-taehv` | `self-forcing-wan2.1-t2v-1.3b-taehv` |
| `t2v-self-forcing-wan2.1-t2v-1.3b-sink5-window7-rerope` | `self-forcing-wan2.1-t2v-1.3b-sink5-window7-rerope` |

## Install

The plugin is registered as a `uv` workspace member in the repo-root
`pyproject.toml`, so a single `uv sync` from the repo root pulls it in:

```bash
uv sync
```

Standalone (outside the workspace) also works:

```bash
uv pip install -e integrations_v2/self_forcing
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

```bash
uv run --package flashdreams-self-forcing flashdreams-run-v2 \
  t2v-self-forcing-wan2.1-t2v-1.3b --output-path artifacts/t2v-self-forcing-wan2.1-t2v-1.3b.mp4 -- \
  --prompt "A cat surfing." --total-blocks 7 --no-compile
```

See [apps/t2v/README.md](apps/t2v/README.md) for the concise launch command
and [the shared T2V guide](../../apps/t2v/README.md) for common arguments.

## Programmatic access

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
uv run --extra dev pytest integrations_v2/self_forcing/tests
```
