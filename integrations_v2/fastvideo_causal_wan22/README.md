# flashdreams-fastvideo-causal-wan22

FastVideo CausalWan 2.2 14B MoE distilled streaming T2V inference,
packaged as a [`flashdreams`](../..) plugin, in a standalone repo.

This is a worked example of the
[Add a new method](https://nvidia.github.io/flashdreams/main/developer_guides/new_integration.html)
developer-guide flow.

## Shipped pipeline configs

| config name | description |
| --- | --- |
| `fastvideo-causal-wan2.2-t2v-14b` | FastVideo CausalWan 2.2 14B MoE T2V (Wan VAE decoder, 8-step). |

## Application integrations

| application slug | pipeline config |
| --- | --- |
| `t2v-fastvideo-causal-wan2.2-t2v-14b` | `fastvideo-causal-wan2.2-t2v-14b` |

The two MoE branches share every Wan 2.1 14B knob and only differ by
checkpoint: `high_noise` runs above the boundary
(`timestep / num_train_timesteps >= boundary_ratio`), `low_noise` runs
below. T2V only -- the FastVideo Wan 2.2 checkpoint's I2V protocol
(one-shot first-frame VAE-seed warmup) does not fit the unified
streaming pipeline's per-AR-step mask-injection I2V and is not wired
here.

## Install

The plugin is registered as a `uv` workspace member in the repo-root
`pyproject.toml`, so a single `uv sync` from the repo root pulls it in:

```bash
uv sync
```

Standalone (outside the workspace) also works:

```bash
uv pip install -e integrations_v2/fastvideo_causal_wan22
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

Generate a seven-block MP4 with the v2 application:

```bash
uv run --package flashdreams-fastvideo-causal-wan22 flashdreams-run-v2 \
  t2v-fastvideo-causal-wan2.2-t2v-14b \
  --output-path artifacts/t2v-fastvideo-causal-wan2.2-t2v-14b.mp4 -- \
  --total-blocks 7 --prompt "A cat surfing." --no-compile
```

See the [application README](apps/t2v/README.md) for the concise launch command
and the [shared T2V guide](../../apps/t2v/README.md) for common arguments.

## Programmatic access

```python
import torch
from fastvideo_causal_wan22.config import PIPELINE_WAN22_T2V_14B as pipeline_config

pipeline = pipeline_config.setup().to("cuda").eval()

sp = pipeline.decoder.spatial_compression_ratio
cache = pipeline.initialize_cache(
    text=["This is a new prompt"], # set a new prompt
    height=480 // sp, # latent height for DiT
    width=832 // sp, # latent width for DiT
)

total_blocks: int = 21
generated_chunks: list[torch.Tensor] = []
for i in range(total_blocks):
    video_chunk = pipeline.generate(autoregressive_index=i, cache=cache)
    pipeline.finalize(autoregressive_index=i, cache=cache) # update KV cache
    generated_chunks.append(video_chunk.cpu()) # each chunk is [T, C, H, W]
```

## Tests

```bash
uv run --extra dev pytest integrations_v2/fastvideo_causal_wan22/tests
```
