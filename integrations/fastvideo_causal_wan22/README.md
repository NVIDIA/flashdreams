# flashdreams-fastvideo-causal-wan22

FastVideo CausalWan 2.2 14B MoE distilled streaming T2V inference,
packaged as a [`flashdreams`](../..) plugin, in a standalone repo.

This is a worked example of the
[Add a new method](https://verbose-adventure-7plnn5m.pages.github.io/main/developer_guides/new_integration.html)
developer-guide flow.

## Shipped slugs

| slug | description |
| --- | --- |
| `fastvideo-causal-wan2.2-t2v-14b` | FastVideo CausalWan 2.2 14B MoE T2V (Wan VAE decoder, 8-step). |

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
uv pip install -e integrations/fastvideo_causal_wan22
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
# List every registered runner (this plugin's slug appears under "fastvideo-causal-wan2.2-*").
uv run flashdreams-run --help

# Per-runner help: every overridable field is a CLI flag.
uv run flashdreams-run fastvideo-causal-wan2.2-t2v-14b --help

# Single-GPU run with the inline demo prompt (DEFAULT_T2V_PROMPT).
uv run flashdreams-run fastvideo-causal-wan2.2-t2v-14b --total-blocks 21

# Inline prompt override.
uv run flashdreams-run fastvideo-causal-wan2.2-t2v-14b \
    --prompt "A cat surfing." --total-blocks 21

# Path override (any .txt; first non-empty line is used as the prompt).
uv run flashdreams-run fastvideo-causal-wan2.2-t2v-14b \
    --prompt /path/to/my_prompt.txt --total-blocks 21
```

Multi-GPU via context-parallelism:

```bash
# e.g. 4GPUs
uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
    fastvideo-causal-wan2.2-t2v-14b --total-blocks 21
```

## Programmatic access

Access via runner.
```python
from fastvideo_causal_wan22.config import RUNNER_WAN22_T2V_14B as runner_config
from flashdreams.infra.config import derive_config

# set a new prompt
cfg = derive_config(runner_config, prompt="This is a new prompt")
runner = cfg.setup()
runner.run()
```

Access via pipeline.
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
uv run --extra dev pytest integrations/fastvideo_causal_wan22/tests
```
