# flashdreams-causal-forcing

Causal-Forcing chunkwise / framewise streaming T2V + I2V inference for Wan 2.1 1.3B,
packaged as a [`flashdreams`](../..) plugin, in a standalone repo.

This is a worked example of the
[Adding a new recipe](../../docs/source/developer_guides/new_recipes.rst)
developer-guide flow.

## Shipped slugs

| slug | description |
| --- | --- |
| `causal-forcing-wan2.1-t2v-1.3b-chunkwise` | Causal-Forcing chunkwise Wan 2.1 1.3B T2V (`len_t=3`). |
| `causal-forcing-wan2.1-t2v-1.3b-framewise` | Causal-Forcing framewise Wan 2.1 1.3B T2V (`len_t=1`). |
| `causal-forcing-wan2.1-i2v-1.3b-framewise` | Causal-Forcing framewise Wan 2.1 1.3B I2V (`len_t=1`). |

The I2V slug defaults to `DEFAULT_I2V_IMAGE_URL`, which is downloaded
on first use into `~/.cache/flashdreams/i2v_images/` (honors
`FLASHDREAMS_CACHE_DIR`) and reused thereafter; override with
`--image-path /path/to/frame.png` for a local file. Causal-Forcing only
releases a framewise I2V checkpoint,
so there is no chunkwise I2V counterpart.

## Install

The plugin is registered as a `uv` workspace member in the repo-root
`pyproject.toml`, so a single `uv sync` from the repo root pulls it in:

```bash
uv sync
```

Standalone (outside the workspace) also works:

```bash
uv pip install -e integrations/causal_forcing
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
# List every registered runner (this plugin's slugs appear under "causal-forcing-*").
uv run flashdreams-run --help

# Per-runner help: every overridable field is a CLI flag.
uv run flashdreams-run causal-forcing-wan2.1-t2v-1.3b-framewise --help

# Single-GPU T2V run with the inline demo prompt (DEFAULT_T2V_PROMPT).
uv run flashdreams-run causal-forcing-wan2.1-t2v-1.3b-framewise --total-blocks 21

# Inline prompt override.
uv run flashdreams-run causal-forcing-wan2.1-t2v-1.3b-framewise \
    --prompt "A cat surfing." --total-blocks 21

# Path override (any .txt; first non-empty line is used as the prompt).
uv run flashdreams-run causal-forcing-wan2.1-t2v-1.3b-framewise \
    --prompt /path/to/my_prompt.txt --total-blocks 21

# I2V: defaults to DEFAULT_I2V_PROMPT + DEFAULT_I2V_IMAGE_URL (the latter
# is downloaded once into ~/.cache/flashdreams/i2v_images/ and reused;
# honors FLASHDREAMS_CACHE_DIR).
# Override either with --prompt "..." --image-path /path/to/frame.png.
uv run flashdreams-run causal-forcing-wan2.1-i2v-1.3b-framewise --total-blocks 21
```

Multi-GPU via context-parallelism:

```bash
# e.g. 4GPUs
uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
    causal-forcing-wan2.1-t2v-1.3b-framewise --total-blocks 21
```

## Programmatic access

Access via runner.
```python
from causal_forcing.config import RUNNER_WAN21_T2V_1PT3B_FRAMEWISE as runner_config
from flashdreams.infra.config import derive_config

# set a new prompt
cfg = derive_config(runner_config, prompt="This is a new prompt")
runner = cfg.setup()
runner.run()
```

Access via pipeline.
```python
import torch
from causal_forcing.config import PIPELINE_WAN21_T2V_1PT3B_FRAMEWISE as pipeline_config

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
    pipeline.finalize(autoregressive_index=i, cache=cache) # advance streaming caches
    generated_chunks.append(video_chunk.cpu()) # each chunk is [T, C, H, W]
```

## Tests

```bash
uv run --extra dev pytest integrations/causal_forcing/tests
```
