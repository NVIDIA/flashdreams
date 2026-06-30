# flashdreams-cosmos-predict2

Cosmos-Predict2 bidirectional T2V and I2V inference,
packaged as a [`flashdreams`](../..) plugin, in a standalone repo.

This is a worked example of the
[Add a new method](https://nvidia.github.io/flashdreams/main/developer_guides/new_integration.html)
developer-guide flow.

**In this plugin, bidirectional video generation is treated as a 1-rollout (large-windowed) causal rollout.**

## Shipped slugs

| slug | description |
| --- | --- |
| `cosmos2-t2v-2b-720p` | Cosmos-Predict2 2B T2V at 720p (single AR step, prompt-only). |
| `cosmos2-i2v-2b-720p` | Cosmos-Predict2 2B I2V at 720p (single AR step, prompt + first-frame image). |

## Install

The plugin is registered as a `uv` workspace member in the repo-root
`pyproject.toml`, so a single `uv sync` from the repo root pulls it in:

```bash
uv sync
```

Standalone (outside the workspace) also works:

```bash
uv pip install -e integrations/cosmos_predict2
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
# List every registered runner (this plugin's slugs appear under "cosmos2-*").
uv run flashdreams-run --help

# Per-runner help: every overridable field is a CLI flag.
uv run flashdreams-run cosmos2-t2v-2b-720p --help

# Single-GPU T2V run with the inline demo prompt (DEFAULT_T2V_PROMPT).
uv run flashdreams-run cosmos2-t2v-2b-720p

# Inline prompt override.
uv run flashdreams-run cosmos2-t2v-2b-720p --prompt "A cat surfing."

# I2V run with a local first-frame image.
uv run flashdreams-run cosmos2-i2v-2b-720p \
    --image-path /path/to/first_frame.png \
    --prompt "A cat surfing."

# Path override (any .txt; first non-empty line is used as the prompt).
uv run flashdreams-run cosmos2-t2v-2b-720p --prompt /path/to/my_prompt.txt
```

Multi-GPU via context-parallelism:

```bash
# e.g. 4 GPUs
uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
    cosmos2-t2v-2b-720p
```

## Programmatic access

Access via runner.
```python
from flashdreams.infra.config import derive_config
from cosmos_predict2.config import RUNNER_COSMOS2_I2V_2B_720P as runner_config

# set a new prompt and first-frame image
cfg = derive_config(
    runner_config,
    prompt="This is a new prompt",
    image_path="/path/to/first_frame.png",
)
runner = cfg.setup()
runner.run()
```

Access via pipeline.
```python
from cosmos_predict2.config import PIPELINE_COSMOS2_T2V_2B_720P as pipeline_config

pipeline = pipeline_config.setup().to("cuda").eval()

sp = pipeline.decoder.spatial_compression_ratio
cache = pipeline.initialize_cache(
    text=["This is a new prompt"], # set a new prompt
    height=720 // sp, # latent height for DiT
    width=1280 // sp, # latent width for DiT
)

video = pipeline.generate(autoregressive_index=0, cache=cache)
pipeline.finalize(autoregressive_index=0, cache=cache) # update one-step stats
```

## Tests

```bash
uv run --extra dev pytest integrations/cosmos_predict2/tests
```
