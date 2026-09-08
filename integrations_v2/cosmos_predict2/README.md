# flashdreams-cosmos-predict2

Cosmos-Predict2 bidirectional T2V and I2V inference,
packaged as a [`flashdreams`](../..) plugin, in a standalone repo.

This is a worked example of the
[Add a new method](https://nvidia.github.io/flashdreams/main/developer_guides/new_integration.html)
developer-guide flow.

Cosmos-Predict2 is **bidirectional**: it generates the complete clip in one
rollout instead of advancing through multiple causal blocks. It therefore
requires exactly one block (`--total-blocks 1`); multi-block generation is not
supported.

## Shipped pipeline configs

| config name | description |
| --- | --- |
| `cosmos2-t2v-2b-720p` | Cosmos-Predict2 2B T2V at 720p (single AR step, prompt-only). |
| `cosmos2-i2v-2b-720p` | Cosmos-Predict2 2B I2V at 720p (single AR step, prompt + first-frame image). |

## Application integrations

| application slug | pipeline config |
| --- | --- |
| `t2v-cosmos2-t2v-2b-720p` | `cosmos2-t2v-2b-720p` |

The I2V config remains available for direct pipeline use.

## Install

The plugin is registered as a `uv` workspace member in the repo-root
`pyproject.toml`, so a single `uv sync` from the repo root pulls it in:

```bash
uv sync
```

Standalone (outside the workspace) also works:

```bash
uv pip install -e integrations_v2/cosmos_predict2
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
uv run --package flashdreams-cosmos-predict2 flashdreams-run-v2 \
  t2v-cosmos2-t2v-2b-720p --output-path artifacts/t2v-cosmos2-t2v-2b-720p.mp4 -- \
  --prompt "A cat surfing." --no-compile
```

See [apps/t2v/README.md](apps/t2v/README.md) for the concise launch command
and [the shared T2V guide](../../apps/t2v/README.md) for common arguments.

## Programmatic access

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
uv run --extra dev pytest integrations_v2/cosmos_predict2/tests
```
