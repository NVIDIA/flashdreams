# flashdreams-wan21

Wan 2.1 bidirectional T2V + I2V inference,
packaged as a [`flashdreams`](../..) plugin, in a standalone repo.

This is a worked example of the
[Add a new method](https://nvidia.github.io/flashdreams/main/developer_guides/new_integration.html)
developer-guide flow.

Wan 2.1 is **bidirectional**: it generates the complete clip in one rollout
instead of advancing through multiple causal blocks. It therefore requires
exactly one block (`--total-blocks 1`); multi-block generation is not
supported.

## Shipped pipeline configs

| config name | description |
| --- | --- |
| `wan21-t2v-1.3b-480p` | Wan 2.1 T2V 1.3B at 480p (single AR step, prompt-only). |
| `wan21-i2v-14b-480p` | Wan 2.1 I2V 14B at 480p (single AR step, prompt + first-frame). |

## Application integrations

| application slug | pipeline config |
| --- | --- |
| `t2v-wan21-t2v-1.3b-480p` | `wan21-t2v-1.3b-480p` |

The I2V config remains available for direct pipeline use.

## Install

The plugin is registered as a `uv` workspace member in the repo-root
`pyproject.toml`, so a single `uv sync` from the repo root pulls it in:

```bash
uv sync
```

Standalone (outside the workspace) also works:

```bash
uv pip install -e integrations_v2/wan21
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

Generate the model's single-block MP4 with the v2 application:

```bash
uv run --package flashdreams-wan21 flashdreams-run-v2 \
  t2v-wan21-t2v-1.3b-480p --output-path artifacts/t2v-wan21-t2v-1.3b-480p.mp4 -- \
  --prompt "A cat surfing." --no-compile
```

See the [application README](apps/t2v/README.md) for the concise launch command
and the [shared T2V guide](../../apps/t2v/README.md) for common arguments.

## Programmatic access

```python
from wan21.config import PIPELINE_WAN21_T2V_1PT3B_480P as pipeline_config

pipeline = pipeline_config.setup().to("cuda").eval()

sp = pipeline.decoder.spatial_compression_ratio
cache = pipeline.initialize_cache(
    text=["This is a new prompt"], # set a new prompt
    height=480 // sp, # latent height for DiT
    width=832 // sp, # latent width for DiT
)

video = pipeline.generate(autoregressive_index=0, cache=cache)
pipeline.finalize(autoregressive_index=0, cache=cache) # update one-step stats
```

## Tests

```bash
uv run --extra dev pytest integrations_v2/wan21/tests
```
