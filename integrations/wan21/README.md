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

## Shipped slugs

| slug | description |
| --- | --- |
| `wan21-t2v-1.3b-480p` | Wan 2.1 T2V 1.3B at 480p (single AR step, prompt-only). |
| `wan21-i2v-14b-480p` | Wan 2.1 I2V 14B at 480p (single AR step, prompt + first-frame). |

The higher-level T2V application is registered as `t2v-wan21` and uses the
same single-block bidirectional rollout.

## Install

The plugin is registered as a `uv` workspace member in the repo-root
`pyproject.toml`, so a single `uv sync` from the repo root pulls it in:

```bash
uv sync
```

Standalone (outside the workspace) also works:

```bash
uv pip install -e integrations/wan21
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

### T2V application

The simplest application launch uses the default local window and model
settings:

```bash
uv run --package flashdreams-wan21 flashdreams-run t2v-wan21 \
  --prompt "A cat surfing."
```

Per-application help:

```bash
uv run --package flashdreams-wan21 flashdreams-run t2v-wan21 --help
```

Generate an MP4 instead:

```bash
uv run --package flashdreams-wan21 flashdreams-run t2v-wan21 \
  --output mp4 \
  --output-path artifacts/t2v-wan21.mp4 \
  --prompt "A cat surfing."
```

Serve the same application through WebRTC:

```bash
uv run --package flashdreams-wan21 flashdreams-run t2v-wan21 \
  --output webrtc \
  --host 0.0.0.0 \
  --port 8080 \
  --prompt "A cat surfing."
```

Then open `http://localhost:8080/request_session`. See the
[shared T2V application guide](../../apps/t2v/README.md) for all output modes.

### Legacy runners

The legacy runner slugs are also discovered automatically by
`flashdreams-run`:

```bash
# List every registered runner (this plugin's slugs appear under "wan21-*").
uv run flashdreams-run --help

# Per-runner help: every overridable field is a CLI flag.
uv run flashdreams-run wan21-t2v-1.3b-480p --help

# Single-GPU T2V run with the inline demo prompt (DEFAULT_T2V_PROMPT).
uv run flashdreams-run wan21-t2v-1.3b-480p

# Inline prompt override.
uv run flashdreams-run wan21-t2v-1.3b-480p --prompt "A cat surfing."

# Path override (any .txt; first non-empty line is used as the prompt).
uv run flashdreams-run wan21-t2v-1.3b-480p --prompt /path/to/my_prompt.txt

# I2V defaults to DEFAULT_I2V_PROMPT + DEFAULT_I2V_IMAGE_URL (the latter
# is downloaded once into ~/.cache/flashdreams/wan21/ and reused;
# honors FLASHDREAMS_CACHE_DIR).
uv run flashdreams-run wan21-i2v-14b-480p

# I2V override with custom prompt + image.
uv run flashdreams-run wan21-i2v-14b-480p \
    --prompt "A reindeer in cinematic sunset light." \
    --image-path /path/to/frame.png
```

Multi-GPU via context-parallelism:

```bash
# e.g. 4GPUs
uv run torchrun --nproc_per_node=4 --no-python flashdreams-run \
    wan21-t2v-1.3b-480p
```

## Programmatic access

Access via runner.
```python
from flashdreams.infra.config import derive_config
from wan21.config import RUNNER_WAN21_T2V_1PT3B_480P as runner_config

# set a new prompt
cfg = derive_config(runner_config, prompt="This is a new prompt")
runner = cfg.setup()
runner.run()
```

Access via pipeline.
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
uv run --extra dev pytest integrations/wan21/tests
```
