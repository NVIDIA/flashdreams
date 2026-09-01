# flashdreams-causal-forcing

Causal-Forcing streaming inference for Wan 2.1 1.3B, packaged as a
[`flashdreams`](../..) integration.

## Install

The integration is a `uv` workspace member:

```bash
uv sync --package flashdreams-causal-forcing
```

Set `HF_TOKEN` if your Hugging Face access requires authentication. Checkpoints
are downloaded on first use and honor `HF_HOME`.

## Demo

Run the shared T2V application with a local window:

```bash
uv run --package flashdreams-causal-forcing flashdreams-run t2v-causal-forcing \
  --prompt "A robot walking through a forest."
```

Write the same rollout to MP4:

```bash
uv run --package flashdreams-causal-forcing flashdreams-run t2v-causal-forcing \
  --output mp4 --output-path artifacts/causal-forcing.mp4 \
  --prompt "A robot walking through a forest."
```

Use `--output webrtc --host 0.0.0.0 --port 8080` for the browser backend. The
application also accepts `--total-blocks`, `--pixel-height`, `--pixel-width`,
`--fps`, `--device`, and `--compile`.

## Pipeline presets

The package keeps three importable presets for lower-level use:

| preset | purpose |
| --- | --- |
| `PIPELINE_WAN21_T2V_1PT3B_CHUNKWISE` | Chunkwise T2V (`len_t=3`); used by the demo. |
| `PIPELINE_WAN21_T2V_1PT3B_FRAMEWISE` | Framewise T2V (`len_t=1`). |
| `PIPELINE_WAN21_I2V_1PT3B_FRAMEWISE` | Framewise I2V with first-frame conditioning. |

```python
from causal_forcing.config import PIPELINE_WAN21_T2V_1PT3B_FRAMEWISE

pipeline = PIPELINE_WAN21_T2V_1PT3B_FRAMEWISE.setup().to("cuda").eval()
ratio = pipeline.decoder.spatial_compression_ratio
cache = pipeline.initialize_cache(
    text=["A cat surfing."],
    image=None,
    height=480 // ratio,
    width=832 // ratio,
)

for index in range(21):
    video_chunk = pipeline.generate(autoregressive_index=index, cache=cache)
    pipeline.finalize(autoregressive_index=index, cache=cache)
```

## Tests

```bash
uv run --package flashdreams-causal-forcing --extra dev \
  pytest integrations/causal_forcing/tests
```
