# LTX-Video × FlashDreams

First-party FlashDreams integration for [LTX-Video](https://github.com/Lightricks/LTX-Video) 2B causal streaming text-to-video.

## Runner slugs

| Slug | Mode |
|------|------|
| `ltx-video-t2v-2b` | Streaming `LTXPipeline` wrapper (native `pipe()` per chunk) |
| `ltx-video-t2v-2b-optimized` | Manual denoise + KV-cache + `torch.compile` + FlashAttention |
| `ltx-video-t2v-2b-taehv` | Optimized path + TAEHV fast decoder |

## Install

```bash
# from the repo root
export HF_TOKEN=<your-hf-token>
uv sync --project integrations/ltx_video
```

## Run

```bash
uv run --project integrations/ltx_video \
    flashdreams-run ltx-video-t2v-2b \
    --prompt "A coastal road at dusk, waves breaking on rocky cliffs, cinematic wide shot" \
    --pixel-height 512 --pixel-width 768 \
    --total-blocks 7
```

Optimized path:

```bash
uv run --project integrations/ltx_video \
    flashdreams-run ltx-video-t2v-2b-optimized \
    --total-blocks 7
```

## Tests

```bash
uv run --project integrations/ltx_video pytest integrations/ltx_video/tests/test_smoke.py -v
uv run --project integrations/ltx_video pytest integrations/ltx_video/tests/test_optimizations.py -v -m ci_gpu
```

See the [LTX-Video model page](https://nvidia.github.io/flashdreams/main/models/ltx_video.html) for full setup notes.
