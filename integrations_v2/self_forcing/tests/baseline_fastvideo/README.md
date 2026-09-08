# FastVideo Self-Forcing parity check

Self-contained benchmark of upstream [FastVideo](https://github.com/hao-ai-lab/FastVideo) for
the Self-Forcing causal model path, aligned with flashdreams parity-check conventions.

This harness:
- clones upstream FastVideo at a pinned commit,
- applies `changes.patch`,
- uses an isolated `uv` env based on local `flashdreams` plus minimal extras,
- runs a benchmark script that emits parity-style per-block timing JSON.

## Run

From this directory:

```bash
bash run.sh
```

The script is idempotent and will skip clone/checkout/patch/dependency setup when already satisfied.

## Outputs

Written under `FastVideo/`:
- `videos/offline.mp4` - generated video
- `videos/stats_offline.json` - per-block timings with parity-style fields (`denoise_ms`, `kv_update_ms`, `decode_ms`, `total_ms`, `total_ms_wo_finalize`)

## Backend and speed settings

The benchmark runs with:
- CPU offload disabled (`dit/text_encoder/vae` offload all `False`)
- `--enable_torch_compile` enabled
- `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA`
- `FASTVIDEO_FORCE_CUDNN_SDPA=1`

so DiT timing is measured on a fast-path configuration focused on cuDNN SDPA.

## Files tracked here

- `run.sh` - clone + setup + patch + benchmark runner
- `pyproject.toml` - isolated venv definition
- `changes.patch` - local edits on top of pinned FastVideo commit
- `.gitignore` - ignores local clone and venv
