# FastVideo causal Wan2.2 T2V parity check

Self-contained benchmark of upstream [FastVideo](https://github.com/hao-ai-lab/FastVideo)
for the self-forcing causal Wan2.2 text-to-video (T2V) path, aligned with
flashdreams parity conventions.

This harness:
- clones upstream FastVideo at a pinned commit,
- applies `changes.patch`,
- uses an isolated `uv` env with FastVideo-compatible deps, then installs local
  `flashdreams` (no-deps) so the benchmark can import `flashdreams.infra.profiler`,
- runs a benchmark script that emits parity-style per-block timing JSON.

## Run

From this directory:

```bash
bash run.sh
```

The script is idempotent and skips clone/checkout/patch/dependency setup when already satisfied.

## Outputs

Written under `FastVideo/`:
- `videos/offline*.mp4` - generated video(s)
- `videos/stats_offline.json` - parity-style timing JSON per autoregressive block (`denoise_ms`, `kv_update_ms`, `decode_ms`, `total_ms`, `total_ms_wo_finalize`)

## Backend and speed settings

The benchmark runs with:
- CPU offload disabled (`dit/text_encoder/vae` offload all `False`)
- `--enable_torch_compile` enabled
- `FASTVIDEO_ATTENTION_BACKEND=TORCH_SDPA`
- `FASTVIDEO_FORCE_CUDNN_SDPA=1`

This enforces a fast-path configuration and applies strict cuDNN SDPA forcing
inside FastVideo's SDPA backend when the env flag is set.

`flashdreams` is installed with `uv pip install --no-deps -e ...` to expose the
profiler module without forcing flashdreams' full dependency set into this
FastVideo parity environment.

## Notes

The benchmark uses the default prompt text from FastVideo's
`basic_self_forcing_causal_wan2_2_t2v.py` example.

## Files tracked here

- `run.sh` - clone + setup + patch + benchmark runner
- `pyproject.toml` - isolated venv definition
- `changes.patch` - local edits on top of pinned FastVideo commit
- `.gitignore` - ignores local clone and venv
