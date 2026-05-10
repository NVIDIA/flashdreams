# Self-Forcing parity check

Self-contained benchmark of upstream
[Self-Forcing](https://github.com/guandeh17/Self-Forcing) with a small local
patch (`changes.patch`) that adds `EventProfiler`-based per-block timing
and JSON stats output (mirroring `flashdreams`'s pipeline profiling). Note default we test with torch CUDNN attention backend.

## Run

From this directory — i.e.

```
/workspace/flashdreams/integrations/self_forcing/tests/parity_check/
```

run:

```bash
bash run.sh
```

That's it. The script is idempotent: on first run it clones upstream at a
pinned commit, downloads `Wan-AI/Wan2.1-T2V-1.3B` and the
`self_forcing_dmd.pt` checkpoint, applies `changes.patch`, and runs the
benchmark. Subsequent runs skip whatever's already in place and just
re-run the benchmark.

## Outputs

Written under `Self-Forcing/`:

- `videos/offline.mp4` — generated video
- `videos/stats_offline.json` — per-block timings (`denoise_ms`,
  `kv_update_ms`, `decode_ms`, `total_ms`, `total_ms_wo_finalize`) plus
  GPU memory stats, one entry per autoregressive block

## Isolation

Deps are pinned in this directory's `pyproject.toml` and live in
`./.venv/`. Because `uv run` walks upward looking for a project, calls
from inside `Self-Forcing/` resolve to *this* venv, not the surrounding
flashdreams one.

## Files tracked here

- `README.md` — this file
- `run.sh` — clone + setup + patch + benchmark, idempotent
- `pyproject.toml` — isolated venv definition (materialized via `uv sync`)
- `changes.patch` — local edits on top of the pinned upstream commit
  (`EventProfiler` timing, JSON stats dump, route attention through the
  `attention()` dispatcher so `FORCE_CUDNN_ATTN=1` works end-to-end)
- `.gitignore` — ignores the cloned `Self-Forcing/` tree and `./.venv/`
