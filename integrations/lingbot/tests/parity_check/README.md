# LingBot-World parity check

Self-contained benchmark of upstream
[LingBot-World](https://github.com/Robbyant/lingbot-world) with a small
local patch (`changes.patch`) that adds `EventProfiler`-based per-chunk
timing (mirroring `flashdreams`'s pipeline profiling) and routes the
attention call sites through the `attention()` dispatcher so
`FORCE_CUDNN_ATTN=1` works end-to-end. Note: default we test with the
torch cuDNN attention backend.

## Run

From this directory — i.e.

```
/workspace/flashsim/integrations/lingbot/tests/parity_check/
```

run:

```bash
bash run.sh
```

That's it. The script is idempotent: on first run it clones upstream at
a pinned commit, downloads `robbyant/lingbot-world-base-cam` and
`robbyant/lingbot-world-fast`, applies `changes.patch`, and runs the
benchmark. Subsequent runs skip whatever's already in place and just
re-run the benchmark.

## Outputs

Written under `lingbot-world/`:

- `output/*.mp4` — generated video
- per-chunk timings printed to stdout (`denoise_ms`, `kv_update_ms`,
  `total_ms`, `total_ms_wo_finalize`) plus GPU memory stats, one entry
  per autoregressive chunk

## Isolation

Deps are pinned in this directory's `pyproject.toml` and live in
`./.venv/`. Because `uv run` walks upward looking for a project, calls
from inside `lingbot-world/` resolve to *this* venv, not the surrounding
flashdreams one. The patch also deletes `lingbot-world/pyproject.toml`
so uv can't accidentally pick it up and try to install the upstream's
heavy deps (torch, flash_attn, ...).

## Files tracked here

- `README.md` — this file
- `run.sh` — clone + setup + patch + benchmark, idempotent
- `pyproject.toml` — isolated venv definition (materialized via `uv sync`)
- `changes.patch` — local edits on top of the pinned upstream commit
  (`EventProfiler` timing in `wan/image2video_fast.py`, route attention
  through the `attention()` dispatcher so `FORCE_CUDNN_ATTN=1` works
  end-to-end, drop the upstream `pyproject.toml`)
- `.gitignore` — ignores the cloned `lingbot-world/` tree and `./.venv/`
