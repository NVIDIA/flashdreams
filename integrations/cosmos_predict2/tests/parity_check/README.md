# Cosmos-Predict2.5 parity check

Self-contained reproducer of upstream
[`nvidia-cosmos/cosmos-predict2.5`](https://github.com/nvidia-cosmos/cosmos-predict2.5)
T2V + I2V base-model inference, used as a numerical / qualitative
reference for the in-tree `cosmos_predict2` flashdreams plugin.

## Run

From this directory — i.e.

```
/workspace/flashdreams/integrations/cosmos_predict2/tests/parity_check/
```

run:

```bash
bash run.sh
```

That's it. Idempotent: on first run it

1. clones `cosmos-predict2.5` at the pinned commit,
2. checks out that commit,
3. applies `changes.patch` on top (skipped if already applied),
4. re-fetches the three LFS-tracked input assets the cmds need,
5. materializes `cosmos-predict2.5/.venv/` via `uv sync --extra=cu130`,
6. runs the T2V inference cmd,
7. runs the I2V inference cmd.

Subsequent runs skip whatever's already in place and just re-run
inference.

## Outputs

Written under `cosmos-predict2.5/`:

- `outputs/base_text2world/` — T2V result for `assets/base/robot_welding.json`
- `outputs/base_image2world/` — I2V result for the same sample

Both cmds mirror the canonical reference snippet in
`data_local/cosmos-install.md`.

## Isolation

We use cosmos-predict2.5's own `pyproject.toml` verbatim — `uv sync
--extra=cu130` materializes the venv at `cosmos-predict2.5/.venv` and
pulls torch 2.9.1 + cu130 plus the matching prebuilt
`flash-attn` / `decord` / `transformer-engine` / `natten` wheels from
NVIDIA's custom index. None of those wheels are ABI-compatible with
flashdreams' `torch>=2.11` pin, so we deliberately keep this venv
independent of flashdreams' rather than stacking on top of it. The
in-tree `cosmos_predict2` flashdreams integration is exercised separately
via the main `flashdreams/.venv`.

## Files tracked here

- `README.md` — this file
- `run.sh` — clone + checkout + patch + LFS-fetch + `uv sync` + run T2V + run I2V, idempotent
- `changes.patch` — local edits layered on top of the pinned upstream commit:
  - swap `misc.arch_invariant_rand` for a `torch.randn` + permute that
    matches flashdreams' `[B,T,C,H,W]` RNG layout in both
    `text2world_model.py` and `text2world_model_rectified_flow.py`,
  - default `seed: 0 -> 42`
- `.gitignore` — ignores the cloned `cosmos-predict2.5/` tree (which
  carries its own `.venv/`, lockfile, outputs, etc.)
