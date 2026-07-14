# LingBot-World v2 parity check

Self-contained parity/benchmark harness for the upstream
[LingBot-World v2](https://github.com/robbyant/lingbot-world-v2) official
`generate.py` inference path.

The harness mirrors `integrations/lingbot/tests/parity_check`: it clones the
official repository at a pinned commit, applies a small local patch, downloads
the official Hugging Face checkpoint layout, and runs the upstream causal-fast
inference script from an isolated `uv` environment.

## Run

From this directory:

```bash
bash run.sh
```

By default the script runs a 1-GPU causal-fast job for the official example
`03`, using a shortened 81-frame rollout:

```bash
NPROC=1 EXAMPLE_IDX=03 FRAME_NUM=81 bash run.sh
```

For the official multi-GPU shape, set `NPROC=8`:

```bash
NPROC=8 EXAMPLE_IDX=03 FRAME_NUM=361 bash run.sh
```

When `NPROC > 1`, the script passes the official `--dit_fsdp`, `--t5_fsdp`,
and `--ulysses_size ${NPROC}` flags.

## Outputs

Written under `lingbot-world-v2/output/`:

- `lingbot-world-v2-parity-<example>-<nproc>gpu.mp4` - generated video.
- upstream logs from `generate.py`.

## Files Tracked Here

- `README.md` - this file.
- `run.sh` - clone, patch, checkpoint download, and upstream inference launch.
- `pyproject.toml` - isolated parity-check environment.
- `uv.lock` - pinned dependency resolution for the isolated environment.
- `changes.patch` - local patch applied to the upstream clone; it removes the
  upstream `pyproject.toml` so `uv run` resolves to this harness environment.
- `.gitignore` - ignores the cloned upstream tree and local venv.

## Notes

The official v2 checkpoint is large. The script downloads
`robbyant/lingbot-world-v2-14b-causal-fast` into the upstream clone and expects
CUDA-capable hardware with enough memory for the selected `NPROC` mode.
