---
name: flashdreams-run-model
description: Select, install, launch, and troubleshoot an existing FlashDreams video or world-model demo using the V2 CLI. Use for model/demo presets, browser or file output, runtime versus application arguments, and first-run failures. Model porting, new demo development, and offline executable packaging are separate workflows.
license: Apache-2.0
metadata:
  author: "FlashDreams Team <flashdreams@nvidia.com>"
---

# Run a FlashDreams model

## Locate the product and choose a supported preset

Use the user's FlashDreams checkout and requested version. If this skill was
installed separately, locate that checkout; when setup is requested and none
exists, clone `https://github.com/NVIDIA/flashdreams.git` into a new directory.
Commands below run at the repository root. For command-only questions, answer
without installing packages or launching inference.

A model pipeline config is not necessarily a runnable demo. Choose a compatible
demo/model preset from `docs/source/models/<model>.md` and check its registration
under `project.entry-points."flashdreams.applications_v2"` in
`integrations_v2/<model>/pyproject.toml`. The model page owns install commands,
hardware requirements, checkpoints and presets; `docs/source/demos/<demo>.md`
owns controls and application arguments. Do not invent a preset by joining names.

Check OS, GPU/VRAM, CUDA/toolchain, free disk, and model access against the chosen
model page. Requirements differ by model; the smallest parameter count does not
guarantee a small VRAM requirement. Keep credentials in the user's environment,
not in commands printed with their values or in generated files.

## Install only the selected integration

Read the distribution name and optional extras from its `pyproject.toml`:

```bash
uv sync --package flashdreams-wan21 --inexact
uv run --no-sync flashdreams-run-v2 --help
uv run --no-sync flashdreams-run-v2 t2v-wan21-t2v-1.3b-480p -- --help
```

The first help command lists the runtime interface; the second requests the
application's parser. Help does not prove checkpoint access or GPU execution.
Installing dependencies may build CUDA extensions. On a CPU-only inspection
task, read manifests and source rather than synchronizing the GPU workspace.
After changing entry points, resynchronize the selected package.

## Construct the V2 command

The shape is `flashdreams-run-v2 PRESET SYSTEM_ARGS -- APP_ARGS`.

- Runtime arguments before `--`: `--mode`, `--output-path`, `--timeout`,
  `--total-model-steps`, `--host`, `--port`, `--stats-path`, session overrides.
- Application arguments after `--`: prompts, example data, device, model-specific
  block limits and controls. Confirm spelling with that application's help.
- `mp4` and `webm` require an output path and a time or model-step limit;
  `unbound` explicitly opts out of a limit. These limits span session replacement.
- `webrtc` serves a browser; `native-window` opens a local window. Use loopback
  by default. Bind `0.0.0.0` only when remote access is part of the request.
- To retain every generated frame for file output, use `--backpressure-mode
  block --presentation-mode on_demand`. Interactive latency may justify different
  pacing; do not promise that `drop_oldest` preserves all frames.

For example, a bounded Wan 2.1 file run:

```bash
uv run --no-sync flashdreams-run-v2 t2v-wan21-t2v-1.3b-480p \
  --mode mp4 --output-path artifacts/wan21.mp4 --total-model-steps 1 \
  --backpressure-mode block --presentation-mode on_demand \
  -- --prompt "A cat surfing"
```

Use `flashdreams-run-v2` for current applications. The legacy
`flashdreams-run --no-instantiate` resolves V1 runner configs and is not a V2
dry-run flag. `--preload-application` is also not a CPU dry run: it initializes
resources and, by default, executes one model block.

## Diagnose before retrying

| Symptom | Next check |
| --- | --- |
| Unknown preset | Installed integration, its V2 entry point, and package resync. |
| Unrecognized argument | Runtime/application side of `--` and application help. |
| Missing output or run limit | Explicit mode, output path, timeout/model-step policy. |
| Asset access failure | Requested repository/revision and user's access; never print the token. |
| Slow first launch | Download/compile/autotune progress; separate cold setup from steady state. |
| Out of memory or native build failure | Model-specific VRAM, supported GPU, driver and PyTorch/CUDA toolchain; avoid speculative global dependency upgrades. |

Report the exact command, observed result and remaining uncertainty. A successful
parser check is not a successful rollout. For an authorized runtime check, verify
the file or browser output and retain logs; use `--stats-path` when measurements
are requested, noting its synchronized profiling overhead.

## References

- [Models and presets](https://flashdreams.org/main/models/)
- [Demo controls](https://flashdreams.org/main/demos/)
- [CLI](https://flashdreams.org/main/documentation/cli/)
- [Troubleshooting](https://flashdreams.org/main/troubleshooting/)
