<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Runner slugs and demo launch dispatch

A runner slug is the short public name after `flashdreams-run`. It selects a
registered `RunnerConfig`; it does not need to duplicate the detailed name of
the pipeline preset inside that config.

## Quick start

After installing the OmniDreams workspace package, the default MP4 demo is:

```bash
uv sync --package flashdreams-omnidreams
uv run flashdreams-run omnidreams mp4
```

This uses the bundled single-view example data and writes
`outputs/omnidreams.mp4`. Use a launch manifest when you need to change the
scenario or output:

```bash
uv run flashdreams-run omnidreams mp4 \
  --manifest configs/launch_manifest/omnidreams_mp4.yaml
```

The shipped public OmniDreams runners are:

| Runner slug | Registered config literal | Internal pipeline preset |
| --- | --- | --- |
| `omnidreams` | `RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE` | `omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae` |
| `omnidreams-perf` | `RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF` | `omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf` |

The short slug is stable user interface. The internal pipeline name remains
specific enough for recipe selection, checkpoints, profiling, and direct demo
APIs.

## How the command is dispatched

For this command:

```bash
uv run flashdreams-run omnidreams mp4
```

the control flow is:

```text
flashdreams-run console script
  -> flashdreams.scripts.cli:entrypoint
  -> _prepare_cli_args(["omnidreams", "mp4"])
  -> all_runners()
       -> built-in runner registry
       -> discover_runners()
            -> importlib.metadata entry points
            -> group: flashdreams.runner_configs
  -> registry lookup by RunnerConfig.runner_name == "omnidreams"
  -> remove the positional mode "mp4" before Tyro parsing
  -> Tyro resolves runner flags over the registered RunnerConfig
  -> main(..., mode="mp4")
  -> resolve_launch()
  -> omnidreams.launch:LAUNCH_CAPABILITY
  -> OmnidreamsLaunchCapability.resolve()
  -> ResolvedLaunch.launch()
  -> omnidreams.demo.app.launch_from_runner()
```

Important details:

1. `_prepare_cli_args` finds the runner token by checking the keys returned by
   `all_runners()`. The next positional token is interpreted as a mode only
   when it is one of `run`, `mp4`, `null`, `webrtc`, or
   `local-window`.
2. A launch manifest must name the same runner and mode as the command. Its
   `runner_overrides` are applied before Tyro parses explicit CLI overrides.
3. `run` calls `config.setup()` and the regular runner. Other modes are
   delegated through the config's `launch_capability`.
4. The OmniDreams capability validates integration-specific scenario and
   output fields, then calls the shared demo API directly. It does not invoke a
   second CLI.
5. The demo derives its `preset_id` from `config.pipeline.name`. Therefore
   `omnidreams` still selects the detailed stable non-performance pipeline
   preset shown in the table.

## How a slug is registered

External integrations register runners with Python package entry points. The
OmniDreams package declares:

```toml
[project.entry-points."flashdreams.runner_configs"]
"omnidreams" = "omnidreams.config:RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE"
"omnidreams-perf" = "omnidreams.config:RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE_PERF"
```

Each target resolves to an `OmnidreamsRunnerConfig` literal:

```python
RUNNER_SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE = OmnidreamsRunnerConfig(
    runner_name="omnidreams",
    pipeline=SV_2STEPS_CHUNK2_LOC6_LIGHTVAE_LIGHTTAE,
    # ...
)
```

Discovery loads every entry point in the `flashdreams.runner_configs` group.
The registry key comes from `cfg.runner_name`, not from the entry-point name.
The entry-point name should still match `runner_name`: keeping both aligned
makes installed package metadata understandable, and the OmniDreams CPU tests
enforce that invariant.

Slug collisions are deterministic. Built-in runners take precedence over
plugins, and the first discovered plugin with a given `runner_name` wins over
later plugins. The registry logs skipped collisions.

## Adding or changing a public slug

1. Choose a short, integration-level default such as `omnidreams`. Add a
   concise suffix only when users need to select a materially different public
   variant, such as `omnidreams-perf`.
2. Set `runner_name` on the exported runner config literal.
3. Add an entry with the same name under
   `[project.entry-points."flashdreams.runner_configs"]`.
4. Keep `pipeline.name` unchanged unless the actual model recipe identity is
   changing.
5. Update launch manifests, CI commands, benchmark scenarios, and user-facing
   documentation. Do not rename checkpoint keys, asset paths, reference
   artifacts, or internal preset IDs merely because the public slug changed.
6. Refresh the editable package metadata and run CPU-only checks:

```bash
uv sync --package flashdreams-omnidreams --package flashdreams-lingbot \
  --group test
uv run flashdreams-run --help
uv run flashdreams-run omnidreams mp4 --no-instantiate
uv run pytest -m ci_cpu \
  integrations/omnidreams/tests/test_recipe_configs.py \
  integrations/omnidreams/tests/test_demo_api.py \
  flashdreams/tests/test_launch.py \
  flashdreams/tests/test_launch_manifest.py
```

The `--no-instantiate` check resolves registration, parsing, mode dispatch,
manifest validation, and default launch settings without loading checkpoints or
initializing the GPU.
