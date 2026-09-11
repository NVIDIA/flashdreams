<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlashDreams Action2V application

Reusable FlashDreams v2 infrastructure for world models conditioned by direct
keyboard and mouse actions. The shared package owns application/session
lifecycle, first-frame loading, live event accumulation, model cache and RNG
state, reset handling, generation, and presentation. Model integrations retain
their pipeline configuration, first-frame resolution, and action vocabulary mapping.

## Dummy demo

Use the packaged example image for a model-free keyboard and mouse demo:

```bash
uv run --package flashdreams-action2v flashdreams-run-v2 action2v-dummy \
  --mode webrtc -- --example-data
```

## Command-line options

Options after `--` belong to Action2V. Runtime options such as `--mode`,
`--output-path`, `--stats-path`, `--host`, and `--port` go before the
separator; run `flashdreams-run-v2 --help` for that list.

| Option | Default | Description |
| --- | --- | --- |
| `--image-path PATH` | unset | Image that establishes the initial world state. |
| `--example-data`, `--no-example-data` | off | Enable or disable the integration's example image. |
| `--device DEVICE` | integration default | Select the model device, normally `cuda`; the dummy defaults to `cpu`. |
| `--total-blocks N` | integration default | Stop after this many generated action blocks. |
| `--ui`, `--no-ui` | on | Enable or disable interactive pointer capture. |
| `--seed N` | pipeline default | Override the model RNG seed for reproducible generation. |
| `--mouse-sensitivity SCALE` | `1.0` | Multiply pointer motion by a finite, non-negative scale. |
| `--reset-key CHAR` | `T` | Reset the current session when this ASCII letter is pressed. |
| `-h`, `--help` | — | Print the application options without loading the model. |

## Integration contract

An adapter supplies an `Action2VApplicationDefaults` value with a pipeline config
and hooks that resolve the input image, load its display frames, and map a
model-neutral `ActionSnapshot` to the model action type. The shared application
owns session-local cache, RNG, generation, finalize, reset, and close behavior.

`ActionSnapshot` preserves held key and mouse-button state while mouse motion
and wheel deltas are consumed once per model action. Reset and focus loss clear
held state and pointer history.

## Tests

```bash
uv run --no-sync pytest apps/action2v/tests -m ci_cpu
```
