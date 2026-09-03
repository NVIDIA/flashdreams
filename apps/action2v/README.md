<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlashDreams Action2V application

Reusable FlashDreams v2 infrastructure for world models conditioned by direct
keyboard and mouse actions. The shared package owns application/session
lifecycle, first-frame loading, live event accumulation, reset
handling, and presentation. Model integrations retain action vocabulary mapping, cache,
RNG, and generation behavior.

## Dummy demo

For live keyboard and mouse input, point `--first-frame` at an image to select the initial world state:
```bash
uv run --package flashdreams-action2v flashdreams-run-v2 action2v-dummy \
  --mode webrtc -- --first-frame apps/action2v/action2v/assets/dummy_frame.ppm
```

## Command-line options

Options after `--` belong to Action2V. Runtime options such as `--mode`,
`--output-path`, `--stats-path`, `--host`, and `--port` go before the separator;
run `flashdreams-run-v2 --help` for that list.

| Option | Default | Description |
| --- | --- | --- |
| `--first-frame PATH` | integration default | Image that establishes the initial world state. Required when the adapter supplies no default. |
| `--seed N` | random | Set a non-negative model RNG seed for reproducible generation. |
| `--device DEVICE` | integration default | Select the model device, normally `cuda`; the dummy defaults to `cpu`. |
| `--profile` | off | Enable integration pipeline profiling. |
| `--mouse-sensitivity SCALE` | `1.0` | Multiply pointer motion by a finite, non-negative scale. |
| `-h`, `--help` | — | Print the application options without loading the model. |

Adapters may add model-specific options.

## Integration contract

An adapter supplies `Action2VApplicationDefaults` with four model-owned hooks:

- load the application-owned pipeline;
- load initial display frames;
- map a model-neutral `ActionSnapshot` to the model action type;
- build session-local cache/RNG/generation state.

`ActionSnapshot` preserves held key and mouse-button state while mouse motion
and wheel deltas are consumed once per model action. Reset and focus loss clear
held state and pointer history.

## Tests

```bash
uv run --no-sync pytest apps/action2v/tests -m ci_cpu
```
