<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

Text-to-video on the v2 API: one application every t2v model configures rather
than writes. A prompt goes in, an MP4 comes out.

## Run a model

```bash
export HF_TOKEN=<your-hf-token>
uv run --project integrations_v2/t2v_self_forcing flashdreams-run-v2 \
    t2v-self-forcing --output-path clip.mp4 \
    -- --prompt "A cat surfing" --total-blocks 7 --no-compile
```

The slug names the model, arguments before `--` describe the run, and arguments
after it go to the model. The first run downloads the checkpoint, which is tens
of gigabytes.

| Slug | Model | A run |
| --- | --- | --- |
| `t2v-self-forcing` | Self-Forcing Wan 2.1 1.3B, 480p | streams, 9 frames then 12 a block |
| `t2v-causal-forcing` | Causal-Forcing Wan 2.1 1.3B, 480p | streams, 9 frames then 12 a block |
| `t2v-fastvideo-causal-wan22` | CausalWan 2.2 14B, 480p | streams, two transformers |
| `t2v-wan21` | Wan 2.1 1.3B, 480p | one block, 81 frames |
| `t2v-cosmos-predict2` | Cosmos Predict2 2B, 720p | one block, 93 frames |

Each has a README of its own under `integrations_v2/`, with the frame
arithmetic for that model.

## Arguments

After the `--`, the same for every model, and listed by
`flashdreams-run-v2 SLUG -- --help`:

| Argument | |
| --- | --- |
| `--prompt` | Text to generate from. Required. |
| `--total-blocks` | Blocks to generate, which is how long the clip is. |
| `--device` | Device to load the model on. |
| `--compile` / `--no-compile` | Compile the network: minutes once, milliseconds a step. |
| `--seed` | Seed the noise, so the same command generates the same clip. |

Before the `--` are `flashdreams-run-v2`'s own, including `--output-path`,
`--pixel-width`, `--pixel-height`, and `--fps`. Unasked, a model generates at
the size and rate its checkpoint was trained for, which is what `session_desc()`
answers with.

## What is here

- `defaults.py`: what an integration contributes, read off the runner config it
  already ships so a model's frame size and rate are not written down twice.
- `application.py`: the shared command line above, and the model loaded once for
  every session. `_configure_argument_parser` and `_apply_parsed_arguments` are
  where a model adds a flag of its own.
- `session.py`: one rollout. The prompt is encoded into a cache, a block is
  generated per step, and the session reports itself finished when it has
  generated `--total-blocks` of them.
- `testing.py`: the check an integration's tests run, and the stand-in model
  they run it against on a CPU.

## Adding a model

Subclass `T2VApplication` with defaults from the integration's runner config,
which is all any of the five packages behind it are. Override
`_validate_total_blocks` for a model that generates its whole clip in one
bidirectional block, and `_apply_compile_override` for one whose transformers
the shared override does not reach.

Comparing the models against each other is
[`configs/v2_model_benchmarks.json`](../../../configs/v2_model_benchmarks.json),
and running it is [beside the harness](../../tools/benchmarks/README.md).
