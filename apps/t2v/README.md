<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlashDreams T2V application

`flashdreams-t2v` provides the reusable v2 application, session, model loop,
command-line arguments, and CPU test helpers for text-conditioned video models. Its importable source package lives in `apps/t2v/t2v`. Concrete model
packages own one root `config.py`; their entry points and adapters live under
`integrations_v2/<model>/apps/t2v`, with non-config model implementation under
`integrations_v2/<model>/impl`.

## Controls

An interactive WebRTC run shows an ImGui prompt field over the latest generated
frame. Submitting it starts a fresh session and rollout cache without unloading
the model. Reaching `--total-blocks` leaves the final frame and prompt UI active.
Model adapters may also add startup inputs such as a first-frame image. Use
`--seed` to make sampling repeatable when the model supports it.

## Usage

Install one model package, then launch its application. Application arguments
follow `--`:

```bash
uv run --package flashdreams-self-forcing flashdreams-run-v2 \
  t2v-self-forcing-wan2.1-t2v-1.3b --output-path clip.mp4 -- \
  --prompt "A cat surfing" --total-blocks 7 --no-compile
```

To keep the model resident and submit prompts from a browser:

```bash
uv run --package flashdreams-causal-forcing flashdreams-run-v2 \
  t2v-causal-forcing-wan2.1-t2v-1.3b-chunkwise --mode webrtc --port 8080
```

Open `http://localhost:8080`. An initial `--prompt` is optional for interactive
runs; the prompt UI can start the first and subsequent sessions.

Shared application arguments are:

- `--prompt TEXT` — optional initial text; explicitly empty text is rejected.
- `--total-blocks N` — autoregressive blocks to generate; defaults to the model
  adapter value (one for bidirectional models).
- `--device DEVICE` — model device; defaults to `cuda`.
- `--compile` / `--no-compile` — override model compilation; defaults to the
  selected pipeline config.
- `--seed N` — override the diffusion seed; defaults to the selected pipeline
  config.

Run `flashdreams-run-v2 <slug> -- --help` for model-specific additions.

The default application name is `<demo-slug>-<model-slug>`. Additional model
configurations may append a suffix such as `-fast` and use a matching
`create_app_fast` factory. Current examples include
`t2v-causal-forcing-wan2.1-t2v-1.3b-framewise`,
`t2v-self-forcing-wan2.1-t2v-1.3b-taehv`, and `t2v-wan22-ti2v-5b`.

## Tests

```bash
uv sync --package flashdreams-t2v --extra dev --no-default-groups --inexact
uv run --no-sync pytest apps/t2v/tests -m ci_cpu
```

Model-specific tests live under `integrations_v2/<model>/tests`.
