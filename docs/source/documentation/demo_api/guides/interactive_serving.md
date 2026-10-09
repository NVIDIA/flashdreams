---
title: 'Interactive serving'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

`flashdreams-run-v2` can present a compatible application in a browser or a
local native window. Applications generate frames and consume input events;
`flashdreams.runtime_v2` owns transport, pacing, and window lifecycle.

## Launch a client

Serve a browser on the local machine:

```bash
uv run --no-sync flashdreams-run-v2 APPLICATION_SLUG \
    --mode webrtc --host 127.0.0.1 --port 0
```

Port `0` selects an available port and prints the URL. Use `0.0.0.0` only when
remote clients must connect; it exposes the server on available network
interfaces, so apply the host's normal firewall and access controls.

Open a GPU-backed local window instead:

```bash
uv run --no-sync flashdreams-run-v2 APPLICATION_SLUG \
    --mode native-window --window-title FlashDreams
```

Runtime options precede `--`. Application-specific options follow it:

```bash
uv run --no-sync flashdreams-run-v2 cam2v-lingbot \
    --mode webrtc --host 127.0.0.1 --port 8089 -- \
    --example-data --total-blocks 21
```

Inspect both sides before launching:

```bash
uv run --no-sync flashdreams-run-v2 --help
uv run --no-sync flashdreams-run-v2 cam2v-lingbot -- --help
```

## Understand the boundary

- `IApplication` owns shared model state and application arguments.
- `ISession` owns one run and registers model and optional UI loops.
- The model loop receives `UserInputEvents` and publishes `StepResult` channels.
- The UI loop composites frames and controls; without one, the runtime uses its
  default blit loop.
- The selected client-window mode owns browser, file, or native presentation.

The model and UI loops run on separate threads. They must not mutate each
other's state directly; use `flashdreams.api_v2.loop.invoke_async`. Keep camera,
action, and conditioning semantics in the app or adapter rather than teaching
the WebRTC transport about a model.

## Distributed applications

Launch all model ranks through the same command. Rank zero owns the client
window and output sinks; the session's parallel context synchronizes admitted
steps and input across ranks.

```bash
uv run torchrun --nproc_per_node=2 --no-python flashdreams-run-v2 \
    cam2v-lingbot --mode webrtc -- \
    --example-data --total-blocks 21
```

The integration remains responsible for its tensor/context parallelism and for
gathering any output required by the presenting rank.

Use [Create a demo](create_demo.md) when the interaction needs new session or UI
behavior. Use [Integrate a model with a demo](integrate_model.md) when the
interaction already exists. The [CLI reference](../../cli.md) documents common
runtime and presentation options.
