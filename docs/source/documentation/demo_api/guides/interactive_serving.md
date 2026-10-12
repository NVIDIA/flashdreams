---
title: 'Interactive serving'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

`flashdreams-run-v2` can present any compatible `flashdreams.api_v2`
application in a browser. The application and model loop do not implement
network transport; `flashdreams.runtime_v2` supplies the WebRTC client window.

## Launch a browser session

```bash

uv run flashdreams-run-v2 APPLICATION_SLUG \
    --mode webrtc --host 0.0.0.0 --port 8089

```

Runtime options precede `--`. Application-specific arguments follow it:

```bash

uv run flashdreams-run-v2 cam2v-lingbot \
    --mode webrtc --host 0.0.0.0 --port 8089 \
    -- --example-data --total-blocks 21

```

Use `127.0.0.1` when only the local machine should connect. Port `0` selects an
available port and prints the resulting URL.

For a distributed model, launch the same application through `torchrun`:

```bash

uv run torchrun --nproc_per_node=2 --no-python flashdreams-run-v2 \
    cam2v-lingbot --mode webrtc \
    -- --example-data --total-blocks 21

```

## Ownership

- The application implements `IApplication`, creates `ISession` objects, and
  owns application arguments and shared model state.
- The model loop receives browser input as `UserInputEvents` and publishes
  `StepResult` channels.
- An optional UI loop reads presented model frames and renders controls or
  overlays.
- `flashdreams.runtime_v2.serving` owns HTTP, signaling, peer connections,
  browser events, video transport, and packaged web assets.
- `WebRTCClientWindow` implements the api_v2 client input/output contracts and
  stays open across compatible replacement sessions.

Browser keyboard, pointer, focus, query-string, reset, and close events are
translated into runtime event classes before either loop sees them. Keep model
conditioning and control semantics in the demo or adapter, not in the transport.

## Add interactive behavior

Use [Create a demo](create_demo.md) when a new interaction needs its own session
or UI behavior. Use [Integrate a model with a demo](integrate_model.md) when
the interaction already exists and only model defaults are missing.

The reusable references are:

- `apps/cam2v/` for live camera control;
- `apps/action2v/` for keyboard and pointer actions;
- `apps/interactive_drive/` for a larger simulation-driven application;
- `flashdreams/flashdreams/runtime_v2/serving/` for the transport implementation.

See the [CLI Reference](../../cli.md) for all runtime modes and the
[Demo Application API reference](../api_reference/application.md) for
application contracts.
