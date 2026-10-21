---
title: 'Interactive serving'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Run a demo in a browser or a local window with `flashdreams-run-v2`.
First install the integration using its [model page](../../../models/index.md).

## Browser

```bash
uv run --no-sync flashdreams-run-v2 APPLICATION_SLUG \
    --mode webrtc --host 127.0.0.1 --port 0
```

Open the URL printed by the server. Port `0` picks an available port.
To accept connections from another machine, use `--host 0.0.0.0` and a fixed
port, then connect to the server's address.

## Local window

```bash
uv run --no-sync flashdreams-run-v2 APPLICATION_SLUG --mode native-window
```

## Demo arguments

Runtime options go before `--`; demo options go after it:

```bash
uv run --no-sync flashdreams-run-v2 cam2v-lingbot \
    --mode webrtc --host 127.0.0.1 --port 8089 -- \
    --example-data --total-blocks 21
```

List runtime options with `flashdreams-run-v2 --help`. For this demo's options,
use `flashdreams-run-v2 cam2v-lingbot -- --help`.

## Input and display

The runtime sends user input to the app's loops and displays their output.
The app handles controls and model inputs. A custom UI loop can draw controls
over the generated frames; otherwise, the runtime displays the frames directly.

Model and UI loops run on separate threads. Use `invoke_async` to send changes
between them. [Create a demo](create_demo.md) shows an ImGui text field sending
a prompt to the model loop.

## Record a preview

Use an app that starts generating without menu or controller input. Supply its
prompt, scene, or recorded actions explicitly; recording an idle UI produces
no useful demonstration. For example, with the Wan 2.1 integration installed:

```bash
uv run --no-sync flashdreams-run-v2 t2v-wan21-t2v-1.3b-480p \
  --mode mp4 --output-path artifacts/t2v.mp4 \
  --total-model-steps 1 --timeout unbound \
  --backpressure-mode block --presentation-mode on_demand -- \
  --prompt "A cat surfing a wave" --no-ui
```

## Multiple GPUs

For a model adapter that supports distributed execution, launch all ranks:

```bash
uv run torchrun --nproc_per_node=2 --no-python flashdreams-run-v2 \
    cam2v-lingbot --mode webrtc -- --example-data --total-blocks 21
```

Rank zero handles the window and output. The session's parallel context
synchronizes steps and inputs; the model integration handles parallel inference
and gathers the output for rank zero.

See the [CLI reference](../../cli.md) for runtime options, or
[Integrate model adapter for a demo](integrate_model_adapter_for_a_demo.md) to connect another model.
