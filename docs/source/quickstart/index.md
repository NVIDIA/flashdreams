---
title: 'Get Started'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

Welcome to FlashDreams!

This page will guide you into running your first world-model.

We will use [NVIDIA OmniDreams](../models/omnidreams.md), an interactive driving world model, as the
example:

<div class="fd-card-grid">
  <div class="fd-card fd-model-card">
    <img class="fd-card-preview" src="../_static/demo_clips/interactive-drive.avif" alt="OmniDreams interactive driving demo." />
  </div>
</div>

## Install

First, clone FlashDreams:
```bash
git clone https://github.com/NVIDIA/flashdreams.git
cd flashdreams
```

Now we need to install `uv`, FlashDreams's package manager.
[Installation instructions here](https://docs.astral.sh/uv/getting-started/installation/).

With `uv` installed, clone the repository and synchronize `uv` to the OmniDreams workspace via:
```bash
uv sync --package flashdreams-omnidreams --extra interactive-drive --inexact
```

Most runs need a Hugging Face token for model/example-asset downloads. For OmniDreams, use a token with
read access to
[nvidia/omni-dreams-models](https://huggingface.co/nvidia/omni-dreams-models) and
[nvidia/omni-dreams-scenes](https://huggingface.co/datasets/nvidia/omni-dreams-scenes):

```bash

export HF_TOKEN=<your-hf-token>

```

## Run your first model

Launch the interactive driving demo in a browser over WebRTC:
```bash
uv run --no-sync flashdreams-run-v2 \
    interactive-drive-omnidreams --mode webrtc \
    --host 0.0.0.0 --port 8089
# Open in browser: http://<server-ip>:8089/
```

The first launch spends several minutes loading checkpoints and compiling kernels.

See our [How to Run a Model](../models/index.md#running-a-model) for more information on the CLI run command to tune the arguments to your liking.

- See our [Omnidreams Overview](../models/omnidreams.md) for a brief Omnidreams world-model overview.
- See our [Interactive Drive Overview](../demos/interactive_drive.md) for more information on the demo.

## Where to next

- [Model Gallery](../models/index.md): Every FlashDreams default-shipped model & how to run models!
- [Demo API guides](../documentation/demo_api/index.md): Create and configure
  demos, then integrate models with them.
- [Inferencing API guides](../documentation/inferencing_api/index.md): Add
  models and understand the stream inference pipeline.
- [Troubleshooting](../troubleshooting.md): Common first-run failures and fixes.
