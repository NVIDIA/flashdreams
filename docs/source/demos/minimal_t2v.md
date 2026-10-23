---
title: 'Minimal T2V'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Minimal T2V is the tutorial application built in [Create a demo](../documentation/demo_api/guides/create_demo.md).
It owns an ImGui prompt field, a model loop, and a per-prompt generation cache.

## Quick Start:

<div class="fd-cta-row" markdown="span">
  [Wan 2.1](../models/wan21.md#minimal-t2v){ .fd-button }
</div>

## Demo Arguments

Edit the prompt and click **Send** to start a new clip. Empty text pauses
generation. The window remains open between clips; close it to stop.

| App argument | Default | Purpose |
| --- | --- | --- |
| `--default-prompt` | Empty | Initial prompt; omit it to wait for input. |
| `--device` | `cuda` | Device on which to load the model. |
