---
title: 'Community'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

Use the issue tracker for work that needs a record. Use Discord for informal
questions and conversation. Do not report security problems in public.

<div class="fd-card-grid fd-card-grid-four fd-overview-grid">
  <a class="fd-card fd-model-card" href="https://github.com/NVIDIA/flashdreams/blob/main/CONTRIBUTING.md"><span class="fd-card-title">Contribute</span><span>Development workflow, testing requirements, and contribution policy.</span></a>
  <a class="fd-card fd-model-card" href="https://github.com/NVIDIA/flashdreams/issues"><span class="fd-card-title">Bugs and feature requests</span><span>Search existing reports or open an issue with a reproducible example.</span></a>
  <a class="fd-card fd-model-card" href="https://discord.gg/cMt2mHm4aN"><span class="fd-card-title">Discord</span><span>Ask questions and share results in #flashdreams on the NVIDIA Omniverse server.</span></a>
  <a class="fd-card fd-model-card" href="https://github.com/NVIDIA/flashdreams/blob/main/SECURITY.md"><span class="fd-card-title">Security</span><span>Follow the private disclosure process for vulnerabilities.</span></a>
</div>

## Before opening an issue

Check [Troubleshooting](../troubleshooting.md), the
[model page](../models/index.md), and existing issues first. Reduce the
problem to the shortest command you can. Include:

- what you ran and what you expected;
- the full error text, not a screenshot;
- FlashDreams, Python, CUDA, and GPU versions; and
- anything you already tried.

## Common questions

### What hardware do I need?

FlashDreams needs a recent NVIDIA GPU. Memory requirements depend on the
model, so check its [model page](../models/index.md) before installing it.

### Why are integrations not on PyPI?

Only the core `flashdreams` package is published to PyPI. Clone the
repository and install an integration as a workspace package. For example:

```bash

uv sync --package flashdreams-wan21 --extra dev

```

### Which Python API should I use?

Use `flashdreams.api_v2`; the
[Demo API reference](../documentation/demo_api/index.md) maps its
application, session, and loop contracts.

### How do I add a model?

Start with
[Create a model](../documentation/inferencing_api/guides/create_model.md), then
[integrate it with a demo](../documentation/demo_api/guides/integrate_model.md).

### Can I use FlashDreams commercially?

The Apache 2.0 license permits commercial use. Model weights and datasets may
have separate licenses.
