---
title: 'Community'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

Use the issue tracker for work that needs a record. Use Discord for informal
questions and conversation. Do not report security problems in public.

### [Contribute](https://github.com/NVIDIA/flashdreams/blob/main/CONTRIBUTING.md)

Development workflow, testing requirements, and contribution policy.

### [Bugs and feature requests](https://github.com/NVIDIA/flashdreams/issues)

Search existing issues, then open one with the command, expected result,
actual result, stack trace, and environment.

### [Discord](https://discord.gg/cMt2mHm4aN)

Ask open-ended questions and share results in `#flashdreams` on the
NVIDIA Omniverse server.

### [Security](https://github.com/NVIDIA/flashdreams/blob/main/SECURITY.md)

Follow the private disclosure process. Never file a public issue for a
vulnerability.

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
