---
title: 'Demo API'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Use the Demo API to build user-facing applications, manage sessions, expose
controls, route inputs and outputs, configure a model-backed demo, and launch it
through `flashdreams-run-v2`.

The current application protocol is `flashdreams.api_v2`. The separate
`flashdreams.runtime.demo` API is experimental and provides higher-level
scenario, presentation, replay, warmup, and benchmark utilities.

## Guides

- [Create a demo](guides/create_demo.md)
- [Integrate a model with a demo](guides/integrate_model.md)
- [Demo configuration](guides/configuration.md)
- [Application slugs and model adapter dispatch](guides/application_slugs.md)
- [Interactive serving](guides/interactive_serving.md)
- [Local benchmarks](guides/local_benchmarks.md)

## API Reference

- [Demo Application API](api_reference/application.md)
