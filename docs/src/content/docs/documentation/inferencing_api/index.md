---
title: 'Inferencing API'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Use the Inferencing API to implement model components, pipeline configuration,
streaming generation, caching, and optimized execution behind a Demo API model
loop.

`flashdreams.runtime` provides the experimental model and session contracts.
`flashdreams.infra`, `flashdreams.recipes`, and `flashdreams.core` provide
the lower-level pipeline and model building blocks.

## Guides

- [Create a model](guides/create_model.md)
- [Stream inference pipeline](guides/stream_inference_pipeline.md)
- [Latency tuning](guides/latency_tuning.md)
- [Accelerated building blocks](guides/accelerated.md)

## API Reference

- [Inference Runtime API](api_reference/runtime.md)
- [Pipeline API](api_reference/pipeline.md)
- [Core API](api_reference/core.md)
