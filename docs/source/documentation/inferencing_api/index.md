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

<div class="fd-card-grid fd-card-grid-three fd-overview-grid">
  <a class="fd-card fd-model-card" href="guides/create_model/"><span class="fd-card-title">Create a model</span><span>Implement and register a model integration with FlashDreams.</span></a>
  <a class="fd-card fd-model-card" href="guides/stream_inference_pipeline/"><span class="fd-card-title">Stream inference pipeline</span><span>Build stateful generation with streaming pipeline contracts.</span></a>
  <a class="fd-card fd-model-card" href="guides/latency_tuning/"><span class="fd-card-title">Latency tuning</span><span>Profile and reduce model, decode, transfer, and presentation latency.</span></a>
  <a class="fd-card fd-model-card" href="guides/accelerated/"><span class="fd-card-title">Accelerated building blocks</span><span>Use optimized attention, quantization, caching, and execution components.</span></a>
</div>

## API Reference

<div class="fd-card-grid fd-card-grid-three fd-overview-grid">
  <a class="fd-card fd-model-card" href="api_reference/runtime/"><span class="fd-card-title">Inference Runtime API</span><span>Reference for model loops, sessions, and runtime contracts.</span></a>
  <a class="fd-card fd-model-card" href="api_reference/pipeline/"><span class="fd-card-title">Pipeline API</span><span>Reference for inference pipelines and reusable infrastructure.</span></a>
  <a class="fd-card fd-model-card" href="api_reference/core/"><span class="fd-card-title">Core API</span><span>Reference for model-agnostic core components and utilities.</span></a>
</div>
