---
title: 'Demo API'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Use the Demo API to build user-facing applications, manage sessions, expose
controls, route inputs and outputs, configure a model-backed demo, and launch it
through `flashdreams-run-v2`.

Application protocols live in `flashdreams.api_v2`, and `flashdreams.runtime_v2`
provides session execution, presentation, client windows, and metrics.

## Guides

<div class="fd-card-grid fd-card-grid-three fd-overview-grid">
  <a class="fd-card fd-model-card" href="guides/create_demo/"><span class="fd-card-title">Create a demo</span><span>Build a model-independent application against the Demo API.</span></a>
  <a class="fd-card fd-model-card" href="guides/integrate_model/"><span class="fd-card-title">Integrate a model</span><span>Connect a model implementation to an existing demo.</span></a>
  <a class="fd-card fd-model-card" href="guides/configuration/"><span class="fd-card-title">Demo configuration</span><span>Define arguments, defaults, and application configuration.</span></a>
  <a class="fd-card fd-model-card" href="guides/application_slugs/"><span class="fd-card-title">Application slugs and dispatch</span><span>Register application names and route model adapters.</span></a>
  <a class="fd-card fd-model-card" href="guides/interactive_serving/"><span class="fd-card-title">Interactive serving</span><span>Serve responsive sessions through supported client windows.</span></a>
  <a class="fd-card fd-model-card" href="guides/local_benchmarks/"><span class="fd-card-title">Local benchmarks</span><span>Measure application performance in a local environment.</span></a>
  <a class="fd-card fd-model-card" href="guides/profiling/"><span class="fd-card-title">Profiling a demo</span><span>Collect Nsight Systems traces and per-step performance metrics.</span></a>
</div>

## API Reference

<div class="fd-card-grid fd-card-grid-three fd-overview-grid">
  <a class="fd-card fd-model-card" href="api_reference/application/"><span class="fd-card-title">Demo Application API</span><span>Reference for application, session, loop, input, and output contracts.</span></a>
</div>
