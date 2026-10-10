---
title: 'Profiling a pipeline'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 Praneeth Samineni. -->

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

A `StreamInferencePipeline` reports its stages to the profiler set with
`set_flashdreams_inference_profiler`. To time each stage, use
`CudaEventProfiler`; `finalize()` then returns the timings:

```python
from flashdreams.runtime_v2.profiler import CudaEventProfiler
from flashdreams.runtime_v2.profiler_utils import set_flashdreams_inference_profiler

with set_flashdreams_inference_profiler(CudaEventProfiler()):
    output = pipeline.generate(0, cache, input=control)
    stats = pipeline.finalize(0, cache)  # encode_ms, diffuse_ms, decode_ms, ...
```

To see the stages in Nsight Systems instead, use `NVTXProfiler` and run your
script under `nsys profile`. The ranges are `pipeline.encode`,
`pipeline.diffuse`, `pipeline.decode` and `pipeline.finalize`. To use both,
pass them together to `CompositeProfiler`.

When no profiler is set, `FLASHDREAMS_SYNC_AND_PROFILE=1` turns on the stage
timings.

To use your own profiler, write a class that implements `IProfiler`, defining
`range()` and, if it times stages, `collect_stage_ms()`.

To profile a whole demo, see [Profiling a demo](../../demo_api/guides/profiling.md).
