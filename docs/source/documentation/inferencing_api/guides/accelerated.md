---
title: 'Accelerated building blocks'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

`flashdreams.accelerated` provides low-level inference components used by model
integrations. It currently covers tensor/linear quantization and streaming
multi-head attention. These APIs are implementation tools, not a switch that
makes every pipeline faster.

Start with the [latency tuning guide](latency_tuning.md). Use an accelerated
component only after profiling identifies its stage as important, and keep the
existing precision/backend path as the correctness reference.

## Quantization

### Tensor quantization

`quantize` supports `torch.int8`, `torch.float8_e4m3fn`, and
`torch.float8_e5m2`. It returns the quantized tensor and FP32 scale:

```python
import torch

from flashdreams.accelerated.quantization.quantizer import (
    Granularity,
    dequantize,
    quantize,
)

x = torch.tensor([[0.42, 0.12], [0.37, -0.91]], dtype=torch.float32)
quantized, scale = quantize(
    x,
    torch.int8,
    Granularity.SLICE,
    axis=-1,
    use_triton=False,
)
restored = dequantize(
    quantized,
    scale,
    dtype=x.dtype,
    use_triton=False,
)
```

Granularity controls scale shape:

- `Granularity.TENSOR` reduces every dimension and uses one scale;
- `Granularity.SLICE` reduces `axis` and keeps that dimension with size one;
- for `[rows, columns]`, `axis=-1` produces one scale per row.

CUDA tensors use Triton by default; CPU tensors use the Torch implementation.
A zero-valued slice is safe because the scale is clamped to a small positive
floor. `dequantize` multiplies all supplied scales in order, which supports
composing activation and weight scales after a quantized operation.

### Quantized linear

`QuantizedNonPersistentLinear` derives quantized execution weights from an
existing projection. The derived weight and scale are nonpersistent, so the
canonical full-precision module remains the checkpoint interface.

```python
import torch

from flashdreams.accelerated.quantization.linear import (
    QuantizedNonPersistentLinear,
    WeightGranularity,
)
from flashdreams.accelerated.quantization.quantizer import Granularity

weight = torch.randn(32, 64, device="cuda", dtype=torch.float16)
bias = torch.randn(32, device="cuda", dtype=torch.float16)
layer = QuantizedNonPersistentLinear(
    weight,
    bias,
    WeightGranularity.PER_OUT_CHANNEL,
    torch.float8_e4m3fn,
)

x = torch.randn(2, 8, 64, device="cuda", dtype=torch.float16)
output = layer(x, Granularity.SLICE)
assert output.shape == (2, 8, 32)
```

The layer also accepts an already-quantized activation plus its scale. Reuse
that form only when several operations consume the same quantized input; the
caller then owns dtype and scale-shape correctness.

Quantization changes numerics. Validate model or stage quality on the target
hardware before enabling it by default.

## Streaming multi-head attention

The common interface owns the complete projection, normalization, RoPE, cache,
attention, and output-projection operation. This lets optimized backends fuse
work without changing model call sites.

### Shared policy

`AttentionConfig` defines:

- query and context widths;
- query and K/V head counts plus head dimension;
- Q/K normalization scope and epsilon;
- optional RoPE style and whether keys are rotated before or after caching.

`AttentionType.SELF_ATTENTION` updates a rolling cache from every query chunk.
`AttentionType.CROSS_ATTENTION` reads a static cache produced once with
`compute_kv`.

The core call is:

```text
forward(x, kv_cache, rope_freqs=None, *, attn_mask=None) -> output
```

where `x` is `[..., L, query_dim]`. Output has the same leading shape and query
width.

### Cache lifecycle

For streaming self-attention:

1. allocate a `BlockKVCache` with fixed chunk, window, and sink sizes;
2. call `cache.before_update(chunk_index)`;
3. call attention once for the current chunk;
4. call `cache.after_update(chunk_index)`.

For cross-attention, call `compute_kv(context, rope_freqs)` once and reuse the
returned finalized cache. Apply RoPE at the location required by the model:
keys stored with before-cache RoPE must already be rotated; after-cache RoPE
requires cache-relative frequencies at each query.

Keep this lifecycle outside compiled or captured regions until the eager path
is correct. Test both cache fill and post-eviction steady state.

### Reference and optimized implementations

Use `TorchMultiHeadAttention` as the readable correctness path. Both it and
`OptimizedMultiHeadAttention` are bases for a model adapter: the adapter keeps
checkpoint-native projection names and implements these logical properties:

- `query_projection`;
- `key_projection`;
- `value_projection`;
- `output_projection`;
- `query_norm`;
- `key_norm`.

An optimized adapter additionally calls `_initialize_derived_weights()` after
creating those canonical modules. That method builds nonpersistent fused or
quantized projections and refreshes them after checkpoint loading.

`OptimizedImplConfig` selects execution policy:

| Field | Choices | Constraint |
| --- | --- | --- |
| `sdpa_backend` | `CUDNN`, `FA2`, `FLEX` | support depends on device, dtype, shape, and mask |
| `qkv_fusion_option` | `NONE`, `FULL`, `FUSE_KV` | `FULL` requires equal query/context widths |
| `use_tma` | boolean | used only when the FA2 TMA path supports the device and tensors |
| `quantization` | projection/output/SDPA policy | must pass quality validation |

The optimized implementation requires a power-of-two head dimension from 16 to
256. Grouped-query attention cannot use inner-width Q/K normalization.
`QuantizationOption.quantized_sdpa=True` is a simple unscaled FP8 attention path,
not an accuracy-preserving SageAttention implementation; output quality is not
guaranteed.

Expose self- and cross-attention policies separately on the integration config.
A schedule selected on one GPU should remain an opt-in variant until measured on
other supported hardware.

## Integration workflow

1. Match the model's checkpoint-native module names and exact attention policy.
2. Implement and test the Torch/reference adapter first.
3. Add the optimized adapter behind config fields without changing call sites.
4. Load the same weights into both paths.
5. Compare outputs before and after rolling-cache eviction.
6. Benchmark the target shapes on the target GPU after warmup.
7. Validate end-to-end quality, reset, shape-change, and fallback behavior.

Do not infer the active backend from configuration alone. Confirm it with
runtime logging or profiler kernels; unsupported shapes may select a fallback.

## Tests and benchmarks

Run CPU-safe contracts first:

```bash
uv run --project flashdreams --group test pytest \
    flashdreams/tests/accelerated -m ci_cpu
```

The `ci_cpu` marker avoids GPU and checkpoint work. FlexAttention CPU cases
still require a host and PyTorch build that support CPU compilation; use the
project CI environment when a local backend reports that it is unsupported.

Run CUDA implementations only on a supported NVIDIA system:

```bash
uv run --project flashdreams --group test pytest \
    flashdreams/tests/accelerated -m ci_gpu
```

The manual microbenchmarks are separate from correctness tests:

```bash
uv run --project flashdreams --group test pytest \
    flashdreams/benchmarks/accelerated \
    -p no:manual_marker -m manual --benchmark-only -v
```

Repository scripts save benchmark JSON and plots:

```bash
./scripts/benchmark/flashdreams/accelerated/quantization/run.sh
./scripts/benchmark/flashdreams/accelerated/multi_head_attention/run.sh
```

Record commit, GPU and software stack, exact shapes, dtype, backend, cache
geometry, warmup policy, and raw benchmark JSON. Promote an optimized path only
when the expected stage improves materially, quality passes against the correct
reference, startup/reset behavior is acceptable, and the fallback remains
available.
