<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LongSANA acceleration scope

This document scopes quality-preserving acceleration of the LongSANA Runtime V2
pipeline, with emphasis on reuse and extension of `flashdreams.accelerated`.
The first target is steady-state latency at the released 832 x 480 resolution;
the recurrent cache must remain constant-memory and session-local.

## Baseline and target

The September 2026 RTX PRO 6000 Blackwell baseline for one steady 40-frame
output block is:

| stage | latency | share |
| --- | ---: | ---: |
| Four denoising DiT forwards | about 1,620 ms | 40% |
| Required clean cache-commit forward | about 404 ms | 10% |
| FP32 Wan decode | about 2,015 ms | 50% |
| End to end | about 4,029 ms | 100% |

This is 9.93 output FPS end to end and 19.86 FPS for the DiT plus commit.
Resident allocation is about 8.07 GiB, peak allocation is 37.30 GiB, and the
20-layer recurrent cache remains exactly 152.526855 MiB.

A useful first milestone is a 1.5x steady-state speedup without a measurable
quality regression or cache growth. A 2x DiT speedup alone is bounded to about
1.33x end to end (roughly 3.03 seconds per block), as is a 2x decoder speedup.
Halving both stages would approach 2.02 seconds, or about 19.8 output FPS.

## Compatibility map

| LongSANA path | Existing accelerated component | Status |
| --- | --- | --- |
| Causal recurrent self-attention | None | Requires a new linear-attention primitive |
| Static text cross-attention | `OptimizedMultiHeadAttention` | Needs 112-wide heads, mask support, and an adapter |
| Dense and pointwise projections | `QuantizedNonPersistentLinear` | Opt-in candidate after checkpoint load |
| Wan VAE decoder | None | Use compile/graphs first; accelerated convolution is future work |

LongSANA self-attention is not softmax attention. It applies a positive ReLU
kernel and updates cumulative `V @ K^T` and key-sum tensors. Replacing it with
`OptimizedMultiHeadAttention` would change the model, so the existing MHA path
is only applicable to cross-attention.

LongSANA uses 20 heads with head dimension 112. The current
`OptimizedMultiHeadAttention` validates a power-of-two head dimension in
`[16, 256]`, so it rejects this model even though the cross-attention core is
conventional scaled-dot-product attention. The LongSANA path also applies a
prompt padding mask, while the accelerated `compute_kv` and `forward`
interfaces do not accept a mask, and its checkpoint exposes one fused
`kv_linear` rather than separate key and value accessors.

Reuse therefore requires a checkpoint-preserving adapter, a compatible
cuDNN/SDPA path for 112, and valid-token handling. Compacting valid prompt
tokens before `compute_kv` may preserve the current batch-one semantics, but
must be covered by masked-reference parity tests.

## Phase 0: lock the benchmark and parity gates

Before optimizing, retain the current script outputs for first and steady
blocks:

- end-to-end, diffusion, cache-commit, and decode latency;
- milliseconds per network forward and output FPS;
- resident and peak allocated VRAM;
- recurrent-cache bytes;
- backing-buffer addresses after extending the benchmark harness to record them;
- an operator trace after lazy initialization;
- seeded outputs for a two-block correctness case and a 24-block continuity
  case.

Run each candidate on the same prompt, seed, block count, precision, and GPU.
Report first-block compilation or graph-capture cost separately from warmed
steady-state performance.

## Phase 1: exact, low-risk work

### Cache static text K/V

Each of the 20 cross-attention layers currently recomputes its text K/V
projection and K normalization on every denoising and commit forward even
though the 300-token prompt is fixed for the session. Extend the per-session
cache with one K/V pair per layer, compute it after prompt projection, and reuse
it for every block.

This mirrors the static cross-attention cache owned by
`OptimizedMultiHeadAttention.compute_kv` and is the cleanest first reuse
point. It preserves checkpoint parameters and attention math. The added memory
is prompt-length dependent but duration independent and must be accounted for
separately from the recurrent self-attention state.

### Compile and capture the two block shapes

Measure `compile_network=True` for the DiT and the decoder's existing
`use_compile` and `use_cuda_graph` settings. LongSANA has two stable DiT
shapes: 11 latent frames for block zero and 10 for every steady block. Compile
both shapes.

The recurrent state tensors are allocated on the first clean cache commit.
Capture the steady-state DiT graph only after those tensors exist, and keep
their addresses stable. Preserve the Wan path's intentional eager first decode
and capture only stable steady decoder calls. The clean commit forward is part
of the model algorithm and must be accelerated rather than skipped.

### Cache or fuse RoPE construction

`causal_wan_rope` currently rebuilds complex128 tables and
`_apply_causal_rope` casts Q/K through float64 in every transformer block.
First cache the immutable axis tables by device and slice them by absolute
frame position. Only consider a fused or lower-precision implementation after
numerical and video-quality A/B validation; the current precision matches the
released model.

## Phase 2: extend `flashdreams.accelerated`

### Recurrent causal linear attention

Add a dedicated accelerated primitive rather than adapting softmax MHA. Its
interface must consume current Q/K/V plus the session's cumulative
`value_key` and `key_sum`, produce the normalized output, and optionally
update both states in-place during the clean commit.

The kernel should fuse the highest-traffic operations where profitable:

1. Q/K normalization, causal RoPE, and the positive feature map;
2. blockwise K and `V @ K^T` reductions plus the prior recurrent totals;
3. numerator and denominator contractions;
4. in-place final-state writes.

It must support BF16 inputs, the reference mixed-precision state semantics
(FP32 `value_key` and reference-dtype `key_sum`), head dimension 112, both
11- and 10-frame blocks, and non-mutating denoising forwards. A Torch
implementation should remain selectable for parity and unsupported hardware.

### Cross-attention compatibility

Extend the accelerated attention policy so conventional cross-attention can
use `QKVFusionOption.FUSE_KV`, static `compute_kv`, and either
`SDPABackend.CUDNN` or a compatible fallback at head dimension 112. Add
mask or valid-token compaction semantics and a fused-`kv_linear` checkpoint
adapter. Do not pad silently unless tests prove padded normalization and
projection math are equivalent.

Start with BF16 projections and SDPA. Evaluate projection FP8 separately using
`QuantizationOption`; simple FP8 SDPA is explicitly accuracy-sensitive and
should not be enabled by default.

### Derived fused and quantized projections

Preserve the 418-tensor checkpoint schema. `NonPersistentLinear` provides
derived, nonpersistent buffers but is still the same linear operation; replacing
an existing `nn.Linear` with it is not by itself an acceleration. Construct
nonpersistent packed weights only where a kernel consumes a genuinely fused
layout.

After strict loading, evaluate `QuantizedNonPersistentLinear` as an opt-in for:

- self-attention QKV (2240 to 6720) and output (2240 to 2240);
- cross-attention Q (2240 to 2240), KV (2240 to 4480), and output;
- timestep/modulation (2240 to 13440);
- GLUMB pointwise 1x1 convolutions.

The first three groups are already single canonical `nn.Linear` operations,
so their opportunity is quantization or fusion into a larger kernel, not
projection packing alone.

`linearize_stage1_ffn_for_quant` currently recognizes SANA-WM's
`GLUMBConvTemp`, not `LongSanaCausalGLUMBConvTemp`. Generalize that helper
or add a LongSANA-specific equivalent before applying
`QuantizedNonPersistentLinear`. Benchmark 1x1-convolution linearization
separately, then FP8 and INT8 variants independently so quality and speed
effects are attributable.

## Phase 3: decoder and memory

The official FP32 Wan decoder consumes about half the block latency and drives
the 37.30 GiB peak. Existing `flashdreams.accelerated` APIs do not provide a
drop-in VAE or convolution implementation, so first exhaust decoder compilation,
CUDA graphs, scheduling, and allocator reuse.

Then evaluate reduced-precision decoder convolutions behind an opt-in setting.
If convolution remains the dominant bottleneck, scope accelerated 3D
convolution or a model-specific Wan decoder path as a separate project. Do not
combine decoder precision changes with DiT quantization in the same experiment.

## Acceptance gates

Every default-path optimization must retain:

- strict loading of all 418 tensors and 2,057,553,344 parameters;
- finite outputs and exact first/steady output shapes;
- identical seeded tensors for exact BF16/FP32 modes within an agreed
  operator-level tolerance;
- the same cache lifecycle, session isolation, and in-place backing storage;
- duration-independent recurrent state at exactly 152.526855 MiB;
- successful two-block Runtime V2 generation and a 24-block continuity run.

Quantized or reduced-precision modes additionally require a diverse prompt set
covering people, animals, camera motion, text-like detail, fast motion, and long
scene continuity. Record latent error and stage-level drift, plus VBench, CLIP,
or FVD when available; otherwise retain blinded contact-sheet and full-video
review. Promote a mode to the default only after both quality and performance
gates pass.

## Deliverables

1. Exact-path baseline PR: static cross K/V, cached RoPE tables, compile/graph
   measurements, and stage-separated benchmark results.
2. Accelerated linear-attention PR: reference/kernel parity tests, in-place
   cache tests, and 112-dimension support.
3. Projection optimization PR: nonpersistent fusion followed by separately
   gated FP8/INT8 configurations and quality reports.
4. Decoder PR or follow-up RFC: only if compile and graph capture leave the VAE
   as the limiting stage.

Each PR should report speedup against this baseline, not only isolated kernel
throughput, and should include an Amdahl projection for the next bottleneck.
