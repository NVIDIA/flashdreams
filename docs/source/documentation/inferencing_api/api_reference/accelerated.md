---
title: 'Accelerated API Reference'
---

The `flashdreams.accelerated` package provides optimized attention, quantization,
and shared inference components. See [Accelerated building blocks](../guides/accelerated.md)
for usage and integration guidance.

## Common components

::: flashdreams.accelerated.common.NonPersistentLinear

::: flashdreams.accelerated.common.rms_norm

## Quantization

::: flashdreams.accelerated.quantization.quantizer.Granularity

::: flashdreams.accelerated.quantization.quantizer.quantize

::: flashdreams.accelerated.quantization.quantizer.dequantize

::: flashdreams.accelerated.quantization.linear.WeightGranularity

::: flashdreams.accelerated.quantization.linear.QuantizedNonPersistentLinear

## Attention contracts

::: flashdreams.accelerated.multi_head_attention.AttentionType

::: flashdreams.accelerated.multi_head_attention.AttentionConfig

::: flashdreams.accelerated.multi_head_attention.AttentionMask

::: flashdreams.accelerated.multi_head_attention.QKNormScope

::: flashdreams.accelerated.multi_head_attention.RoPEStyle

::: flashdreams.accelerated.multi_head_attention.RoPEScope

::: flashdreams.accelerated.multi_head_attention.RoPEConfig

::: flashdreams.accelerated.multi_head_attention.MultiHeadAttention

## Attention implementations

::: flashdreams.accelerated.multi_head_attention.torch.TorchMultiHeadAttention

::: flashdreams.accelerated.multi_head_attention.optimized.SDPABackend

::: flashdreams.accelerated.multi_head_attention.optimized.QKVFusionOption

::: flashdreams.accelerated.multi_head_attention.optimized.QuantizationOption

::: flashdreams.accelerated.multi_head_attention.optimized.OptimizedImplConfig

::: flashdreams.accelerated.multi_head_attention.optimized.OptimizedMultiHeadAttention

::: flashdreams.accelerated.multi_head_attention.flex.FlexAttentionOptions

::: flashdreams.accelerated.multi_head_attention.flex.compiled_flex_attention

::: flashdreams.accelerated.multi_head_attention.MoTRoute

::: flashdreams.accelerated.multi_head_attention.mot.MoTMultiHeadAttention

## Masked attention functions

::: flashdreams.accelerated.multi_head_attention.functional.masked_attention

::: flashdreams.accelerated.multi_head_attention.reference.reference_masked_attention

## KV arena

::: flashdreams.accelerated.multi_head_attention.kv_arena.KVSegmentLayout

::: flashdreams.accelerated.multi_head_attention.kv_arena.allocate_kv_arena

::: flashdreams.accelerated.multi_head_attention.kv_arena.validate_kv_arena_segments

::: flashdreams.accelerated.multi_head_attention.kv_arena.stage_current_kv_
