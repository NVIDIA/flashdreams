# MODEL — OPTIMIZED ATTENTION

[MODEL overview](model.md) · [Options index](../OPTIONS.md)

On **MODEL**, find **NETWORK** within the **TRANSFORMER** section. The two optimized implementation sections below appear there.

## SELF ATTN OPTIMIZED IMPL CONFIG

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Qkv Fusion Option:** | `model.pipeline.diffusion_model.transformer.network.self_attn_optimized_impl_config.qkv_fusion_option` | FlashDreams | — | How query, key, and value projections are fused. |
| **Sdpa Backend:** | `model.pipeline.diffusion_model.transformer.network.self_attn_optimized_impl_config.sdpa_backend` | FlashDreams | — | Scaled dot-product attention kernel. |
| **Use Tma:** | `model.pipeline.diffusion_model.transformer.network.self_attn_optimized_impl_config.use_tma` | FlashDreams | — | Prefer TMA FlashAttention when supported. |

**QUANTIZATION**

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Projection:** | `model.pipeline.diffusion_model.transformer.network.self_attn_optimized_impl_config.quantization.projection` | FlashDreams | — | Optional Q/K/V projection precision; blank keeps native precision. |
| **Output Projection:** | `model.pipeline.diffusion_model.transformer.network.self_attn_optimized_impl_config.quantization.output_projection` | FlashDreams | — | Optional output projection precision; blank keeps native precision. |
| **Output Granularity:** | `model.pipeline.diffusion_model.transformer.network.self_attn_optimized_impl_config.quantization.output_granularity` | FlashDreams | — | Granularity of output activation quantization. |
| **Quantized Sdpa:** | `model.pipeline.diffusion_model.transformer.network.self_attn_optimized_impl_config.quantization.quantized_sdpa` | FlashDreams | — | Casts attention Q/K/V to FP8 for the selected backend. This can affect accuracy. |

**FLEX ATTENTION**

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Block Size:** | `model.pipeline.diffusion_model.transformer.network.self_attn_optimized_impl_config.flex_attention.block_size` | FlashDreams | — | Default square mask-block size. |
| **Mask Block M:** | `model.pipeline.diffusion_model.transformer.network.self_attn_optimized_impl_config.flex_attention.mask_block_m` | FlashDreams | — | Optional query-side mask-block size. |
| **Mask Block N:** | `model.pipeline.diffusion_model.transformer.network.self_attn_optimized_impl_config.flex_attention.mask_block_n` | FlashDreams | — | Optional key/value-side mask-block size. |
| **Compile Dynamic:** | `model.pipeline.diffusion_model.transformer.network.self_attn_optimized_impl_config.flex_attention.compile_dynamic` | FlashDreams | — | Dynamic-shape compile policy. |
| **Block M:** | `model.pipeline.diffusion_model.transformer.network.self_attn_optimized_impl_config.flex_attention.block_m` | FlashDreams | — | Optional query-side kernel tile size. |
| **Block N:** | `model.pipeline.diffusion_model.transformer.network.self_attn_optimized_impl_config.flex_attention.block_n` | FlashDreams | — | Optional key/value-side kernel tile size. |
| **Num Warps:** | `model.pipeline.diffusion_model.transformer.network.self_attn_optimized_impl_config.flex_attention.num_warps` | FlashDreams | — | Optional Triton warp count. |
| **Num Stages:** | `model.pipeline.diffusion_model.transformer.network.self_attn_optimized_impl_config.flex_attention.num_stages` | FlashDreams | — | Optional Triton pipeline-stage count. |
| **Prescale Qk:** | `model.pipeline.diffusion_model.transformer.network.self_attn_optimized_impl_config.flex_attention.prescale_qk` | FlashDreams | — | Applies attention scale before the QK reduction. |
| **Use Tma:** | `model.pipeline.diffusion_model.transformer.network.self_attn_optimized_impl_config.flex_attention.use_tma` | FlashDreams | — | Requests TMA for FlexAttention; blank uses its default. |
| **Backend:** | `model.pipeline.diffusion_model.transformer.network.self_attn_optimized_impl_config.flex_attention.backend` | FlashDreams | — | Optional `TRITON` or `FLASH` FlexAttention kernel. |
| **Rows Guaranteed Safe:** | `model.pipeline.diffusion_model.transformer.network.self_attn_optimized_impl_config.flex_attention.rows_guaranteed_safe` | FlashDreams | — | Skips empty-row guards when every query has a valid key. |

## CROSS ATTN OPTIMIZED IMPL CONFIG

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Qkv Fusion Option:** | `model.pipeline.diffusion_model.transformer.network.cross_attn_optimized_impl_config.qkv_fusion_option` | FlashDreams | — | How query, key, and value projections are fused. |
| **Sdpa Backend:** | `model.pipeline.diffusion_model.transformer.network.cross_attn_optimized_impl_config.sdpa_backend` | FlashDreams | — | Scaled dot-product attention kernel. |
| **Use Tma:** | `model.pipeline.diffusion_model.transformer.network.cross_attn_optimized_impl_config.use_tma` | FlashDreams | — | Prefer TMA FlashAttention when supported. |

**QUANTIZATION**

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Projection:** | `model.pipeline.diffusion_model.transformer.network.cross_attn_optimized_impl_config.quantization.projection` | FlashDreams | — | Optional Q/K/V projection precision; blank keeps native precision. |
| **Output Projection:** | `model.pipeline.diffusion_model.transformer.network.cross_attn_optimized_impl_config.quantization.output_projection` | FlashDreams | — | Optional output projection precision; blank keeps native precision. |
| **Output Granularity:** | `model.pipeline.diffusion_model.transformer.network.cross_attn_optimized_impl_config.quantization.output_granularity` | FlashDreams | — | Granularity of output activation quantization. |
| **Quantized Sdpa:** | `model.pipeline.diffusion_model.transformer.network.cross_attn_optimized_impl_config.quantization.quantized_sdpa` | FlashDreams | — | Casts attention Q/K/V to FP8 for the selected backend. This can affect accuracy. |

**FLEX ATTENTION**

| On-screen label | `config.yaml` key | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- | --- |
| **Block Size:** | `model.pipeline.diffusion_model.transformer.network.cross_attn_optimized_impl_config.flex_attention.block_size` | FlashDreams | — | Default square mask-block size. |
| **Mask Block M:** | `model.pipeline.diffusion_model.transformer.network.cross_attn_optimized_impl_config.flex_attention.mask_block_m` | FlashDreams | — | Optional query-side mask-block size. |
| **Mask Block N:** | `model.pipeline.diffusion_model.transformer.network.cross_attn_optimized_impl_config.flex_attention.mask_block_n` | FlashDreams | — | Optional key/value-side mask-block size. |
| **Compile Dynamic:** | `model.pipeline.diffusion_model.transformer.network.cross_attn_optimized_impl_config.flex_attention.compile_dynamic` | FlashDreams | — | Dynamic-shape compile policy. |
| **Block M:** | `model.pipeline.diffusion_model.transformer.network.cross_attn_optimized_impl_config.flex_attention.block_m` | FlashDreams | — | Optional query-side kernel tile size. |
| **Block N:** | `model.pipeline.diffusion_model.transformer.network.cross_attn_optimized_impl_config.flex_attention.block_n` | FlashDreams | — | Optional key/value-side kernel tile size. |
| **Num Warps:** | `model.pipeline.diffusion_model.transformer.network.cross_attn_optimized_impl_config.flex_attention.num_warps` | FlashDreams | — | Optional Triton warp count. |
| **Num Stages:** | `model.pipeline.diffusion_model.transformer.network.cross_attn_optimized_impl_config.flex_attention.num_stages` | FlashDreams | — | Optional Triton pipeline-stage count. |
| **Prescale Qk:** | `model.pipeline.diffusion_model.transformer.network.cross_attn_optimized_impl_config.flex_attention.prescale_qk` | FlashDreams | — | Applies attention scale before the QK reduction. |
| **Use Tma:** | `model.pipeline.diffusion_model.transformer.network.cross_attn_optimized_impl_config.flex_attention.use_tma` | FlashDreams | — | Requests TMA for FlexAttention; blank uses its default. |
| **Backend:** | `model.pipeline.diffusion_model.transformer.network.cross_attn_optimized_impl_config.flex_attention.backend` | FlashDreams | — | Optional `TRITON` or `FLASH` FlexAttention kernel. |
| **Rows Guaranteed Safe:** | `model.pipeline.diffusion_model.transformer.network.cross_attn_optimized_impl_config.flex_attention.rows_guaranteed_safe` | FlashDreams | — | Skips empty-row guards when every query has a valid key. |
