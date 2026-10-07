# MODEL — OPTIMIZED ATTENTION

[MODEL overview](model.md) · [Options index](../OPTIONS.md)

On **MODEL**, find **NETWORK** within the **TRANSFORMER** section. The two optimized implementation sections below appear there.

## SELF ATTN OPTIMIZED IMPL CONFIG

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Qkv Fusion Option:** | FlashDreams | — | How query, key, and value projections are fused. |
| **Sdpa Backend:** | FlashDreams | — | Scaled dot-product attention kernel. |
| **Use Tma:** | FlashDreams | — | Prefer TMA FlashAttention when supported. |

**QUANTIZATION**

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Projection:** | FlashDreams | — | Optional Q/K/V projection precision; blank keeps native precision. |
| **Output Projection:** | FlashDreams | — | Optional output projection precision; blank keeps native precision. |
| **Output Granularity:** | FlashDreams | — | Granularity of output activation quantization. |
| **Quantized Sdpa:** | FlashDreams | — | Casts attention Q/K/V to FP8 for the selected backend. This can affect accuracy. |

**FLEX ATTENTION**

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Block Size:** | FlashDreams | — | Default square mask-block size. |
| **Mask Block M:** | FlashDreams | — | Optional query-side mask-block size. |
| **Mask Block N:** | FlashDreams | — | Optional key/value-side mask-block size. |
| **Compile Dynamic:** | FlashDreams | — | Dynamic-shape compile policy. |
| **Block M:** | FlashDreams | — | Optional query-side kernel tile size. |
| **Block N:** | FlashDreams | — | Optional key/value-side kernel tile size. |
| **Num Warps:** | FlashDreams | — | Optional Triton warp count. |
| **Num Stages:** | FlashDreams | — | Optional Triton pipeline-stage count. |
| **Prescale Qk:** | FlashDreams | — | Applies attention scale before the QK reduction. |
| **Use Tma:** | FlashDreams | — | Requests TMA for FlexAttention; blank uses its default. |
| **Backend:** | FlashDreams | — | Optional `TRITON` or `FLASH` FlexAttention kernel. |
| **Rows Guaranteed Safe:** | FlashDreams | — | Skips empty-row guards when every query has a valid key. |

## CROSS ATTN OPTIMIZED IMPL CONFIG

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Qkv Fusion Option:** | FlashDreams | — | How query, key, and value projections are fused. |
| **Sdpa Backend:** | FlashDreams | — | Scaled dot-product attention kernel. |
| **Use Tma:** | FlashDreams | — | Prefer TMA FlashAttention when supported. |

**QUANTIZATION**

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Projection:** | FlashDreams | — | Optional Q/K/V projection precision; blank keeps native precision. |
| **Output Projection:** | FlashDreams | — | Optional output projection precision; blank keeps native precision. |
| **Output Granularity:** | FlashDreams | — | Granularity of output activation quantization. |
| **Quantized Sdpa:** | FlashDreams | — | Casts attention Q/K/V to FP8 for the selected backend. This can affect accuracy. |

**FLEX ATTENTION**

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Block Size:** | FlashDreams | — | Default square mask-block size. |
| **Mask Block M:** | FlashDreams | — | Optional query-side mask-block size. |
| **Mask Block N:** | FlashDreams | — | Optional key/value-side mask-block size. |
| **Compile Dynamic:** | FlashDreams | — | Dynamic-shape compile policy. |
| **Block M:** | FlashDreams | — | Optional query-side kernel tile size. |
| **Block N:** | FlashDreams | — | Optional key/value-side kernel tile size. |
| **Num Warps:** | FlashDreams | — | Optional Triton warp count. |
| **Num Stages:** | FlashDreams | — | Optional Triton pipeline-stage count. |
| **Prescale Qk:** | FlashDreams | — | Applies attention scale before the QK reduction. |
| **Use Tma:** | FlashDreams | — | Requests TMA for FlexAttention; blank uses its default. |
| **Backend:** | FlashDreams | — | Optional `TRITON` or `FLASH` FlexAttention kernel. |
| **Rows Guaranteed Safe:** | FlashDreams | — | Skips empty-row guards when every query has a valid key. |
