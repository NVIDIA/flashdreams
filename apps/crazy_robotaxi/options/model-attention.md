# MODEL — OPTIMIZED ATTENTION

[MODEL overview](model.md) · [Options index](../OPTIONS.md)

On **MODEL**, find **NETWORK** within the **TRANSFORMER** section. The two optimized implementation sections below appear there.

## SELF ATTN OPTIMIZED IMPL CONFIG

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Qkv Fusion Option:** | FlashDreams | — | Projection fusion policy. |
| **Sdpa Backend:** | FlashDreams | — | Scaled-dot-product attention implementation. |
| **Use Tma:** | FlashDreams | — | Prefer TMA FlashAttention2 when the device and tensors support it. |

**QUANTIZATION**

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Projection:** | FlashDreams | — | Q/K/V projection dtype; None preserves native precision. |
| **Output Projection:** | FlashDreams | — | Output projection dtype; None preserves native precision. |
| **Output Granularity:** | FlashDreams | — | Activation quantization granularity for the output projection. |
| **Quantized Sdpa:** | FlashDreams | — | Use unscaled FP8 e4m3 Q/K/V in scaled-dot-product attention; can reduce accuracy. |

**FLEX ATTENTION**

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Block Size:** | FlashDreams | — | Square fallback used when building a block mask. |
| **Mask Block M:** | FlashDreams | — | Optional query block size; None uses block_size. |
| **Mask Block N:** | FlashDreams | — | Optional key/value block size; None uses block_size. |
| **Compile Dynamic:** | FlashDreams | — | Dynamic-shape policy forwarded to torch.compile. |
| **Block M:** | FlashDreams | — | Optional forward query tile size; None lets PyTorch choose. |
| **Block N:** | FlashDreams | — | Optional forward key/value tile size; None lets PyTorch choose. |
| **Num Warps:** | FlashDreams | — | Optional Triton warp count; None lets PyTorch choose. |
| **Num Stages:** | FlashDreams | — | Optional Triton pipeline-stage count; None lets PyTorch choose. |
| **Prescale Qk:** | FlashDreams | — | Whether to apply the attention scale before the QK reduction. |
| **Use Tma:** | FlashDreams | — | Whether to request TMA from FlexAttention; None uses its default. |
| **Backend:** | FlashDreams | — | Optional FlexAttention kernel backend: TRITON or FLASH. |
| **Rows Guaranteed Safe:** | FlashDreams | — | Skip empty-row guards when every query sees at least one key. |

## CROSS ATTN OPTIMIZED IMPL CONFIG

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Qkv Fusion Option:** | FlashDreams | — | Projection fusion policy. |
| **Sdpa Backend:** | FlashDreams | — | Scaled-dot-product attention implementation. |
| **Use Tma:** | FlashDreams | — | Prefer TMA FlashAttention2 when the device and tensors support it. |

**QUANTIZATION**

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Projection:** | FlashDreams | — | Q/K/V projection dtype; None preserves native precision. |
| **Output Projection:** | FlashDreams | — | Output projection dtype; None preserves native precision. |
| **Output Granularity:** | FlashDreams | — | Activation quantization granularity for the output projection. |
| **Quantized Sdpa:** | FlashDreams | — | Use unscaled FP8 e4m3 Q/K/V in scaled-dot-product attention; can reduce accuracy. |

**FLEX ATTENTION**

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Block Size:** | FlashDreams | — | Square fallback used when building a block mask. |
| **Mask Block M:** | FlashDreams | — | Optional query block size; None uses block_size. |
| **Mask Block N:** | FlashDreams | — | Optional key/value block size; None uses block_size. |
| **Compile Dynamic:** | FlashDreams | — | Dynamic-shape policy forwarded to torch.compile. |
| **Block M:** | FlashDreams | — | Optional forward query tile size; None lets PyTorch choose. |
| **Block N:** | FlashDreams | — | Optional forward key/value tile size; None lets PyTorch choose. |
| **Num Warps:** | FlashDreams | — | Optional Triton warp count; None lets PyTorch choose. |
| **Num Stages:** | FlashDreams | — | Optional Triton pipeline-stage count; None lets PyTorch choose. |
| **Prescale Qk:** | FlashDreams | — | Whether to apply the attention scale before the QK reduction. |
| **Use Tma:** | FlashDreams | — | Whether to request TMA from FlexAttention; None uses its default. |
| **Backend:** | FlashDreams | — | Optional FlexAttention kernel backend: TRITON or FLASH. |
| **Rows Guaranteed Safe:** | FlashDreams | — | Skip empty-row guards when every query sees at least one key. |
