# MODEL — TRANSFORMER

[MODEL overview](model.md) · [Options index](../OPTIONS.md)

On **MODEL**, find **TRANSFORMER** within **PIPELINE**, under **DIFFUSION MODEL**. **NETWORK** appears within **TRANSFORMER**.

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Dtype:** | OmniDreams | — | Parameter and activation precision. |
| **Checkpoint Path:** | OmniDreams | — | Pretrained transformer weights. |
| **Len T:** | OmniDreams | — | Latent frames generated per chunk. |
| **H Extrapolation Ratio:** | OmniDreams | — | Height RoPE extrapolation factor. |
| **W Extrapolation Ratio:** | OmniDreams | — | Width RoPE extrapolation factor. |
| **Window Size T:** | OmniDreams | — | Sliding self-attention history in temporal units before patchification. |
| **Sink Size T:** | OmniDreams | — | Temporal sink tokens retained in attention history. |
| **Early Short History Block Count:** | OmniDreams | — | Number of first blocks limited to one chunk of visual history; blank disables this policy. |
| **Compile Network:** | OmniDreams | `--compile`, `--no-compile` | Uses `torch.compile` for the network. |
| **Use Cuda Graph:** | OmniDreams | — | Captures and replays steady-state model work in a CUDA graph. |
| **Cuda Graph Warmup Iters:** | OmniDreams | — | Eager warmup calls before graph capture. |
| **Skip Finalize Kv Cache:** | OmniDreams | — | Skips the separate cache-finalization pass. |
| **Native Dit Acceleration:** | OmniDreams | — | Native DiT policy: `disabled`, `auto`, or `required`. |
| **Native Dit Build Root:** | OmniDreams | — | Native extension build and cache location. |
| **Native Dit Max Jobs:** | OmniDreams | — | Parallel job cap for the native extension build. |
| **Native Dit Verbose Build:** | OmniDreams | — | Enables detailed native extension build logs. |
| **Native Dit Backend:** | OmniDreams | — | Native compute backend, `fp8_kvcache_cudnn` or `bf16`. |
| **Native Dit Attention Backend:** | OmniDreams | — | Native attention implementation (`auto`, cuDNN/Sage/Sparge policies supported by the runtime). |
| **Native Dit Sparge Topk:** | OmniDreams | — | Top-k selection ratio for Sparge sparse attention. |
| **Native Dit Sparge Hybrid Period:** | OmniDreams | — | Period of a Sparge and SageAttention hybrid schedule. |
| **Native Dit Sparge Hybrid Phase:** | OmniDreams | — | Phase offset for the hybrid sparse-attention schedule. |
| **Guidance Scale:** | OmniDreams | — | Classifier-free guidance strength; `1` disables it. Higher values require negative text embeddings. |

## NETWORK

These fields control timestep scaling and attention execution. The runner preset fixes network dimensions and structural switches.

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Timestep Scale:** | OmniDreams | — | Multiplier applied before timestep embedding. |
| **Apply Rope Before Kvcache:** | OmniDreams | — | Rotates keys before caching; off uses cache-relative RoPE. |
| **Cp Method:** | OmniDreams | — | Context-parallel attention method: `ring` or `ulysses`. |
| **Self Attention Backend:** | OmniDreams | — | Implementation used by transformer self-attention. |
| **Cross Attention Backend:** | OmniDreams | — | Implementation used by text and cross-view attention. |

The optimized self-attention and cross-attention fields are listed in [Optimized attention](model-attention.md).
