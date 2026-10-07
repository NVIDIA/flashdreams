# MODEL — TRANSFORMER

[MODEL overview](model.md) · [Options index](../OPTIONS.md)

On **MODEL**, find **TRANSFORMER** within **PIPELINE**, under **DIFFUSION MODEL**. **NETWORK** appears within **TRANSFORMER**.

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Dtype:** | OmniDreams | — | Network parameter / activation dtype. |
| **Checkpoint Path:** | OmniDreams | — | Optional path to a pretrained checkpoint; None keeps the random init. |
| **Len T:** | OmniDreams | — | Latent frames per AR chunk. |
| **H Extrapolation Ratio:** | OmniDreams | — | RoPE extrapolation along H (3.0 @ 720p). |
| **W Extrapolation Ratio:** | OmniDreams | — | RoPE extrapolation along W. |
| **Window Size T:** | OmniDreams | — | Self-attention sliding window (pre-patchify T). |
| **Sink Size T:** | OmniDreams | — | Sink-token count (pre-patchify T). |
| **Early Short History Block Count:** | OmniDreams | — | Number of initial blocks limited to one chunk of visual history. None disables this policy. |
| **Compile Network:** | OmniDreams | `--compile`, `--no-compile` | torch.compile the network. |
| **Use Cuda Graph:** | OmniDreams | — | Wrap in CUDAGraphWrapper for steady-state replay. Caller must keep non-staged inputs at stable storage addresses across calls. |
| **Cuda Graph Warmup Iters:** | OmniDreams | — | Eager calls before capture (>= 2 to drain Inductor autotune). |
| **Skip Finalize Kv Cache:** | OmniDreams | — | Skip the KV cache finalize step. |
| **Native Dit Acceleration:** | OmniDreams | — | Native optimized DiT policy: disabled, auto, or required. |
| **Native Dit Build Root:** | OmniDreams | — | Optional native extension build/cache root. |
| **Native Dit Max Jobs:** | OmniDreams | — | Optional PyTorch/Ninja job cap for the native DiT build. |
| **Native Dit Verbose Build:** | OmniDreams | — | Forward verbose build output from the native extension loader. |
| **Native Dit Backend:** | OmniDreams | — | Optimized native DiT compute backend. |
| **Native Dit Attention Backend:** | OmniDreams | — | Optimized native attention backend. |
| **Native Dit Sparge Topk:** | OmniDreams | — | Optional Sparge self-attention top-k ratio. |
| **Native Dit Sparge Hybrid Period:** | OmniDreams | — | Optional Sparge/SageAttention-3 hybrid period. |
| **Native Dit Sparge Hybrid Phase:** | OmniDreams | — | Optional Sparge hybrid phase. None uses backend defaults. |
| **Guidance Scale:** | OmniDreams | — | CFG scale. 1.0 disables CFG; > 1.0 requires negative text embeddings. |

## NETWORK

These fields control timestep scaling and attention execution. The runner preset fixes network dimensions and structural switches.

| On-screen label | Defined by | CLI flag | What it changes |
| --- | --- | --- | --- |
| **Timestep Scale:** | OmniDreams | — | Multiplier applied to raw timestep values before sinusoidal embedding. |
| **Apply Rope Before Kvcache:** | OmniDreams | — | Rotate keys before caching. False enables cache-relative RoPE. |
| **Cp Method:** | OmniDreams | — | Context-parallel attention method for transformer attention ops. |
| **Self Attention Backend:** | OmniDreams | — | Self-attention implementation used by every DiT block. |
| **Cross Attention Backend:** | OmniDreams | — | Text and cross-view attention implementation used by every DiT block. |

The optimized self-attention and cross-attention fields are listed in [Optimized attention](model-attention.md).
