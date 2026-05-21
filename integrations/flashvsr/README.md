<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# flashdreams-flashvsr

FlashVSR-v1.1 streaming video super-resolution (LR projector + distilled
Wan 2.1 DiT + TC decoder + AdaIN color corrector), packaged as a
[`flashdreams`](../..) plugin, in a standalone repo. Wraps everything in
a `StreamInferencePipeline` so the same `generate` / `finalize`
lifecycle as the other recipes (`omnidreams`, `lingbot_world`, `wan2_1`)
applies.

This is a worked example of the
[Adding a new recipe](../../docs/source/developer_guides/new_recipes.rst)
developer-guide flow.

## Shipped slugs

| slug | description |
| --- | --- |
| `flashvsr-v1.1-sparse-ratio-2.0` | FlashVSR-v1.1 streaming video super-resolution (2x; `sparse_ratio=2.0` stable preset). |
| `flashvsr-v1.1-sparse-ratio-1.5` | FlashVSR-v1.1 streaming video super-resolution (2x; `sparse_ratio=1.5` faster preset). |

## Install

The plugin is registered as a `uv` workspace member in the repo-root
`pyproject.toml`, so a single `uv sync` from the repo root pulls it in:

```bash
uv sync
```

Standalone (outside the workspace) also works:

```bash
uv pip install -e integrations/flashvsr
```

## HuggingFace setup

Checkpoints are auto-downloaded from HuggingFace at first run. Set an
auth token first.

```bash
# huggingface token.
export HF_TOKEN=<your-hf-token>

# (optional) override the cache location.
export HF_HOME=~/.cache/huggingface  # default
```

## Single-GPU Run

Once installed, the slug is discovered automatically by `flashdreams-run`:

```bash
# List every registered runner (this plugin's slugs appear under "flashvsr-*").
uv run flashdreams-run --help

# Per-runner help: every overridable field is a CLI flag.
uv run flashdreams-run flashvsr-v1.1-sparse-ratio-2.0 --help

# Single-GPU run (stable sparse_ratio=2.0); pipeline ``input_H`` /
# ``input_W`` are auto-set to the input video's native dims. The DiT
# requires the upres dims to be divisible by 128; the encoder
# bicubic-upsamples to ``(H * scale, W * scale)`` and then
# center-crops to the largest 128-multiple. So 704x1280
# (-> 1408x2560, no trim) and 540x960 (-> 1080x1920 -> 1024x1920,
# 32 px top + 32 px bottom trim) both work at the default
# ``scale=2``; inputs need only be at least ``128 / scale = 64``
# pixels on each axis.
uv run flashdreams-run flashvsr-v1.1-sparse-ratio-2.0 --input-path /path/to/clip.mp4

# Faster preset (sparse_ratio=1.5).
uv run flashdreams-run flashvsr-v1.1-sparse-ratio-1.5 --input-path /path/to/clip.mp4

# Dense full-attention preset. This changes model behavior, enables DiT
# torch.compile + CUDA graph by default, and supports multi-GPU CP.
uv run flashdreams-run flashvsr-v1.1-full-attn --input-path /path/to/clip.mp4

# Reduce per-chunk peak VRAM (single DiT iter per chunk: first=5, subseq=8).
uv run flashdreams-run flashvsr-v1.1-sparse-ratio-2.0 --input-path /path/to/clip.mp4 \
    --chunk-size 8

# Strip the HDMap visualization off Omnidreams outputs before upscaling.
uv run flashdreams-run flashvsr-v1.1-sparse-ratio-2.0 \
    --input-path /path/to/omnidreams.mp4 \
    --crop-region bottom_half

# Pick a different upscale factor (the only resolution-side knob; the
# runner ignores ``--pipeline.encoder.input-H/-W`` overrides because
# they're auto-derived from the input video).
uv run flashdreams-run flashvsr-v1.1-sparse-ratio-2.0 --input-path /path/to/clip.mp4 \
    --pipeline.encoder.scale 4
```

## Multi-GPU Run via context-parallelism:

Supported only by the dense full-attention preset. The legacy sparse presets
remain single-GPU because `block_sparse_attn` is not context-parallel aware
and is intentionally not being extended here.

```bash
uv run torchrun --nproc_per_node=2 --no-python flashdreams-run \
    flashvsr-v1.1-full-attn --input-path /path/to/clip.mp4
```

Full attention uses the existing CP-aware Wan dense attention stack with DiT
compile + CUDA graph enabled by default, so peak memory is much higher than the
sparse presets. Use smaller inputs, `--chunk-size 8`, fewer GPUs, or override
`--pipeline.diffusion-model.transformer.use-cuda-graph False` if the dense run
OOMs.

## gRPC server and browser viewer

The FlashVSR integration also ships a gRPC upsampler service that keeps the
pipeline warm, accepts incoming frame chunks, and can publish an MJPEG browser
viewer. The service supports unary chunks (`start_session` / `upscale_chunk` /
`end_session`) and bidirectional streaming (`upscale_video`). Streaming clients
may send 8-frame chunks at live-ingest cadence; the server coalesces those into
FlashVSR-compatible 13-frame cold-start and 16-frame steady-state model calls.

Start a server with an HTTP viewer on port 8080:

```bash
PYTHONPATH=integrations/flashvsr:flashdreams \
uv run --no-sync python -m flashvsr.grpc.uplift_server \
    --port 50051 \
    --viewer_port 8080 \
    --cuda_graph \
    --attention_mode auto \
    --sparse_ratio 1.5
```

Then open `http://<server-host>:8080/` in a browser. The page shows received
input frames and upsampled frames side by side. By default, viewer mode omits
raw output frame bytes from gRPC responses to keep server-to-client bandwidth
low; pass `--viewer_return_grpc_frames` if a client also needs the RGB payloads.
`--attention_mode auto` uses the sparse backend when `block_sparse_attn` is
installed and falls back to the dense full-attention backend otherwise. To force
the dense path in an environment without `block_sparse_attn`, pass
`--attention_mode full`.

Use the live-ingest client to loop a video into the server at 30 fps:

```bash
PYTHONPATH=integrations/flashvsr:flashdreams \
uv run --no-sync python -m flashvsr.grpc.uplift_client \
    --continuous \
    --server localhost:50051 \
    --input /path/to/clip.mp4

# finite smoke test:
PYTHONPATH=integrations/flashvsr:flashdreams \
uv run --no-sync python -m flashvsr.grpc.uplift_client \
    --continuous \
    --server localhost:50051 \
    --input /path/to/clip.mp4 \
    --max_chunks 4
```

For a save-to-disk test client, use:

```bash
PYTHONPATH=integrations/flashvsr:flashdreams \
uv run --no-sync python -m flashvsr.grpc.uplift_client \
    --server localhost:50051 \
    --input /path/to/clip.mp4 \
    --output /tmp/clip_2x.mp4
```


## Streaming chunk contract

`FlashVSRPipeline.generate(autoregressive_index, cache, input)` processes
**one full FlashVSR chunk** per call. The encoder accepts the four
(raw_T -> padded_T) pairs from `FLASHVSR_CHUNK_FRAME_TARGETS`:

| raw_T | padded_T | when | DiT iters per chunk |
|------:|---------:|------|--------------------:|
| `5`   | `8`      | cold-start (`autoregressive_index == 0`) | 1 |
| `8`   | `8`      | any AR step                              | 1 |
| `13`  | `16`     | cold-start (`autoregressive_index == 0`) | 2 |
| `16`  | `16`     | any AR step                              | 2 |

Cold-start sizes are pad-left replicated inside `FlashVSREncoder` so the
projector's 4-frame causal stride aligns. The DiT runs `T_padded // 8`
internal iterations against per-iter (2-latent-frame) noise slices and
LR-latent token slices; the rolling KV cache holds `kv_ratio + 1` chunks
at attention time (default `kv_ratio = 3` -> 4 chunks).

The `--chunk-size` runner flag picks the steady-state size (`8` or
`16`); the cold-start size (`5` or `13`) is auto-derived to match.
`flashvsr.runner._build_chunks(total_frames, first_size, subseq_size)`
produces the `(start, size)` pairs that feed `pipeline.generate`.

## Programmatic access

Access via runner.

```python
from flashvsr.config import RUNNER_FLASHVSR_V1_1_SPARSE_2_0 as runner_config
from flashdreams.infra.config import derive_config

cfg = derive_config(runner_config, input_path="/path/to/clip.mp4")
runner = cfg.setup()
runner.run()
```

Access via pipeline. The shipped `PIPELINE_FLASHVSR_V1_1_SPARSE_*` literals pin
``input_H=704`` / ``input_W=1280`` as a placeholder; callers bypassing
the runner must build their own config with the actual video's dims
(or ``derive_config`` the literal). The encoder handles non-128-aligned
upres dims via a symmetric center-crop after bicubic (matching upstream
FlashVSR's ``upscale_then_center_crop``); ``topk_ratio`` is baked at
builder time from the **post-crop** target, so passing the actual
video's ``(input_H, input_W)`` to :func:`build_flashvsr_v1_1` yields a
``topk_ratio`` that matches what upstream would compute. The runner
re-derives ``topk_ratio`` at run time from ``FlashVSRRunnerConfig.sparse_ratio``
in case the pipeline literal was built at a different placeholder.

```python
import mediapy as media
import numpy as np
import torch
from einops import rearrange

from flashvsr.config import build_flashvsr_v1_1
from flashvsr.runner import _CHUNK_MODES, _build_chunks

video_np = media.read_video("/path/to/clip.mp4")[..., :3]  # uint8 [T, H, W, C]
T, H, W, _ = video_np.shape

pipeline_config = build_flashvsr_v1_1(
    input_H=H,
    input_W=W,
    scale=2,
    compile_network=True,
    use_cuda_graph=True,
)
pipeline = pipeline_config.setup().to("cuda").eval()
cache = pipeline.initialize_cache()

dtype = pipeline.diffusion_model.dtype
device = pipeline.device

video_t = (
    torch.from_numpy(video_np.astype(np.float32)) / 127.5 - 1.0
).to(device=device, dtype=dtype)
video_t = rearrange(video_t, "T H W C -> 1 C T H W")

first_size, subseq_size = _CHUNK_MODES[16]  # (13, 16)
chunks = _build_chunks(video_t.shape[2], first_size, subseq_size)

generated_chunks: list[torch.Tensor] = []
for i, (start, size) in enumerate(chunks):
    clip = video_t[:, :, start : start + size]
    out = pipeline.generate(autoregressive_index=i, cache=cache, input=clip)
    pipeline.finalize(autoregressive_index=i, cache=cache)
    generated_chunks.append(out.cpu())  # each is [1, C, T_out, H_out, W_out]
```

## Builder knobs

`build_flashvsr_v1_1` is the single entry point for assembling a
`FlashVSRPipelineConfig` by hand. The most common knobs:

- `input_H`, `input_W`: low-res input dimensions. Output dims are
  `((input_H * scale) // 128 * 128, (input_W * scale) // 128 * 128)`:
  the encoder bicubic-upsamples to `(input_H * scale, input_W * scale)`
  and symmetric-crops to the largest 128-multiple per axis (matching
  upstream's `upscale_then_center_crop` in
  `examples/WanVSR/infer_flashvsr_v1.1_tiny.py`). Inputs need only be
  at least `128 / scale = 64` pixels on each axis at the default
  `scale=2`.
- `scale`: `2` (default) or `4`.
- `sparse_ratio`: block-sparse attention budget multiplier. `2.0`
  (default, "more stable") or `1.5` ("faster" preset).
- `compile_network`: single `torch.compile` switch applied uniformly to
  the DiT, encoder projector, and decoder.
- `use_cuda_graph`: capture the steady-state DiT call into a CUDA graph
  and replay it (Phase 2 of `internal/upsampler/PERF_NOTES.md`). Requires
  `compile_network=True`. Encoder / decoder cuda graphs are always on
  inside the builder. Defaults to `False`; flip on per-resolution in the
  gRPC server until proven stable.
- `color_corrector_implementation`: `"cuda"` (default; AdaIN-only
  hand-rolled kernel) or `"torch"` (pure-torch wavelet + AdaIN reference).
- `enable_sync_and_profile`: per-AR-step CUDA-event profiling. Adds one
  `cuda.synchronize()` per step.

## Files

| Path | Purpose |
|---|---|
| `flashvsr/pipeline.py` | `FlashVSRPipeline` + `FlashVSRPipelineConfig` (5-step `generate`; 7 profiler events). |
| `flashvsr/runner.py` | `FlashVSRRunner` + `FlashVSRRunnerConfig` (chunked video I/O around `pipeline.generate`). |
| `flashvsr/config.py` | `build_flashvsr_v1_1`, the `PIPELINE_*` / `RUNNER_*` literals, and the `FLASHVSR_CONFIG_BUILDERS` / `RUNNER_CONFIGS` registries. |
| `flashvsr/constants.py` | Chunk-target table, decoder channel split, conditioning patch sizes. |
| `flashvsr/encoder/__init__.py`, `flashvsr/encoder/network.py` | Bicubic upres + `Causal_LQ4x_Proj` LR projector. |
| `flashvsr/transformer/__init__.py`, `flashvsr/transformer/network.py` | `FlashVSRTransformer` (Wan 2.1 subclass) + sparse-attention DiT. |
| `flashvsr/decoder/__init__.py`, `flashvsr/decoder/network.py` | TC decoder (TAEHV) + AdaIN color corrector wrapper. |
| `flashvsr/corrector.py` | `FlashVSRColorCorrector` dispatcher (cuda + torch backends). |
| `flashvsr/csrc/color_corrector_adain_cuda.cu` | Hand-rolled AdaIN CUDA extension. |
| `run_flashvsr.py` | Standalone argparse CLI; kept for backward compatibility with the pre-runner workflow. |

## Tests

CPU smoke + parity tests live under `integrations/flashvsr/tests/`:

```bash
uv run --extra dev pytest integrations/flashvsr/tests -v
```

The CUDA / weight-gated parity tests (`test_projector_*`,
`test_color_corrector_benchmark.py`) auto-skip when GPU or staged
FlashVSR-v1.1 weights are missing. The DiT-side parity check
(`parity_check/test_dit_parity.py`) and the TC decoder parity check
(`parity_check/test_tcdecoder_parity.py`) live next to upstream's
cloned source tree and are invoked from `parity_check/run.sh` (see
below) so both legacy (upstream) and candidate (flashdreams) sides are
loaded from a single parity-check venv.

Upstream parity benchmark + DiT / TC decoder parity tests (clones
FlashVSR at a pinned commit, applies a local patch that adds
`EventProfiler`-instrumented per-chunk timing, runs both parity tests
against upstream's `diffsynth.models.wan_video_dit.WanModel` +
`diffsynth.pipelines.flashvsr_tiny_long.model_fn_wan_video` and
`examples/WanVSR/utils/TCDecoder.py`, then runs the upstream pipeline
end-to-end):

```bash
bash integrations/flashvsr/tests/parity_check/run.sh
```

See [`integrations/flashvsr/tests/parity_check/README.md`](tests/parity_check/README.md)
for the JSON-stats schema, both parity tests, and the one-time
`Block-Sparse-Attention` install.
