# `flashvsr` recipe

Streaming video super-resolution on top of FlashVSR-v1.1 (LR projector +
distilled Wan 2.1 DiT + TC decoder + AdaIN color corrector). Wraps
everything in a `StreamInferencePipeline` so the same `generate` /
`finalize` lifecycle as the other recipes (`alpadreams`,
`lingbot_world`, `wan2_1`) applies.

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

`flashdreams/examples/run_flashvsr.py` is the reference loop:
`build_chunks(total_frames, first_size, subseq_size)` produces the
`(start, size)` pairs that feed `pipeline.generate`.

## Files

| Path | Purpose |
|---|---|
| `pipeline.py` | `FlashVSRPipeline` + `FlashVSRPipelineConfig` (5-step `generate`; 7 profiler events). |
| `config.py` | `build_flashvsr_v1_1` and the `FLASHVSR_CONFIG_BUILDERS` registry. |
| `constants.py` | Chunk-target table, decoder channel split, conditioning patch sizes. |
| `encoder/__init__.py`, `encoder/network.py` | Bicubic upres + `Causal_LQ4x_Proj` LR projector. |
| `transformer/__init__.py`, `transformer/network.py` | `FlashVSRTransformer` (Wan 2.1 subclass) + sparse-attention DiT. |
| `decoder/__init__.py`, `decoder/network.py` | TC decoder (TAEHV) + AdaIN color corrector wrapper. |
| `corrector.py` | `FlashVSRColorCorrector` dispatcher (cuda + torch backends). |
| `csrc/color_corrector_adain_cuda.cu` | Hand-rolled AdaIN CUDA extension. |

## Builder knobs

`build_flashvsr_v1_1` is the single entry point. The most common knobs:

- `input_H`, `input_W`: low-res input dimensions; output is
  `(input_H * scale, input_W * scale)`. Both must be divisible by
  `128 / scale`.
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

The weight cache root is fixed at
`$FLASHDREAMS_CACHE_DIR/upsampler/weights/FlashVSR-v1.1` (default
`~/.cache/flashdreams/upsampler/weights/FlashVSR-v1.1`). Stage weights
with `internal/upsampler/scripts/download_flashvsr_weights.sh`.

## Entry point

```bash
# CLI inference (writes <input_stem>_<scale>x.mp4 next to the input by default).
uv run python flashdreams/examples/run_flashvsr.py \
    --input outputs/clip.mp4 --output outputs/clip_2x.mp4 --scale 2

# Or via the wrapper that also stages weights + builds block-sparse-attn:
./internal/upsampler/scripts/upsample_video.sh --input outputs/clip.mp4 --scale 2

# CPU smoke (no weights needed):
uv run --extra dev pytest flashdreams/tests/test_flashvsr.py -v
```
