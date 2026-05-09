# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CLI to upsample a video using the FlashVSR streaming pipeline.

Granularity contract: one ``pipeline.generate()`` call processes one
full FlashVSR chunk. ``--chunk_size 16`` (default, packs two DiT iters
per chunk; first=13/subseq=16 raw frames) or ``--chunk_size 8`` (one DiT
iter per chunk; first=5/subseq=8). The cold-start sizes (5 / 13) are
pad-left replicated to 8 / 16 inside :class:`FlashVSREncoder`.

Mirrors the legacy ``UltraFlashVSRUpsampler.forward`` contract exactly
(see ``internal/upsampler/ultraflashvsr/_wan_model_dit.py`` parity
reference + ``pipeline.py`` for the per-iter algorithm).

Example::

    uv run python flashdreams/examples/run_flashvsr.py \\
        --input outputs/clip.mp4 --output outputs/clip_2x.mp4 \\
        --scale 2 --chunk_size 16
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict

import mediapy as media
import numpy as np
import torch
from einops import rearrange

from flashdreams.infra.profiler import EventProfiler
from flashdreams.recipes.flashvsr.config import build_flashvsr_v1_1
from flashdreams.recipes.flashvsr.encoder import FlashVSREncoder


def _chunk_modes() -> dict[int, tuple[int, int]]:
    """``{steady_size: (cold_size, steady_size)}`` derived from the encoder.

    Single-source-of-truth: invert :data:`FlashVSREncoder._CHUNK_FRAME_TARGETS`
    (``{raw -> padded}``) into a ``{padded -> raw_cold}`` map and pair each
    ``padded`` with itself as the steady size. Currently yields
    ``{8: (5, 8), 16: (13, 16)}`` -- the legacy
    ``_CHUNK_TARGET = {5: 8, 13: 16, 8: 8, 16: 16}`` table.
    """
    targets = FlashVSREncoder._CHUNK_FRAME_TARGETS
    cold_for: dict[int, int] = {}
    for raw, padded in targets.items():
        if raw != padded:
            assert padded not in cold_for or cold_for[padded] == raw, (
                f"Multiple cold-start sizes map to padded {padded}: "
                f"{cold_for.get(padded)} and {raw}"
            )
            cold_for[padded] = raw
    return {padded: (cold_for.get(padded, padded), padded) for padded in cold_for}


_CHUNK_MODES: dict[int, tuple[int, int]] = _chunk_modes()


def build_chunks(
    total_frames: int, first_size: int, subseq_size: int
) -> list[tuple[int, int]]:
    """Return list of (start, size) pairs for each AR step.

    The first chunk is ``first_size`` frames (cold-start; pad-left
    replicated by the encoder); subsequent chunks are ``subseq_size`` each.
    A trailing partial chunk is dropped with a warning.
    """
    chunks: list[tuple[int, int]] = []
    pos = 0
    first = True
    while pos < total_frames:
        target = first_size if first else subseq_size
        size = min(target, total_frames - pos)
        if size < target:
            print(
                f"Warning: trailing chunk has {size} frames "
                f"(need {target}). Truncating video to {pos} frames.",
                file=sys.stderr,
            )
            break
        chunks.append((pos, size))
        pos += size
        first = False
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FlashVSR: streaming video super-resolution"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input video path (.mp4 or other mediapy-readable format)",
    )
    parser.add_argument("--output", required=True, help="Output video path (.mp4)")
    parser.add_argument(
        "--scale",
        type=int,
        default=2,
        choices=[2, 4],
        help="Spatial upscale factor (default: 2)",
    )
    parser.add_argument(
        "--sparse_ratio",
        type=float,
        default=2.0,
        help="Attention sparsity ratio: 1.5=faster, 2.0=more stable (default: 2.0)",
    )
    parser.add_argument(
        "--fps", type=float, default=None, help="Output FPS (default: same as input)"
    )
    parser.add_argument("--kv_ratio", type=int, default=3)
    parser.add_argument("--local_range", type=int, default=11)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model compute dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--crop_region",
        default="none",
        choices=["none", "bottom_half", "top_half"],
        help=(
            "Crop input frames before upsampling. Use `bottom_half` to drop the "
            "HDMap visualization stacked on top of Alpadreams outputs and "
            "upscale only the generated RGB (default: none)."
        ),
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=16,
        choices=sorted(_CHUNK_MODES.keys()),
        help=(
            "Steady-state frames per AR step (cold-start uses chunk_size - 3). "
            "16 (default): packs two DiT iters per pipeline.generate() call "
            "(first=13, subseq=16). 8: one DiT iter per call (first=5, "
            "subseq=8). 8 roughly halves per-chunk peak VRAM at the cost of "
            "more boundary stitching overhead."
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable per-AR-step component profiling.",
    )
    parser.add_argument(
        "--color_corrector_implementation",
        default="cuda",
        choices=["cuda", "torch"],
        help=(
            "Color corrector backend: cuda (default; AdaIN-only hand-rolled "
            "kernel) or torch (pure-torch wavelet + AdaIN reference)."
        ),
    )
    parser.add_argument("--stats_json", default=None)
    args = parser.parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]
    device = torch.device("cuda")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Read input video.
    print(f"Reading {args.input} ...")
    video_np = media.read_video(args.input)  # uint8 [T, H, W, C]
    T, H, W, _ = video_np.shape
    print(f"  {T} frames, {H}x{W}")

    # Optional vertical crop.
    if args.crop_region != "none":
        H_half = H // 2
        if args.crop_region == "bottom_half":
            video_np = video_np[:, H - H_half :, :, :]
        else:
            video_np = video_np[:, :H_half, :, :]
        H = video_np.shape[1]
        print(f"  cropped to {args.crop_region}: now {H}x{W}")

    input_fps = args.fps
    if input_fps is None:
        try:
            # ``mediapy`` ships without type stubs so ty cannot see the
            # ``VideoMetadata.from_path`` classmethod (added in mediapy 1.1).
            meta = media.VideoMetadata.from_path(args.input)  # ty: ignore[unresolved-attribute]
            input_fps = meta.fps
        except Exception:
            input_fps = 30.0
        print(f"  fps: {input_fps}")

    first_size, subseq_size = _CHUNK_MODES[args.chunk_size]
    chunks = build_chunks(T, first_size, subseq_size)
    if not chunks:
        print(
            f"Error: video is too short to process "
            f"(need at least {first_size} frames for chunk_size={args.chunk_size}).",
            file=sys.stderr,
        )
        sys.exit(1)

    usable_frames = sum(s for _, s in chunks)
    if usable_frames < T:
        print(f"  Using first {usable_frames} of {T} frames.")
        video_np = video_np[:usable_frames]

    # uint8 [T, H, W, C] in [0, 255] -> bf16 [1, C, T, H, W] in [-1, 1].
    video_t = (torch.from_numpy(video_np.astype(np.float32)) / 127.5 - 1.0)
    video_t = rearrange(video_t.to(device=device, dtype=dtype), "T H W C -> 1 C T H W")

    print(f"\nLoading FlashVSR pipeline (scale={args.scale}) ...")
    pipeline_config = build_flashvsr_v1_1(
        input_H=H,
        input_W=W,
        scale=args.scale,
        sparse_ratio=args.sparse_ratio,
        kv_ratio=args.kv_ratio,
        local_range=args.local_range,
        compile_network=True,
        use_cuda_graph=True,
        color_corrector_implementation=args.color_corrector_implementation,
        enable_sync_and_profile=args.profile,
        dtype=dtype,
        seed=args.seed,
    )
    pipeline = pipeline_config.setup().to(device=device)
    cache = pipeline.initialize_cache()

    print(
        f"\nProcessing {len(chunks)} chunk(s) -> {H * args.scale}x{W * args.scale} ..."
    )
    chunks_out: list[np.ndarray] = []
    profilers: list[EventProfiler] = []
    component_stats: list[dict[str, float]] = []
    for chunk_idx, (start, size) in enumerate(chunks):
        clip = video_t[:, :, start : start + size]
        tic = time.time()
        evt = EventProfiler()
        out = pipeline.generate(
            autoregressive_index=chunk_idx,
            cache=cache,
            input=clip,
        )
        finalize_stats = pipeline.finalize(autoregressive_index=chunk_idx, cache=cache)
        if finalize_stats is not None:
            component_stats.append(finalize_stats)
        evt.record("upsample")
        # [1, 3, T, H', W'] -> uint8 [T, H', W', 3] on host.
        out = ((out.float() + 1.0) * 127.5).clamp(0, 255).byte()
        out = rearrange(out, "1 C T H W -> T H W C").contiguous().cpu().numpy()
        evt.record("rearrange")
        chunks_out.append(out)
        del clip
        torch.cuda.empty_cache()
        elapsed = time.time() - tic
        out_frames = chunks_out[-1].shape[0]
        print(
            f"  Chunk {chunk_idx + 1}/{len(chunks)}: "
            f"frames {start}-{start + size - 1} ({out_frames} out) {elapsed:.2f}s"
        )
        profilers.append(evt)

    result = np.concatenate(chunks_out, axis=0)  # [T_out, H', W', 3]

    SKIP_PROFILE = 4

    latencies = defaultdict(list)
    numframes = sum(chunk.shape[0] for chunk in chunks_out[SKIP_PROFILE:])
    for p in profilers[SKIP_PROFILE:]:
        s = p.sync_and_summarize()
        for k, v in s.items():
            latencies[k].append(v)
    print(f"\nLatencies (skip first {SKIP_PROFILE} chunk(s)):")
    for k, v in latencies.items():
        print(f" ~ {k}: {sum(v) / len(v):.2f} ms")
    if latencies.get("upsample"):
        fps = numframes / sum(latencies["upsample"]) * 1000
        print(f"Upsampling FPS: {fps:.2f}")

    if args.profile and component_stats:
        component_summary = defaultdict(list)
        for stats in component_stats[SKIP_PROFILE:]:
            for k, v in stats.items():
                component_summary[k].append(v)
        if component_summary:
            # Stage stats are returned by FlashVSRPipeline.finalize with their
            # unit baked into the key (``<stage>_ms``, ``total_ms_wo_finalize``,
            # ``mem_*_gib``); strip the unit token and print the matching unit
            # so we don't render ``pad_ms: 0.05 ms`` (duplicated unit) or
            # ``mem_alloc_gib: 28.41 ms`` (wrong unit).
            _UNIT_TOKENS = (("_ms", "ms"), ("_gib", "GiB"))
            print(f"\nComponent latencies (skip first {SKIP_PROFILE} chunk(s)):")
            for k, v in component_summary.items():
                name, unit = k, ""
                for token, token_unit in _UNIT_TOKENS:
                    if token in k:
                        name = k.replace(token, "")
                        unit = token_unit
                        break
                print(f" ~ {name}: {sum(v) / len(v):.2f} {unit}".rstrip())

    if args.stats_json is not None:
        stats_payload = {
            "args": vars(args),
            "chunks": [{"start": start, "size": size} for start, size in chunks],
            "latencies_ms": {k: list(v) for k, v in latencies.items()},
            "component_stats": component_stats,
        }
        with open(args.stats_json, "w") as f:
            json.dump(stats_payload, f, indent=2)
        print(f"Wrote stats JSON: {args.stats_json}")

    out_path = args.output
    if not out_path.endswith(".mp4"):
        out_path += ".mp4"
    print(f"\nSaving {result.shape[0]} frames to {out_path} @ {input_fps} fps ...")
    media.write_video(out_path, result, fps=input_fps)
    print("Done.")


if __name__ == "__main__":
    main()
