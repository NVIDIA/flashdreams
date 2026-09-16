# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline capture harness for the latent-native caption model (Phase 1).

Drives the omnidreams world model headless with a scripted, **road-keeping**
WSAD control track and, per rolling K-chunk window, persists a self-contained
training sample:

- ``latents.npy`` — the latent window ``[K*T, 16, 88, 160]`` fp16, taken from
  ``generate_chunk``'s ``latent_chunk`` and narrowed to fp16 exactly as the
  ``raw_f16`` token codec would (so capture ≡ what the browser client streams).
- ``rgb.NNN.jpg`` — evenly-spaced decoded RGB frames spanning the same window
  (for visual QA, and optional VLM-teacher labeling).
- ``json`` — metadata incl. ``label`` (the per-window ego-motion class, derived
  from the controls) + per-chunk WSAD, chunk range, scene, seed.

Open-loop yaw cannot hold a lane, so sustaining one turn direction drives the
ego off-road into buildings (degenerate footage). The plan instead uses short,
self-correcting turn "weaves" that keep the ego on-road — so the control state
is accurate ground truth for the 6 ego-motion classes, labels are written
directly here, and **no VLM teacher is required**. A detail-based guard drops
any residual degenerate (off-road / near-uniform) windows.

Samples are written as WebDataset-compatible tar shards via
:class:`omnidreams.caption.shards.ShardWriter`.

GPU-only: constructing the runtime loads checkpoints and builds CUDA graphs.
Run on the inference host, e.g.::

    omnidreams-caption-capture --out DATA/raw --limit 500 --loop
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from flashdreams.serving.realtime.input import WSAD_SUPPORTED_KEYS, KeyboardResampler
from omnidreams.caption.shards import ShardWriter
from omnidreams.webrtc.session import (
    OmnidreamsInferenceRuntime,
    OmnidreamsRuntimeConfig,
)

#: The 6-class ego-motion caption bank (Phase 0). The per-window ``label`` and
#: the per-segment plan labels are drawn from these.
CAPTION_CLASSES = (
    "driving_straight",
    "turning_left",
    "turning_right",
    "slowing_or_stopping",
    "stopped",
    "reversing",
)


@dataclass(frozen=True)
class PlanSegment:
    """One scripted stretch: hold ``keys`` for ``chunks`` chunks, labeled ``label``."""

    label: str
    keys: frozenset[str]
    chunks: int


def default_plan(turn: int = 3, straight: int = 4) -> list[PlanSegment]:
    """Road-keeping control track: keeps the ego on-road so control == label.

    Open-loop yaw cannot hold a lane, so instead of sustaining one direction
    (which drives the ego off-road into buildings), turns come as short
    forward-arc *weaves* — equal-length left then right — whose headings cancel,
    keeping the ego near the lane center. A short longitudinal block covers
    slowing / stopped / reversing and returns toward the start position.

    ``turn`` and ``straight`` are burst lengths (tune via a pilot: longer turns
    give clearer turning windows but risk drifting off-road).
    """
    fs = frozenset
    fwd_l, fwd_r = fs({"w", "a"}), fs({"w", "d"})
    fwd, coast, back = fs({"w"}), fs(), fs({"s"})
    return [
        PlanSegment("driving_straight", fwd, straight),
        # weave: alternating equal left/right forward-arcs (net-zero heading)
        PlanSegment("turning_left", fwd_l, turn),
        PlanSegment("turning_right", fwd_r, turn),
        PlanSegment("turning_left", fwd_l, turn),
        PlanSegment("turning_right", fwd_r, turn),
        PlanSegment("driving_straight", fwd, straight),
        # longitudinal: decelerate -> rest -> short reverse -> recover forward
        PlanSegment("slowing_or_stopping", coast, 3),
        PlanSegment("stopped", coast, 3),
        PlanSegment("reversing", back, turn),
        PlanSegment("driving_straight", fwd, straight),
    ]


def load_plan(path: Path) -> list[PlanSegment]:
    """Load a control track from JSON: ``[{"label","keys":[...],"chunks"}, ...]``."""
    raw = json.loads(path.read_text())
    return [
        PlanSegment(seg["label"], frozenset(seg.get("keys", [])), int(seg["chunks"]))
        for seg in raw
    ]


def _latents_to_npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _latent_chunk_to_f16(latent_chunk: torch.Tensor) -> np.ndarray:
    """``[B,V,T,Cl,Hl,Wl]`` -> ``[T,Cl,Hl,Wl]`` fp16 (byte-identical to the wire)."""
    frames = latent_chunk[0, 0]
    return frames.detach().to("cpu", torch.float16).numpy()


def _rgb_chunk_to_uint8(video_chunk: torch.Tensor) -> np.ndarray:
    """``[B,V,T,3,H,W]`` uint8 RGB -> ``[T,H,W,3]`` uint8 RGB."""
    frames = video_chunk[0, 0].permute(0, 2, 3, 1)
    return frames.detach().to("cpu", torch.uint8).numpy()


def _encode_jpeg(frame_rgb: np.ndarray, quality: int, max_side: int | None) -> bytes:
    img = frame_rgb
    if max_side is not None:
        h, w = img.shape[:2]
        scale = max_side / max(h, w)
        if scale < 1.0:
            img = cv2.resize(
                img,
                (round(w * scale), round(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed for an RGB frame")
    return buf.tobytes()


def _evenly_spaced(n_total: int, n_pick: int) -> list[int]:
    if n_pick >= n_total:
        return list(range(n_total))
    if n_pick == 1:
        return [n_total // 2]
    return [round(i * (n_total - 1) / (n_pick - 1)) for i in range(n_pick)]


def _dominant_label(labels: list[str]) -> str:
    # Window label = the most common per-chunk control label; ties -> latest.
    counts = Counter(labels)
    best = max(counts.values())
    for label in reversed(labels):
        if counts[label] == best:
            return label
    return labels[-1]


def _window_detail(rgb_frames: np.ndarray) -> float:
    """Mean variance-of-Laplacian over a window's frames (downscaled to 256 px).

    A spatial-detail proxy: on-road scenes (buildings, road texture, lane lines)
    score high (hundreds–thousands); degenerate frames — off-road into a
    featureless field/sky, or blurred long-rollout output — score very low
    (single/low-double digits). Used to drop those windows via ``--min-detail``.
    Downscaling first keeps the score cheap and stable across resolutions.
    """
    scores = []
    for frame in rgb_frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        scale = 256.0 / max(h, w)
        if scale < 1.0:
            gray = cv2.resize(
                gray,
                (round(w * scale), round(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        scores.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    return float(np.mean(scores)) if scores else 0.0


async def _step(
    runtime: OmnidreamsInferenceRuntime,
    resampler: KeyboardResampler,
    held: set[str],
    target_keys: frozenset[str],
    label: str | None,
) -> dict:
    """Set the control state, advance one chunk, return its latent + RGB record."""
    num_frames = runtime.peek_next_chunk_num_frames()
    # Apply key edges just before this chunk's start so they land in the
    # carried state and the whole chunk is one segment with ``target_keys``.
    edge_t = resampler.next_chunk_start_v - 1e-3
    for key in held - target_keys:
        resampler.on_edge(arrival_t=edge_t, event="keyup", key=key)
    for key in target_keys - held:
        resampler.on_edge(arrival_t=edge_t, event="keydown", key=key)
    held.clear()
    held.update(target_keys)

    segments, frame_times = resampler.sample_chunk(num_frames)
    result = await runtime.generate_chunk(segments=segments, frame_times=frame_times)
    return {
        "latent": _latent_chunk_to_f16(result.metadata["latent_chunk"]),
        "rgb": _rgb_chunk_to_uint8(result.video_chunk),
        "keys": sorted(target_keys),
        "label": label,
        "chunk_index": int(result.chunk_index),
    }


def _emit_window(
    writer: ShardWriter,
    recs: list[dict],
    win_id: int,
    session_tag: str,
    args: argparse.Namespace,
) -> bool:
    """Write one window; return ``False`` (writing nothing) if it's degenerate."""
    latents = np.concatenate([r["latent"] for r in recs], axis=0)
    rgb_all = np.concatenate([r["rgb"] for r in recs], axis=0)
    picked_rgb = rgb_all[_evenly_spaced(rgb_all.shape[0], args.rgb_frames)]

    # Drop off-road / near-uniform windows the road-keeping plan didn't avoid.
    if args.min_detail > 0.0 and _window_detail(picked_rgb) < args.min_detail:
        return False

    parts: dict[str, bytes] = {"latents.npy": _latents_to_npy_bytes(latents)}
    for i, frame in enumerate(picked_rgb):
        parts[f"rgb.{i:03d}.jpg"] = _encode_jpeg(
            frame, args.jpeg_quality, args.rgb_max_side
        )

    label = _dominant_label([r["label"] for r in recs])
    meta = {
        "session_id": session_tag,
        "window_id": win_id,
        "label": label,
        "label_source": "control",
        "num_latent_frames": int(latents.shape[0]),
        "latent_shape": [int(d) for d in latents.shape[1:]],
        "window_chunks": args.window_chunks,
        "chunk_index_range": [recs[0]["chunk_index"], recs[-1]["chunk_index"]],
        "rgb_frames": int(picked_rgb.shape[0]),
        "rgb_source_hw": [int(rgb_all.shape[1]), int(rgb_all.shape[2])],
        "wsad_per_chunk": [r["keys"] for r in recs],
        "label_per_chunk": [r["label"] for r in recs],
        "scene_uuid": args.scene_uuid or "default",
        "scene_variant": args.scene_variant,
        "seed": args.seed,
        "pipeline_config_name": args.pipeline_config_name,
    }
    parts["json"] = json.dumps(meta).encode("utf-8")
    writer.write(
        f"{session_tag}-{win_id:07d}",
        parts,
        meta={
            "session_id": session_tag,
            "label": label,
            "chunk_index_range": meta["chunk_index_range"],
        },
    )
    return True


async def run_capture(args: argparse.Namespace) -> None:
    cfg_kwargs: dict = {
        "pipeline_config_name": args.pipeline_config_name,
        "seed": args.seed,
        "device": args.device,
        "scene_uuid": args.scene_uuid,
        "enable_token_stream": False,  # latent_chunk is read directly, no socket
        "debug_serve_hdmaps": False,  # ensure real RGB + latents
    }
    if args.scene_variant is not None:
        cfg_kwargs["scene_variant"] = args.scene_variant
    cfg = OmnidreamsRuntimeConfig(**cfg_kwargs)

    runtime = OmnidreamsInferenceRuntime(config=cfg)
    await runtime.initialize()

    plan = (
        load_plan(Path(args.plan))
        if args.plan
        else default_plan(turn=args.turn_burst, straight=args.straight_run)
    )
    resampler = KeyboardResampler(
        fps=float(cfg.fps), supported_keys=WSAD_SUPPORTED_KEYS
    )
    held: set[str] = set()

    # Discard warmup chunks (first GPU runs include compile/autotune warmup).
    for _ in range(args.warmup_chunks):
        await _step(runtime, resampler, held, frozenset({"w"}), label=None)

    # Unique per-run tag: names this run's shards + window keys and is the
    # session key for the Phase-3 (split-by-session) join. Auto-derived from
    # scene/variant/seed unless given, so repeated runs never collide.
    scene = (args.scene_uuid or "default").replace(".", "_")[:12]
    variant = args.scene_variant or "default"
    session_tag = args.session_id or f"{scene}-{variant}-s{args.seed}"
    session_tag = session_tag.replace(".", "_")

    window: deque[dict] = deque(maxlen=args.window_chunks)
    writer = ShardWriter(
        args.out, prefix=session_tag, max_per_shard=args.max_per_shard
    )
    win_id = 0
    since_emit = 0
    skipped = 0
    try:
        done = False
        while not done:
            for seg in plan:
                for _ in range(seg.chunks):
                    rec = await _step(runtime, resampler, held, seg.keys, seg.label)
                    window.append(rec)
                    since_emit += 1
                    ready = len(window) == args.window_chunks
                    if not ready or since_emit < args.emit_stride:
                        continue
                    since_emit = 0
                    if _emit_window(writer, list(window), win_id, session_tag, args):
                        win_id += 1
                        if win_id % 25 == 0:
                            print(
                                f"[capture] wrote {win_id} windows ({skipped} skipped)",
                                flush=True,
                            )
                        if args.limit and win_id >= args.limit:
                            done = True
                            break
                    else:
                        skipped += 1
                if done:
                    break
            if not args.loop:
                break
    finally:
        writer.close()
        await runtime.close()
    print(
        f"[capture] done: {win_id} windows ({skipped} degenerate skipped) "
        f"-> {args.out}",
        flush=True,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capture latent+RGB caption training pairs."
    )
    p.add_argument("--out", required=True, help="Output dir for tar shards + manifest.")
    p.add_argument(
        "--pipeline-config-name",
        default="omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf",
    )
    p.add_argument(
        "--scene-uuid", default=None, help="Scene UUID (default WebRTC scene)."
    )
    p.add_argument("--scene-variant", default=None, help="default/rain/snow.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--window-chunks", type=int, default=5, help="K: chunks per window.")
    p.add_argument(
        "--emit-stride",
        type=int,
        default=4,
        help="Emit a window every N chunks (mirrors the ~1 Hz inference cadence).",
    )
    p.add_argument(
        "--turn-burst",
        type=int,
        default=3,
        help="Chunks per turn arc in the road-keeping weave (default plan only).",
    )
    p.add_argument(
        "--straight-run",
        type=int,
        default=4,
        help="Chunks of straight driving between maneuvers (default plan only).",
    )
    p.add_argument(
        "--min-detail",
        type=float,
        default=0.0,
        help="Drop windows whose mean variance-of-Laplacian is below this "
        "(off-road/degenerate frames). 0 disables the guard; calibrate on a "
        "pilot, then set it for the full run.",
    )
    p.add_argument(
        "--rgb-frames", type=int, default=8, help="RGB frames stored per window."
    )
    p.add_argument(
        "--rgb-max-side", type=int, default=None, help="Downscale RGB long side."
    )
    p.add_argument("--jpeg-quality", type=int, default=90)
    p.add_argument("--warmup-chunks", type=int, default=10)
    p.add_argument(
        "--session-id",
        default=None,
        help="Names this run's shards + keys (default: scene-variant-seed). "
        "Give a distinct value per run so multiple runs share --out safely.",
    )
    p.add_argument("--max-per-shard", type=int, default=1000)
    p.add_argument(
        "--limit", type=int, default=0, help="Stop after N windows (0 = plan once)."
    )
    p.add_argument(
        "--plan", default=None, help="JSON control-track file (else the default)."
    )
    p.add_argument(
        "--loop",
        action="store_true",
        help="Repeat the plan until --limit windows are captured.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    asyncio.run(run_capture(args))


if __name__ == "__main__":
    main()
