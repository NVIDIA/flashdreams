#!/usr/bin/env python3
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

"""Uplift client for the FlashVSR gRPC service.

Usage (from repo root):
    uv run --no-sync python -m flashvsr.grpc.uplift_client --input clip.mp4 --output out.mp4

    # Use unary chunk flow instead of streaming:
    uv run --no-sync python -m flashvsr.grpc.uplift_client --input clip.mp4 --output out.mp4 --unary

    # Connect to a remote server:
    uv run --no-sync python -m flashvsr.grpc.uplift_client --server my-host:50051 --input ...

    # Send JPEG inputs and rely on the server browser viewer for output:
    uv run --no-sync python -m flashvsr.grpc.uplift_client --input clip.mp4 --input_format jpeg --display_only

    # Live-ingest stress mode: loop one video as 8-frame chunks at 30 fps:
    uv run --no-sync python -m flashvsr.grpc.uplift_client --continuous --input clip.mp4
"""

import argparse
import sys
import time
import uuid
from collections import deque
from collections.abc import Iterator

import grpc
import mediapy as media
import numpy as np

from flashdreams.serving.realtime.media import encode_rgb_frame_to_jpeg
from flashvsr.grpc.protos import flashvsr_pb2 as pb2
from flashvsr.grpc.protos import flashvsr_pb2_grpc as pb2_grpc

DEFAULT_SERVER = "localhost:50051"
DEFAULT_MAX_MESSAGE_MB = 512
CONTINUOUS_CHUNK_FRAMES = 8
ANSI_GREEN = "\033[32m"
ANSI_RESET = "\033[0m"

# Supported (first_chunk, chunk_size) pairs.
CHUNK_MODES: dict[int, tuple[int, int]] = {
    16: (13, 16),
    8: (5, 8),
}


def grpc_error_details(exc: grpc.RpcError) -> str:
    details = getattr(exc, "details", None)
    if callable(details):
        try:
            return str(details())
        except Exception:
            pass
    return str(exc)


def video_fps(path: str) -> float:
    try:
        get_video_metadata = getattr(media, "_get_video_metadata")
        return float(get_video_metadata(path).fps)
    except Exception:
        return 30.0


def build_chunks(
    total_frames: int, first_chunk: int, chunk_size: int
) -> list[tuple[int, int]]:
    """Return list of (start, size) pairs for the given first/subsequent chunk sizes."""
    chunks, pos, first = [], 0, True
    while pos < total_frames:
        need = first_chunk if first else chunk_size
        size = min(need, total_frames - pos)
        if size < need:
            print(
                f"Warning: last chunk has {size} frames (need {need}). "
                f"Truncating to {pos} frames."
            )
            break
        chunks.append((pos, size))
        pos += size
        first = False
    return chunks


def video_to_rgb_bytes(frames_np: np.ndarray) -> bytes:
    """uint8 [T,H,W,3] → raw bytes."""
    return frames_np.tobytes()


def encode_jpeg_frames(frames_np: np.ndarray, quality: int) -> list[bytes]:
    """uint8 [T,H,W,3] → one JPEG byte string per frame."""
    try:
        return [
            encode_rgb_frame_to_jpeg(frame, quality=quality, value_range="uint8")
            for frame in frames_np
        ]
    except ModuleNotFoundError as exc:
        if exc.name not in ("PIL", "PIL.Image"):
            raise
        raise RuntimeError(
            "JPEG input requires Pillow in the client environment"
        ) from exc


def build_chunk_request(
    *,
    chunk_idx: int,
    frame_data: np.ndarray,
    scale: int,
    sparse_ratio: float,
    input_format: str,
    jpeg_quality: int,
    display_only: bool,
) -> pb2.UpscaleChunkRequest:
    T, H, W, _ = frame_data.shape
    req = pb2.UpscaleChunkRequest(
        chunk_index=chunk_idx,
        num_frames=T,
        height=H,
        width=W,
        display_only=display_only,
    )
    if input_format == "jpeg":
        req.frame_encoding = pb2.FRAME_ENCODING_JPEG
        req.frames_jpeg.extend(encode_jpeg_frames(frame_data, jpeg_quality))
    else:
        req.frame_encoding = pb2.FRAME_ENCODING_RAW_RGB
        req.frames_rgb = video_to_rgb_bytes(frame_data)
    if chunk_idx == 0:
        req.input_height = H
        req.input_width = W
        req.scale = scale
        req.sparse_ratio = sparse_ratio
    return req


def rgb_bytes_to_video(data: bytes, T: int, H: int, W: int) -> np.ndarray:
    """Raw bytes → uint8 [T,H,W,3]."""
    return np.frombuffer(data, dtype=np.uint8).reshape(T, H, W, 3)


def read_rgb_video(path: str) -> np.ndarray:
    frames = media.read_video(path)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected RGB video [T,H,W,3], got shape {frames.shape}")
    if frames.dtype == np.uint8:
        return np.ascontiguousarray(frames)
    if np.issubdtype(frames.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(frames)) <= 1.0 else 1.0
        frames = np.clip(frames * scale, 0, 255)
    return np.ascontiguousarray(frames.astype(np.uint8))


def circular_chunk(frames: np.ndarray, start: int, size: int) -> np.ndarray:
    total = int(frames.shape[0])
    if total <= 0:
        raise ValueError("input video has no frames")
    indexes = (np.arange(size) + start) % total
    return np.ascontiguousarray(frames[indexes])


# ---------------------------------------------------------------------------
# Streaming flow: upscale_video
# ---------------------------------------------------------------------------


def upsample_stream(
    stub: pb2_grpc.FlashVSRStub,
    video_np: np.ndarray,
    scale: int,
    sparse_ratio: float,
    first_chunk: int,
    chunk_size: int,
    input_format: str,
    jpeg_quality: int,
    display_only: bool,
) -> np.ndarray | None:
    T, H, W, _ = video_np.shape
    chunks = build_chunks(T, first_chunk, chunk_size)
    if not chunks:
        raise ValueError(f"Video too short (need ≥ {first_chunk} frames)")

    usable = sum(s for _, s in chunks)
    if usable < T:
        video_np = video_np[:usable]
        print(f"  Using first {usable} of {T} frames.")

    def request_iter():
        for chunk_idx, (start, size) in enumerate(chunks):
            frame_data = video_np[start : start + size]  # [T, H, W, 3]
            yield build_chunk_request(
                chunk_idx=chunk_idx,
                frame_data=frame_data,
                scale=scale,
                sparse_ratio=sparse_ratio,
                input_format=input_format,
                jpeg_quality=jpeg_quality,
                display_only=display_only,
            )

    output_chunks: dict[int, np.ndarray] = {}
    t_start = time.time()
    for response in stub.upscale_video(request_iter()):
        if response.error:
            raise RuntimeError(
                f"Server error on chunk {response.chunk_index}: {response.error}"
            )
        if response.frames_omitted or not response.frames_rgb:
            if not display_only:
                raise RuntimeError(
                    "Server omitted output frames; viewer mode may be enabled. "
                    "Use --display_only to skip saving an output video."
                )
        elif not display_only:
            arr = rgb_bytes_to_video(
                response.frames_rgb,
                response.num_frames,
                response.height,
                response.width,
            )
            output_chunks[response.chunk_index] = arr
        print(
            f"  Chunk {response.chunk_index + 1}/{len(chunks)}: "
            f"{response.num_frames} frames → {response.height}×{response.width}"
            f"  ({response.elapsed_ms:.0f} ms)"
        )

    total_ms = (time.time() - t_start) * 1000
    print(f"  Total: {total_ms:.0f} ms for {len(chunks)} chunks")

    if display_only:
        return None
    ordered = [output_chunks[i] for i in sorted(output_chunks)]
    return np.concatenate(ordered, axis=0)


# ---------------------------------------------------------------------------
# Unary chunk-by-chunk flow: start_session + upscale_chunk + end_session
# ---------------------------------------------------------------------------


def upsample_unary(
    stub: pb2_grpc.FlashVSRStub,
    video_np: np.ndarray,
    scale: int,
    sparse_ratio: float,
    first_chunk: int,
    chunk_size: int,
    input_format: str,
    jpeg_quality: int,
    display_only: bool,
) -> np.ndarray | None:
    T, H, W, _ = video_np.shape
    chunks = build_chunks(T, first_chunk, chunk_size)
    if not chunks:
        raise ValueError(f"Video too short (need ≥ {first_chunk} frames)")

    usable = sum(s for _, s in chunks)
    if usable < T:
        video_np = video_np[:usable]
        print(f"  Using first {usable} of {T} frames.")

    session_id = str(uuid.uuid4())
    resp = stub.start_session(
        pb2.StartSessionRequest(
            session_id=session_id,
            input_height=H,
            input_width=W,
            scale=scale,
            sparse_ratio=sparse_ratio,
        )
    )
    if not resp.success:
        raise RuntimeError(f"start_session failed: {resp.error}")
    print(f"  Session: {resp.session_id}")

    output_chunks = []
    t_start = time.time()
    try:
        for chunk_idx, (start, size) in enumerate(chunks):
            frame_data = video_np[start : start + size]
            req = build_chunk_request(
                chunk_idx=chunk_idx,
                frame_data=frame_data,
                scale=scale,
                sparse_ratio=sparse_ratio,
                input_format=input_format,
                jpeg_quality=jpeg_quality,
                display_only=display_only,
            )
            req.session_id = session_id
            response = stub.upscale_chunk(req)
            if response.error:
                raise RuntimeError(
                    f"Server error on chunk {chunk_idx}: {response.error}"
                )
            if response.frames_omitted or not response.frames_rgb:
                if not display_only:
                    raise RuntimeError(
                        "Server omitted output frames; viewer mode may be enabled. "
                        "Use --display_only to skip saving an output video."
                    )
            elif not display_only:
                arr = rgb_bytes_to_video(
                    response.frames_rgb,
                    response.num_frames,
                    response.height,
                    response.width,
                )
                output_chunks.append(arr)
            print(
                f"  Chunk {chunk_idx + 1}/{len(chunks)}: "
                f"{response.num_frames} frames → {response.height}×{response.width}"
                f"  ({response.elapsed_ms:.0f} ms)"
            )
    finally:
        stub.end_session(pb2.EndSessionRequest(session_id=session_id))

    total_ms = (time.time() - t_start) * 1000
    print(f"  Total: {total_ms:.0f} ms for {len(chunks)} chunks")
    if display_only:
        return None
    return np.concatenate(output_chunks, axis=0)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_file_client(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="FlashVSR gRPC test client")
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help=f"host:port (default: {DEFAULT_SERVER})",
    )
    parser.add_argument("--input", required=True, help="Input video path (.mp4)")
    parser.add_argument(
        "--output",
        default=None,
        help="Output video path (.mp4). Required unless --display_only is set.",
    )
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4])
    parser.add_argument(
        "--sparse_ratio",
        type=float,
        default=0.0,
        help=(
            "Block-sparse attention ratio. Use 0 to use the server default; "
            "1.5 is faster, 2.0 is more stable."
        ),
    )
    parser.add_argument("--max_message_mb", type=int, default=DEFAULT_MAX_MESSAGE_MB)
    parser.add_argument(
        "--unary",
        action="store_true",
        help="Use unary upscale_chunk flow instead of streaming",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=16,
        choices=[8, 16],
        help="Chunk size: 16 → first=13/subsequent=16, 8 → first=5/subsequent=8 (default: 16)",
    )
    parser.add_argument(
        "--input_format",
        choices=["raw", "jpeg"],
        default="raw",
        help="Client→server frame payload format (default: raw).",
    )
    parser.add_argument(
        "--input_jpeg_quality",
        type=int,
        default=90,
        help="JPEG quality when --input_format=jpeg (default: 90).",
    )
    parser.add_argument(
        "--display_only",
        action="store_true",
        help=(
            "Ask the server to omit output frames from gRPC responses. Use with "
            "server --viewer_port."
        ),
    )
    parser.add_argument("--fps", type=float, default=None)
    args = parser.parse_args(argv)
    if not args.display_only and not args.output:
        parser.error("--output is required unless --display_only is set")
    if not 1 <= args.input_jpeg_quality <= 100:
        parser.error("--input_jpeg_quality must be between 1 and 100")

    max_bytes = args.max_message_mb * 1024 * 1024
    channel_options = [
        ("grpc.max_send_message_length", max_bytes),
        ("grpc.max_receive_message_length", max_bytes),
    ]
    channel = grpc.insecure_channel(args.server, options=channel_options)
    stub = pb2_grpc.FlashVSRStub(channel)

    # Health check
    print(f"Checking server at {args.server} ...")
    try:
        status = stub.get_status(pb2.StatusRequest(), timeout=10)
    except grpc.RpcError as exc:
        print(f"Cannot reach server: {grpc_error_details(exc)}", file=sys.stderr)
        sys.exit(1)
    print(f"  ready={status.ready}  device={status.device}  model={status.model_name}")
    if not status.ready:
        print("Server not ready.", file=sys.stderr)
        sys.exit(1)

    # Read input video
    print(f"\nReading {args.input} ...")
    video_np = read_rgb_video(args.input)  # [T, H, W, 3] uint8
    T, H, W, _ = video_np.shape
    print(f"  {T} frames  {H}×{W}")

    fps = args.fps
    if fps is None:
        fps = video_fps(args.input)
    print(f"  fps: {fps}")

    first_chunk, chunk_size = CHUNK_MODES[args.chunk_size]
    mode = "unary" if args.unary else "streaming"
    print(f"\nUpscaling ({mode}, chunks {first_chunk}/{chunk_size}) via gRPC ...")
    if args.unary:
        result = upsample_unary(
            stub,
            video_np,
            args.scale,
            args.sparse_ratio,
            first_chunk,
            chunk_size,
            args.input_format,
            args.input_jpeg_quality,
            args.display_only,
        )
    else:
        result = upsample_stream(
            stub,
            video_np,
            args.scale,
            args.sparse_ratio,
            first_chunk,
            chunk_size,
            args.input_format,
            args.input_jpeg_quality,
            args.display_only,
        )

    if args.display_only:
        print("Done. Output frames were published by the server viewer.")
        return

    assert result is not None
    out_path = args.output
    assert out_path is not None
    if not out_path.endswith(".mp4"):
        out_path += ".mp4"
    print(
        f"\nSaving {result.shape[0]} frames → {out_path}  ({result.shape[1]}×{result.shape[2]}) @ {fps} fps"
    )
    media.write_video(out_path, result, fps=fps)
    print("Done.")


def run_continuous_client(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Loop one video to FlashVSR as 8-frame live-ingest chunks"
    )
    parser.add_argument(
        "--server",
        default=DEFAULT_SERVER,
        help=f"host:port (default: {DEFAULT_SERVER})",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input video to loop continuously.",
    )
    parser.add_argument("--scale", type=int, default=2, choices=[2, 4])
    parser.add_argument(
        "--sparse_ratio",
        type=float,
        default=0.0,
        help=(
            "Block-sparse attention ratio. Use 0 to use the server default; "
            "1.5 is faster, 2.0 is more stable."
        ),
    )
    parser.add_argument("--max_message_mb", type=int, default=DEFAULT_MAX_MESSAGE_MB)
    parser.add_argument(
        "--target_fps",
        type=float,
        default=30.0,
        help=(
            "Ingress rate to simulate. Default 30 fps models production; "
            "local FlashVSR may only consume around 7 fps."
        ),
    )
    parser.add_argument(
        "--no_pace",
        action="store_true",
        help="Send as fast as gRPC backpressure allows instead of pacing.",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=0,
        help="Stop after this many 8-frame chunks. Default 0 means endless.",
    )
    parser.add_argument(
        "--input_format",
        choices=["raw", "jpeg"],
        default="jpeg",
        help="Client→server frame payload format (default: jpeg).",
    )
    parser.add_argument(
        "--input_jpeg_quality",
        type=int,
        default=90,
        help="JPEG quality when --input_format=jpeg (default: 90).",
    )
    parser.add_argument(
        "--return_frames",
        action="store_true",
        help=(
            "Ask the server to return raw upsampled frames over gRPC. By default "
            "responses are metadata-only and frames go to the server viewer."
        ),
    )
    parser.add_argument(
        "--report_every",
        type=int,
        default=10,
        help="Print progress every N sent/received chunks (default: 10).",
    )
    parser.add_argument(
        "--ingress_window_chunks",
        type=int,
        default=16,
        help=(
            "Number of recent sent chunks used for observed ingress FPS (default: 16)."
        ),
    )
    args = parser.parse_args(argv)

    if args.target_fps <= 0:
        parser.error("--target_fps must be positive")
    if args.max_chunks < 0:
        parser.error("--max_chunks must be non-negative")
    if args.report_every <= 0:
        parser.error("--report_every must be positive")
    if args.ingress_window_chunks < 2:
        parser.error("--ingress_window_chunks must be at least 2")
    if not 1 <= args.input_jpeg_quality <= 100:
        parser.error("--input_jpeg_quality must be between 1 and 100")

    frames = read_rgb_video(args.input)
    total_frames, height, width, _ = frames.shape
    source_fps = video_fps(args.input)
    print(
        f"Loaded {args.input}: {total_frames} frames, {width}x{height}, "
        f"source_fps={source_fps:.2f}"
    )
    print(
        f"Streaming {CONTINUOUS_CHUNK_FRAMES}-frame chunks to {args.server}; "
        f"target_ingress={args.target_fps:.2f} fps; "
        f"mode={'return_frames' if args.return_frames else 'display_only'}"
    )

    max_bytes = args.max_message_mb * 1024 * 1024
    channel = grpc.insecure_channel(
        args.server,
        options=[
            ("grpc.max_send_message_length", max_bytes),
            ("grpc.max_receive_message_length", max_bytes),
        ],
    )
    stub = pb2_grpc.FlashVSRStub(channel)

    try:
        status = stub.get_status(pb2.StatusRequest(), timeout=10)
    except grpc.RpcError as exc:
        print(f"Cannot reach server: {grpc_error_details(exc)}", file=sys.stderr)
        sys.exit(1)
    if not status.ready:
        print("Server is not ready.", file=sys.stderr)
        sys.exit(1)
    print(f"Server ready: device={status.device} model={status.model_name}")

    session_id = str(uuid.uuid4())
    state = {"sent": 0, "received": 0}
    send_times: deque[float] = deque(maxlen=args.ingress_window_chunks)
    receive_times: deque[float] = deque(maxlen=args.ingress_window_chunks)
    frame_interval = CONTINUOUS_CHUNK_FRAMES / args.target_fps

    def requests() -> Iterator[pb2.UpscaleChunkRequest]:
        next_send_at = time.perf_counter()
        source_pos = 0
        chunk_idx = 0
        while args.max_chunks == 0 or chunk_idx < args.max_chunks:
            if not args.no_pace:
                now = time.perf_counter()
                if now < next_send_at:
                    time.sleep(next_send_at - now)
                else:
                    next_send_at = now
                next_send_at += frame_interval

            chunk = circular_chunk(frames, source_pos, CONTINUOUS_CHUNK_FRAMES)
            req = build_chunk_request(
                chunk_idx=chunk_idx,
                frame_data=chunk,
                scale=args.scale,
                sparse_ratio=args.sparse_ratio,
                input_format=args.input_format,
                jpeg_quality=args.input_jpeg_quality,
                display_only=not args.return_frames,
            )
            req.session_id = session_id

            send_times.append(time.perf_counter())
            state["sent"] += 1
            source_pos = (source_pos + CONTINUOUS_CHUNK_FRAMES) % total_frames
            chunk_idx += 1
            if state["sent"] % args.report_every == 0:
                elapsed = max(send_times[-1] - send_times[0], 1e-6)
                ingress_fps = (len(send_times) - 1) * CONTINUOUS_CHUNK_FRAMES / elapsed
                print(
                    f"sent_chunks={state['sent']} "
                    f"sent_frames={state['sent'] * CONTINUOUS_CHUNK_FRAMES} "
                    f"{ANSI_GREEN}observed_ingress={ingress_fps:.2f} fps"
                    f"{ANSI_RESET} window_chunks={len(send_times)}"
                )
            yield req

    try:
        for response in stub.upscale_video(requests()):
            if response.error:
                raise RuntimeError(
                    f"server error on chunk {response.chunk_index}: {response.error}"
                )
            state["received"] += 1
            if args.return_frames and not response.frames_rgb:
                raise RuntimeError(
                    f"chunk {response.chunk_index} returned no frames; "
                    "server may be in viewer-only mode"
                )
            receive_times.append(time.perf_counter())
            if state["received"] % args.report_every == 0:
                elapsed = max(receive_times[-1] - receive_times[0], 1e-6)
                receive_fps = (
                    (len(receive_times) - 1) * CONTINUOUS_CHUNK_FRAMES / elapsed
                )
                print(
                    f"received_chunks={state['received']} "
                    f"last_chunk={response.chunk_index} "
                    f"last_elapsed_ms={response.elapsed_ms:.0f} "
                    f"frames_omitted={response.frames_omitted} "
                    f"observed_receive={receive_fps:.2f} fps "
                    f"window_chunks={len(receive_times)}"
                )
    except KeyboardInterrupt:
        print("\nInterrupted; closing stream.")
    finally:
        channel.close()

    print(
        f"Done. sent_chunks={state['sent']} received_chunks={state['received']} "
        f"session={session_id}"
    )


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--continuous" in args:
        args.remove("--continuous")
        run_continuous_client(args)
    else:
        run_file_client(args)


if __name__ == "__main__":
    main()
