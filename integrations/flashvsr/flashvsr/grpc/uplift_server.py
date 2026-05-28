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

"""gRPC server for FlashVSR video super-resolution.

Drives :class:`flashvsr.pipeline.FlashVSRPipeline` once per
``(input_H, input_W, scale)`` combination; per-session streaming state
lives on :class:`FlashVSRPipelineCache` instances obtained from
``upsampler.initialize_cache()``.

``upscale_video`` pipelines receive/decode → buffer/coalesce → GPU → send/viewer:
a reader thread decodes and enqueues incoming chunks (FIFO), a worker optionally frame-coalesces
8-frame ingest into FlashVSR's 13-frame cold start plus steady 16-frame calls,
and bounded queues provide backpressure plus paced browser playback.

Usage (from repo root):

    uv run --no-sync python -m flashvsr.grpc.uplift_server --viewer_port 8080
"""

import argparse
import importlib
import os
import queue
import signal
import sys
import threading
import time
import uuid
from collections.abc import Callable
from concurrent import futures
from dataclasses import dataclass, field
from typing import Literal

import grpc
import numpy as np
import torch
from loguru import logger

from flashvsr.config import build_flashvsr_v1_1
from flashvsr.grpc.protos import flashvsr_pb2 as pb2
from flashvsr.grpc.protos import flashvsr_pb2_grpc as pb2_grpc
from flashvsr.grpc.streaming_view import (
    DEFAULT_VIEWER_CHUNK_QUEUE_DEPTH,
    DEFAULT_VIEWER_FRAME_STRIDE,
    DEFAULT_VIEWER_JPEG_BACKEND,
    DEFAULT_VIEWER_JPEG_QUALITY,
    DEFAULT_VIEWER_MAX_FPS,
    StreamingViewer,
    decode_jpeg_rgb,
    encode_jpeg_cuda_tensor,
)
from flashvsr.pipeline import FlashVSRPipeline, FlashVSRPipelineCache

DEFAULT_PORT = 50051
DEFAULT_MAX_MESSAGE_MB = 512
# upscale_video: bounded FIFO queues so receive / GPU / send can overlap (depth per stream).
DEFAULT_STREAM_INBOUND_QUEUE_DEPTH = 16
DEFAULT_STREAM_CHUNK_QUEUE_DEPTH = 16
DEFAULT_VIEWER_METADATA_QUEUE_DEPTH = 64

_INBOUND_END = object()
_OUTBOUND_END = object()
AttentionMode = Literal["sparse", "full"]
RequestedAttentionMode = Literal["sparse", "full", "auto"]
Scale = Literal[2, 4]


def _default_model_cache_path() -> str:
    return os.path.expanduser(
        os.getenv("FLASHDREAMS_CACHE_DIR", "~/.cache/flashdreams")
    )


def _synchronize_device(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _resolve_scale(scale: int) -> Scale:
    if scale == 2:
        return 2
    if scale == 4:
        return 4
    raise ValueError(f"FlashVSR scale must be 2 or 4, got {scale}")


def _block_sparse_attn_available() -> bool:
    try:
        importlib.import_module("block_sparse_attn")
    except ModuleNotFoundError:
        return False
    return True


def _resolve_attention_mode(attention_mode: RequestedAttentionMode) -> AttentionMode:
    if attention_mode == "full":
        return "full"
    if _block_sparse_attn_available():
        return "sparse"
    if attention_mode == "auto":
        logger.warning(
            "block_sparse_attn is unavailable; falling back to attention_mode=full"
        )
        return "full"
    raise RuntimeError(
        "FlashVSR attention_mode='sparse' requires the block_sparse_attn CUDA "
        "extension. Install block-sparse-attn or pass --attention_mode full/auto."
    )


@dataclass
class _UpscaleVideoReaderError:
    """Reader thread failed before or while filling the inbound queue."""

    exc: BaseException


@dataclass
class _Session:
    key: tuple  # (input_H, input_W, scale, sparse_ratio, attention_mode)
    cache: FlashVSRPipelineCache
    created_at: float = field(default_factory=time.time)


@dataclass
class _RunChunkResult:
    frames_rgb: bytes
    frames_out: np.ndarray | None
    num_frames: int
    height: int
    width: int
    elapsed_ms: float


@dataclass
class _BufferedStreamRequest:
    request: pb2.UpscaleChunkRequest
    output_parts: list[np.ndarray] = field(default_factory=list)
    output_count: int = 0
    elapsed_ms: float = 0.0
    height: int = 0
    width: int = 0


@dataclass
class _DecodedStreamRequest:
    request: pb2.UpscaleChunkRequest
    frames: np.ndarray
    decode_ms: float


class FlashVSR(pb2_grpc.FlashVSRServicer):
    def __init__(
        self,
        model_path: str,
        model_name: str,
        default_H: int,
        default_W: int,
        default_scale: int,
        default_sparse_ratio: float,
        attention_mode: AttentionMode,
        compile_network: bool,
        use_cuda_graph: bool,
        dtype: torch.dtype,
        device: str,
        stream_inbound_queue_depth: int = DEFAULT_STREAM_INBOUND_QUEUE_DEPTH,
        stream_chunk_queue_depth: int = DEFAULT_STREAM_CHUNK_QUEUE_DEPTH,
        combine_8_frame_chunks: bool = True,
        viewer: StreamingViewer | None = None,
        omit_grpc_frames_when_viewing: bool = False,
    ):
        self._model_path = model_path
        self._model_name = model_name
        self._default_H = default_H
        self._default_W = default_W
        self._default_scale = default_scale
        self._default_sparse_ratio = default_sparse_ratio
        self._attention_mode = attention_mode
        self._compile_network = bool(compile_network)
        self._use_cuda_graph = bool(use_cuda_graph)
        self._dtype = dtype
        self._device = device
        self._stream_inbound_queue_depth = max(1, int(stream_inbound_queue_depth))
        self._combine_8_frame_chunks = bool(combine_8_frame_chunks)
        self._viewer = viewer
        self._omit_grpc_frames_when_viewing = bool(
            omit_grpc_frames_when_viewing and viewer is not None
        )
        self._stream_chunk_queue_depth = max(1, int(stream_chunk_queue_depth))
        if self._omit_grpc_frames_when_viewing:
            self._stream_chunk_queue_depth = max(
                self._stream_chunk_queue_depth,
                DEFAULT_VIEWER_METADATA_QUEUE_DEPTH,
            )
        self._warned_cuda_jpeg_fallback = False

        # One upsampler per shape/quality key; weights shared, caches per session.
        self._upsampler_pool: dict[tuple, FlashVSRPipeline] = {}
        self._pool_lock = threading.Lock()

        self._sessions: dict[str, _Session] = {}
        self._sessions_lock = threading.Lock()

        # Serialize GPU ops so multiple concurrent gRPC calls don't fight.
        self._gpu_lock = threading.Lock()

        logger.info(
            "Warming up model at ({}×{}) scale={} attention_mode={} ...",
            default_H,
            default_W,
            default_scale,
            attention_mode,
        )
        self._get_upsampler(default_H, default_W, default_scale, default_sparse_ratio)
        self._warm_up_viewer_jpeg()
        logger.info("Model ready.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _warm_up_viewer_jpeg(self) -> None:
        if self._viewer is None or self._viewer.jpeg_backend == "pillow":
            return
        if not str(self._device).startswith("cuda"):
            return
        try:
            dummy = torch.zeros((1, 3, 1, 16, 16), device=self._device)
            encode_jpeg_cuda_tensor(
                dummy,
                quality=self._viewer.jpeg_quality,
                frame_stride=1,
            )
            _synchronize_device(self._device)
            logger.info("Viewer CUDA JPEG encoder ready.")
        except Exception:
            if self._viewer.jpeg_backend == "cuda":
                raise
            if not self._warned_cuda_jpeg_fallback:
                logger.opt(exception=True).warning(
                    "CUDA viewer JPEG encode unavailable; falling back to "
                    "CPU/Pillow encoding"
                )
                self._warned_cuda_jpeg_fallback = True

    def _get_upsampler(
        self, H: int, W: int, scale: int, sparse_ratio: float
    ) -> FlashVSRPipeline:
        resolved_scale = _resolve_scale(scale)
        key = (H, W, resolved_scale, float(sparse_ratio), self._attention_mode)
        with self._pool_lock:
            if key not in self._upsampler_pool:
                logger.info(
                    "Loading upsampler for ({}×{}) scale={} sparse_ratio={:.3g} "
                    "attention_mode={} compile={} cuda_graph={} ...",
                    H,
                    W,
                    resolved_scale,
                    sparse_ratio,
                    self._attention_mode,
                    self._compile_network,
                    self._use_cuda_graph,
                )
                # ``build_flashvsr_v1_1`` follows the integration config:
                # checkpoints come from HuggingFace/GitHub URLs and are cached
                # under the normal FlashDreams/HuggingFace cache roots. The
                # ``--model_path`` flag remains a compatibility/status hint
                # for clients migrated from the standalone upsampler tree.
                config = build_flashvsr_v1_1(
                    input_H=H,
                    input_W=W,
                    scale=resolved_scale,
                    sparse_ratio=sparse_ratio,
                    compile_network=self._compile_network,
                    use_cuda_graph=self._use_cuda_graph,
                    attention_mode=self._attention_mode,
                    dtype=self._dtype,
                )
                self._upsampler_pool[key] = (
                    config.setup().to(device=self._device).eval()
                )
            return self._upsampler_pool[key]

    def _frames_rgb_to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(arr.astype(np.float32)) / 127.5 - 1.0  # [T,H,W,3]
        return (
            t.permute(3, 0, 1, 2)
            .unsqueeze(0)
            .to(device=self._device, dtype=self._dtype)
        )  # [1,3,T,H,W]

    def _request_to_frames_rgb(self, request: pb2.UpscaleChunkRequest) -> np.ndarray:
        T, H, W = request.num_frames, request.height, request.width
        if request.frame_encoding == pb2.FRAME_ENCODING_JPEG:
            if len(request.frames_jpeg) != T:
                raise ValueError(
                    f"frame_encoding=JPEG requires {T} frames_jpeg payloads; "
                    f"got {len(request.frames_jpeg)}"
                )
            frames = [decode_jpeg_rgb(frame) for frame in request.frames_jpeg]
            arr = np.stack(frames, axis=0)
            if H and W and arr.shape[1:3] != (H, W):
                raise ValueError(
                    f"JPEG payload dimensions {arr.shape[1]}×{arr.shape[2]} "
                    f"do not match request height/width {H}×{W}"
                )
            return arr

        expected = T * H * W * 3
        if len(request.frames_rgb) != expected:
            raise ValueError(
                f"RAW_RGB payload has {len(request.frames_rgb)} bytes; "
                f"expected {expected} for {T}×{H}×{W}×3"
            )
        arr = np.frombuffer(request.frames_rgb, dtype=np.uint8).reshape(T, H, W, 3)
        return np.ascontiguousarray(arr)

    def _tensor_to_frames_array(
        self, out: torch.Tensor, *, frame_stride: int = 1
    ) -> np.ndarray:
        """out: [1, 3, T, H, W] in [-1,1] → uint8 [T, H, W, 3]"""
        if frame_stride > 1:
            out = out[:, :, ::frame_stride]
        # np.ascontiguousarray avoids a second copy from transpose.
        arr = ((out.float() + 1.0) * 127.5).clamp(0, 255).byte().cpu().numpy()
        arr = np.ascontiguousarray(arr[0].transpose(1, 2, 3, 0))  # [T, H, W, 3]
        return arr

    def _should_omit_response_frames(self, request: pb2.UpscaleChunkRequest) -> bool:
        return bool(request.display_only or self._omit_grpc_frames_when_viewing)

    def _run_frames(
        self,
        upsampler: FlashVSRPipeline,
        cache: FlashVSRPipelineCache,
        frames: np.ndarray,
        chunk_index: int,
        *,
        return_frames: bool,
        pre_decode_ms: float = 0.0,
        on_after_infer: Callable[[int], None] | None = None,
    ) -> _RunChunkResult:
        """Run one frame array through the model."""
        t0 = time.perf_counter()
        video_t = self._frames_rgb_to_tensor(frames)
        t1 = time.perf_counter()

        with self._gpu_lock:
            out = upsampler.generate(
                autoregressive_index=chunk_index,
                cache=cache,
                input=video_t,
            )
            upsampler.finalize(autoregressive_index=chunk_index, cache=cache)
        _synchronize_device(self._device)
        t2 = time.perf_counter()

        out_T, out_H, out_W = int(out.shape[2]), int(out.shape[3]), int(out.shape[4])
        frames_out: np.ndarray | None = None
        viewer_frames: np.ndarray | None = None
        viewer_jpegs: list[bytes] | None = None
        if return_frames:
            frames_out = self._tensor_to_frames_array(out)
            out_T, out_H, out_W, _ = frames_out.shape
            if self._viewer is not None:
                stride = self._viewer.frame_stride
                viewer_frames = frames_out[::stride] if stride > 1 else frames_out
        elif self._viewer is not None:
            backend = self._viewer.jpeg_backend
            if backend in ("auto", "cuda") and out.is_cuda:
                try:
                    viewer_jpegs = encode_jpeg_cuda_tensor(
                        out,
                        quality=self._viewer.jpeg_quality,
                        frame_stride=self._viewer.frame_stride,
                    )
                except Exception:
                    if backend == "cuda":
                        raise
                    if not self._warned_cuda_jpeg_fallback:
                        logger.opt(exception=True).warning(
                            "CUDA viewer JPEG encode unavailable; falling back "
                            "to CPU/Pillow encoding"
                        )
                        self._warned_cuda_jpeg_fallback = True
            if viewer_jpegs is None:
                viewer_frames = self._tensor_to_frames_array(
                    out,
                    frame_stride=self._viewer.frame_stride,
                )
        infer_ms = (t2 - t1) * 1000.0
        rgb = frames_out.tobytes() if return_frames and frames_out is not None else b""
        t3 = time.perf_counter()

        decode_ms = pre_decode_ms + (t1 - t0) * 1000.0
        encode_ms = (t3 - t2) * 1000.0
        critical_ms = (t3 - t0) * 1000.0
        viewer_elapsed_ms = critical_ms
        total_ms = pre_decode_ms + critical_ms
        if self._viewer is not None:
            stride = self._viewer.frame_stride
            original_frames = frames[::stride] if stride > 1 else frames
            self._viewer.enqueue_original_chunk(
                original_frames,
                elapsed_ms=viewer_elapsed_ms,
            )
            if viewer_jpegs is not None:
                self._viewer.enqueue_upscaled_jpeg_chunk(
                    viewer_jpegs,
                    elapsed_ms=viewer_elapsed_ms,
                )
            elif viewer_frames is not None:
                self._viewer.enqueue_upscaled_chunk(
                    viewer_frames,
                    elapsed_ms=viewer_elapsed_ms,
                )
        logger.info(
            "  chunk {} timing:  decode_in={:.0f} ms  infer={:.0f} ms  "
            "encode_out={:.0f} ms  total={:.0f} ms  critical={:.0f} ms  "
            "(out {}×{})",
            chunk_index,
            decode_ms,
            infer_ms,
            encode_ms,
            total_ms,
            critical_ms,
            out_W,
            out_H,
        )
        if on_after_infer is not None:
            on_after_infer(chunk_index)
        return _RunChunkResult(
            frames_rgb=rgb,
            frames_out=frames_out,
            num_frames=out_T,
            height=out_H,
            width=out_W,
            elapsed_ms=infer_ms,
        )

    def _run_chunk(
        self,
        upsampler: FlashVSRPipeline,
        cache: FlashVSRPipelineCache,
        request: pb2.UpscaleChunkRequest,
        chunk_index: int,
        *,
        return_frames: bool,
        on_after_infer: Callable[[int], None] | None = None,
    ) -> _RunChunkResult:
        """Run one request through the model."""
        t0 = time.perf_counter()
        frames = self._request_to_frames_rgb(request)
        decode_ms = (time.perf_counter() - t0) * 1000.0
        return self._run_frames(
            upsampler,
            cache,
            frames,
            chunk_index,
            return_frames=return_frames,
            pre_decode_ms=decode_ms,
            on_after_infer=on_after_infer,
        )

    # ------------------------------------------------------------------
    # RPC implementations
    # ------------------------------------------------------------------

    def get_status(self, request, context):
        with self._sessions_lock:
            active = list(self._sessions.keys())
        return pb2.StatusResponse(
            ready=True,
            device=str(self._device),
            model_name=f"{self._model_name}/{self._attention_mode}",
            active_sessions=active,
        )

    def start_session(self, request, context):
        session_id = request.session_id or str(uuid.uuid4())
        H = request.input_height or self._default_H
        W = request.input_width or self._default_W
        scale = request.scale or self._default_scale
        sparse_ratio = request.sparse_ratio or self._default_sparse_ratio

        with self._sessions_lock:
            if session_id in self._sessions:
                return pb2.StartSessionResponse(
                    session_id=session_id,
                    success=False,
                    error=f"session '{session_id}' already exists; call end_session first",
                )

        try:
            upsampler = self._get_upsampler(H, W, scale, sparse_ratio)
            cache = upsampler.initialize_cache()
            with self._sessions_lock:
                self._sessions[session_id] = _Session(
                    key=(H, W, scale, sparse_ratio, self._attention_mode),
                    cache=cache,
                )
            logger.info(
                "start_session {} ({}×{} scale={} sparse_ratio={:.3g} attention_mode={})",
                session_id,
                H,
                W,
                scale,
                sparse_ratio,
                self._attention_mode,
            )
            return pb2.StartSessionResponse(session_id=session_id, success=True)
        except Exception as exc:
            logger.exception("start_session failed")
            return pb2.StartSessionResponse(success=False, error=str(exc))

    def end_session(self, request, context):
        with self._sessions_lock:
            removed = self._sessions.pop(request.session_id, None)
        if removed:
            logger.info("end_session {}", request.session_id)
            # FlashVSRPipelineCache holds GPU buffers via its sub-caches;
            # dropping the reference is enough — the next initialize_cache
            # call rebuilds them. No explicit free() needed.
        return pb2.EndSessionResponse(success=True)

    def upscale_chunk(self, request, context):
        with self._sessions_lock:
            session = self._sessions.get(request.session_id)
        if session is None:
            msg = f"session '{request.session_id}' not found; call start_session first"
            context.set_code(grpc.StatusCode.NOT_FOUND)
            context.set_details(msg)
            return pb2.UpscaleChunkResponse(error=msg)

        H_in, W_in, scale, sparse_ratio, _attention_mode = session.key
        upsampler = self._get_upsampler(H_in, W_in, scale, sparse_ratio)
        omit_frames = self._should_omit_response_frames(request)

        try:
            result = self._run_chunk(
                upsampler,
                session.cache,
                request,
                request.chunk_index,
                return_frames=not omit_frames,
            )
            logger.info(
                "upscale_chunk {} chunk={} T={} -> {:.0f} ms",
                request.session_id,
                request.chunk_index,
                request.num_frames,
                result.elapsed_ms,
            )
            return pb2.UpscaleChunkResponse(
                session_id=request.session_id,
                frames_rgb=result.frames_rgb,
                num_frames=result.num_frames,
                height=result.height,
                width=result.width,
                chunk_index=request.chunk_index,
                elapsed_ms=result.elapsed_ms,
                frames_omitted=omit_frames,
            )
        except Exception as exc:
            logger.exception("upscale_chunk failed")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return pb2.UpscaleChunkResponse(error=str(exc))

    def upscale_video(self, request_iterator, context):
        """Pipelined streaming: Rx thread → inbound FIFO → GPU worker → outbound FIFO → yields.

        Receive and gRPC send can overlap with inference. Chunk order is preserved end-to-end
        (single GPU worker per stream). Inbound and outbound queues are bounded so a
        continuous producer gets backpressure instead of unbounded server memory growth.
        When enabled, streams that start with 8-frame requests are frame-coalesced
        into FlashVSR's 13-frame cold start plus steady 16-frame calls, while
        responses are emitted for the original request chunk indexes.
        """
        in_depth = self._stream_inbound_queue_depth
        out_depth = self._stream_chunk_queue_depth
        inbound: queue.Queue = queue.Queue(maxsize=in_depth)
        outbound: queue.Queue = queue.Queue(maxsize=out_depth)
        abort = threading.Event()
        context.add_callback(abort.set)
        cache_holder: list[FlashVSRPipelineCache | None] = [None]
        session_id_box: list[str] = [str(uuid.uuid4())]
        count_lock = threading.Lock()
        counts = {"rx": 0, "gpu_done": 0, "sent": 0}

        def put_inbound(item: object) -> bool:
            while not abort.is_set():
                try:
                    inbound.put(item, timeout=0.25)
                    return True
                except queue.Full:
                    continue
            return False

        def put_inbound_end() -> None:
            while True:
                try:
                    inbound.put(_INBOUND_END, timeout=0.25)
                    return
                except queue.Full:
                    if abort.is_set():
                        return

        def put_outbound(item: object) -> bool:
            while not abort.is_set() and context.is_active():
                try:
                    outbound.put(item, timeout=0.25)
                    return True
                except queue.Full:
                    continue
            return False

        def reader_fn() -> None:
            reader_err: _UpscaleVideoReaderError | None = None
            try:
                for req in request_iterator:
                    if abort.is_set():
                        break
                    decode_t0 = time.perf_counter()
                    frames = self._request_to_frames_rgb(req)
                    decoded = _DecodedStreamRequest(
                        request=req,
                        frames=frames,
                        decode_ms=(time.perf_counter() - decode_t0) * 1000.0,
                    )
                    if not put_inbound(decoded):
                        break
                    with count_lock:
                        counts["rx"] += 1
            except grpc.RpcError as exc:
                if abort.is_set() or not context.is_active():
                    logger.info("upscale_video: client stream closed")
                else:
                    logger.warning(
                        "upscale_video: receive loop ended with gRPC error: {}",
                        exc,
                    )
                    reader_err = _UpscaleVideoReaderError(exc)
            except BaseException as exc:
                logger.exception("upscale_video: receive loop failed")
                reader_err = _UpscaleVideoReaderError(exc)
            finally:
                if reader_err is not None:
                    put_inbound(reader_err)
                put_inbound_end()

        def gpu_fn() -> None:
            upsampler_local: FlashVSRPipeline | None = None
            cache_local: FlashVSRPipelineCache | None = None
            sid = session_id_box[0]
            model_chunk_index = 0
            combine_eights_for_stream: bool | None = None
            frame_buffer: list[tuple[_BufferedStreamRequest, np.ndarray]] = []
            request_buffer: list[_BufferedStreamRequest] = []
            coalesced_decode_ms = 0.0

            def ensure_stream(req: pb2.UpscaleChunkRequest) -> None:
                nonlocal upsampler_local, cache_local, sid
                if upsampler_local is not None:
                    return
                H = req.input_height or req.height or self._default_H
                W = req.input_width or req.width or self._default_W
                scale = req.scale or self._default_scale
                sparse_ratio = req.sparse_ratio or self._default_sparse_ratio
                if req.session_id:
                    sid = req.session_id
                    session_id_box[0] = sid
                upsampler_local = self._get_upsampler(H, W, scale, sparse_ratio)
                cache_local = upsampler_local.initialize_cache()
                cache_holder[0] = cache_local
                logger.info(
                    "upscale_video stream {} ({}×{} scale={}) pipelined "
                    "inbound_depth={} outbound_depth={} combine_8_frame_chunks={}",
                    sid,
                    H,
                    W,
                    scale,
                    in_depth,
                    out_depth,
                    self._combine_8_frame_chunks,
                )

            def responses_for_result(
                reqs: list[pb2.UpscaleChunkRequest],
                result: _RunChunkResult,
            ) -> list[pb2.UpscaleChunkResponse]:
                total_in = sum(int(req.num_frames) for req in reqs)
                if result.frames_out is not None and len(reqs) > 1:
                    if result.frames_out.shape[0] != total_in:
                        raise RuntimeError(
                            "combined FlashVSR output frame count "
                            f"{result.frames_out.shape[0]} did not match "
                            f"input frame count {total_in}"
                        )

                responses: list[pb2.UpscaleChunkResponse] = []
                offset = 0
                for req in reqs:
                    omit_frames = self._should_omit_response_frames(req)
                    frames_rgb = b""
                    num_frames = result.num_frames
                    if result.frames_out is not None:
                        if len(reqs) == 1:
                            frames_part = result.frames_out
                        else:
                            end = offset + int(req.num_frames)
                            frames_part = result.frames_out[offset:end]
                            offset = end
                        num_frames = int(frames_part.shape[0])
                        if not omit_frames:
                            frames_rgb = frames_part.tobytes()
                    elif len(reqs) > 1:
                        num_frames = int(req.num_frames)

                    elapsed_ms = result.elapsed_ms
                    if len(reqs) > 1 and total_in > 0:
                        elapsed_ms = result.elapsed_ms * int(req.num_frames) / total_in
                    responses.append(
                        pb2.UpscaleChunkResponse(
                            session_id=sid,
                            frames_rgb=frames_rgb,
                            num_frames=num_frames,
                            height=result.height,
                            width=result.width,
                            chunk_index=req.chunk_index,
                            elapsed_ms=elapsed_ms,
                            frames_omitted=omit_frames,
                        )
                    )
                return responses

            def process_requests(items: list[_DecodedStreamRequest]) -> bool:
                nonlocal model_chunk_index
                assert items
                reqs = [item.request for item in items]
                try:
                    ensure_stream(reqs[0])
                except Exception as exc:
                    logger.exception("upscale_video: model init failed")
                    put_outbound(pb2.UpscaleChunkResponse(error=str(exc)))
                    put_outbound(_OUTBOUND_END)
                    return False

                assert upsampler_local is not None and cache_local is not None

                def _on_after_infer(ci: int) -> None:
                    with count_lock:
                        counts["gpu_done"] += 1
                        cum_rx = counts["rx"]
                        cum_gpu = counts["gpu_done"]
                        cum_sent = counts["sent"]
                    logger.info(
                        "  model chunk {} pipeline: buf_inbound={} buf_outbound={} "
                        "cum_rx_chunks={} cum_model_done={} cum_sent={}",
                        ci,
                        inbound.qsize(),
                        outbound.qsize(),
                        cum_rx,
                        cum_gpu,
                        cum_sent,
                    )

                wall_t0 = time.perf_counter()
                source_indexes = [int(req.chunk_index) for req in reqs]
                source_frames = [int(req.num_frames) for req in reqs]
                try:
                    frames_parts = [item.frames for item in items]
                    decode_ms = sum(item.decode_ms for item in items)
                    frames = (
                        frames_parts[0]
                        if len(frames_parts) == 1
                        else np.concatenate(frames_parts, axis=0)
                    )
                    return_frames = any(
                        not self._should_omit_response_frames(req) for req in reqs
                    )
                    result = self._run_frames(
                        upsampler_local,
                        cache_local,
                        frames,
                        model_chunk_index,
                        return_frames=return_frames,
                        pre_decode_ms=decode_ms,
                        on_after_infer=_on_after_infer,
                    )
                    for resp in responses_for_result(reqs, result):
                        if not put_outbound(resp):
                            return False
                    wall_ms = (time.perf_counter() - wall_t0) * 1000.0
                    with count_lock:
                        cum_rx = counts["rx"]
                        cum_gpu = counts["gpu_done"]
                        cum_sent = counts["sent"]
                    logger.info(
                        "upscale_video {} model_chunk={} source_chunks={} "
                        "source_frames={} model_T={} infer={:.0f} ms wall={:.0f} ms "
                        "(grpc_overhead={:.0f} ms) buf_inbound={} buf_outbound={} "
                        "cum_rx_chunks={} cum_model_done={} cum_sent={}",
                        sid,
                        model_chunk_index,
                        source_indexes,
                        source_frames,
                        int(frames.shape[0]),
                        result.elapsed_ms,
                        wall_ms,
                        wall_ms - result.elapsed_ms,
                        inbound.qsize(),
                        outbound.qsize(),
                        cum_rx,
                        cum_gpu,
                        cum_sent,
                    )
                    model_chunk_index += 1
                    return True
                except Exception as exc:
                    logger.exception(
                        "upscale_video: model chunk {} failed for source chunks {}",
                        model_chunk_index,
                        source_indexes,
                    )
                    put_outbound(pb2.UpscaleChunkResponse(error=str(exc)))
                    put_outbound(_OUTBOUND_END)
                    return False

            def emit_coalesced_responses(*, final: bool) -> bool:
                while request_buffer:
                    br = request_buffer[0]
                    if br.output_count == 0 and final:
                        request_buffer.pop(0)
                        continue
                    if not final and br.output_count < br.request.num_frames:
                        break
                    request_buffer.pop(0)

                    omit_frames = self._should_omit_response_frames(br.request)
                    frames_rgb = b""
                    if not omit_frames and br.output_parts:
                        frames_rgb = np.ascontiguousarray(
                            np.stack(br.output_parts, axis=0)
                        ).tobytes()
                    elif not omit_frames and br.output_count:
                        raise RuntimeError(
                            "coalesced response frames were requested but not retained"
                        )
                    if not put_outbound(
                        pb2.UpscaleChunkResponse(
                            session_id=sid,
                            frames_rgb=frames_rgb,
                            num_frames=br.output_count,
                            height=br.height,
                            width=br.width,
                            chunk_index=br.request.chunk_index,
                            elapsed_ms=br.elapsed_ms,
                            frames_omitted=omit_frames,
                        )
                    ):
                        return False
                return True

            def next_coalesced_target(*, final: bool) -> int:
                available = len(frame_buffer)
                if model_chunk_index == 0:
                    if available >= 13:
                        return 13
                    if final and available >= 5:
                        return 5
                    return 0
                if available >= 16:
                    return 16
                if final:
                    for target in (13, 8, 5):
                        if available >= target:
                            return target
                return 0

            def run_coalesced_model(target_frames: int) -> bool:
                nonlocal model_chunk_index, coalesced_decode_ms
                assert target_frames > 0
                assert upsampler_local is not None and cache_local is not None
                items = frame_buffer[:target_frames]
                del frame_buffer[:target_frames]

                source_indexes: list[int] = []
                source_frames: list[int] = []
                seen_requests: set[int] = set()
                return_frames = False
                for br, _frame in items:
                    ident = id(br)
                    if ident not in seen_requests:
                        seen_requests.add(ident)
                        source_indexes.append(int(br.request.chunk_index))
                        source_frames.append(int(br.request.num_frames))
                        if not self._should_omit_response_frames(br.request):
                            return_frames = True

                frames = np.ascontiguousarray(
                    np.stack([frame for _br, frame in items], axis=0)
                )

                def _on_after_infer(ci: int) -> None:
                    with count_lock:
                        counts["gpu_done"] += 1
                        cum_rx = counts["rx"]
                        cum_gpu = counts["gpu_done"]
                        cum_sent = counts["sent"]
                    logger.info(
                        "  coalesced model chunk {} pipeline: "
                        "buf_frames={} buf_requests={} buf_inbound={} "
                        "buf_outbound={} cum_rx_chunks={} cum_model_done={} "
                        "cum_sent={}",
                        ci,
                        len(frame_buffer),
                        len(request_buffer),
                        inbound.qsize(),
                        outbound.qsize(),
                        cum_rx,
                        cum_gpu,
                        cum_sent,
                    )

                wall_t0 = time.perf_counter()
                try:
                    result = self._run_frames(
                        upsampler_local,
                        cache_local,
                        frames,
                        model_chunk_index,
                        return_frames=return_frames,
                        pre_decode_ms=coalesced_decode_ms,
                        on_after_infer=_on_after_infer,
                    )
                    coalesced_decode_ms = 0.0
                    if result.num_frames != len(items):
                        raise RuntimeError(
                            "coalesced FlashVSR output frame count "
                            f"{result.num_frames} did not match input frame count "
                            f"{len(items)}"
                        )

                    out_frames = result.frames_out
                    per_frame_ms = result.elapsed_ms / max(1, result.num_frames)
                    for frame_idx, (br, _source_frame) in enumerate(items):
                        br.output_count += 1
                        br.elapsed_ms += per_frame_ms
                        br.height = result.height
                        br.width = result.width
                        if (
                            out_frames is not None
                            and not self._should_omit_response_frames(br.request)
                        ):
                            br.output_parts.append(out_frames[frame_idx])

                    wall_ms = (time.perf_counter() - wall_t0) * 1000.0
                    with count_lock:
                        cum_rx = counts["rx"]
                        cum_gpu = counts["gpu_done"]
                        cum_sent = counts["sent"]
                    logger.info(
                        "upscale_video {} coalesced_model_chunk={} "
                        "source_chunks={} source_request_frames={} model_T={} "
                        "infer={:.0f} ms wall={:.0f} ms (grpc_overhead={:.0f} ms) "
                        "buf_frames={} buf_requests={} buf_inbound={} "
                        "buf_outbound={} cum_rx_chunks={} cum_model_done={} "
                        "cum_sent={}",
                        sid,
                        model_chunk_index,
                        source_indexes,
                        source_frames,
                        int(frames.shape[0]),
                        result.elapsed_ms,
                        wall_ms,
                        wall_ms - result.elapsed_ms,
                        len(frame_buffer),
                        len(request_buffer),
                        inbound.qsize(),
                        outbound.qsize(),
                        cum_rx,
                        cum_gpu,
                        cum_sent,
                    )
                    model_chunk_index += 1
                    return True
                except Exception as exc:
                    logger.exception(
                        "upscale_video: coalesced model chunk {} failed for "
                        "source chunks {}",
                        model_chunk_index,
                        source_indexes,
                    )
                    put_outbound(pb2.UpscaleChunkResponse(error=str(exc)))
                    put_outbound(_OUTBOUND_END)
                    return False

            def drain_coalesced(*, final: bool) -> bool:
                while True:
                    target = next_coalesced_target(final=final)
                    if target <= 0:
                        break
                    if not run_coalesced_model(target):
                        return False
                    if not emit_coalesced_responses(final=False):
                        return False

                if final:
                    if frame_buffer:
                        dropped = len(frame_buffer)
                        logger.warning(
                            "upscale_video stream {}: dropping {} trailing "
                            "frame(s) that do not form a supported FlashVSR "
                            "final chunk",
                            sid,
                            dropped,
                        )
                        frame_buffer.clear()
                    if not emit_coalesced_responses(final=True):
                        return False
                return True

            def append_coalesced_request(item: _DecodedStreamRequest) -> bool:
                nonlocal coalesced_decode_ms
                req = item.request
                try:
                    ensure_stream(req)
                    br = _BufferedStreamRequest(request=req)
                    frames = item.frames
                    coalesced_decode_ms += item.decode_ms
                except Exception as exc:
                    logger.exception("upscale_video: coalesced request prepare failed")
                    put_outbound(pb2.UpscaleChunkResponse(error=str(exc)))
                    put_outbound(_OUTBOUND_END)
                    return False

                request_buffer.append(br)
                for frame in frames:
                    frame_buffer.append((br, frame))
                return drain_coalesced(final=False)

            try:
                while True:
                    msg = inbound.get()
                    if msg is _INBOUND_END:
                        if combine_eights_for_stream and not drain_coalesced(
                            final=True
                        ):
                            return
                        put_outbound(_OUTBOUND_END)
                        return
                    if isinstance(msg, _UpscaleVideoReaderError):
                        if combine_eights_for_stream:
                            if not drain_coalesced(final=True):
                                return
                        if not put_outbound(
                            pb2.UpscaleChunkResponse(error=str(msg.exc))
                        ):
                            return
                        while True:
                            tail = inbound.get()
                            if tail is _INBOUND_END:
                                break
                        put_outbound(_OUTBOUND_END)
                        return

                    item = msg
                    assert isinstance(item, _DecodedStreamRequest)
                    req = item.request
                    if combine_eights_for_stream is None:
                        combine_eights_for_stream = (
                            self._combine_8_frame_chunks and req.num_frames == 8
                        )
                    if combine_eights_for_stream and req.num_frames == 8:
                        if not append_coalesced_request(item):
                            return
                        continue

                    if combine_eights_for_stream and not drain_coalesced(final=True):
                        return
                    if not process_requests([item]):
                        return
            except Exception:
                logger.exception("upscale_video: GPU worker crashed")
                try:
                    put_outbound(
                        pb2.UpscaleChunkResponse(error="internal GPU worker error")
                    )
                    put_outbound(_OUTBOUND_END)
                except Exception:
                    pass

        rx = threading.Thread(target=reader_fn, name="UpscaleVideoRx", daemon=True)
        gx = threading.Thread(target=gpu_fn, name="UpscaleVideoGPU", daemon=True)
        rx.start()
        gx.start()
        try:
            while not abort.is_set():
                try:
                    item = outbound.get(timeout=0.25)
                except queue.Empty:
                    if not context.is_active():
                        break
                    continue
                if item is _OUTBOUND_END:
                    break
                assert isinstance(item, pb2.UpscaleChunkResponse)
                with count_lock:
                    counts["sent"] += 1
                yield item
        finally:
            abort.set()
            gx.join(timeout=120.0)
            rx.join(timeout=5.0)
            cache_holder[0] = None
            logger.info("upscale_video stream {}: cache released", session_id_box[0])


def _add_flash_vsr_servicer_to_server(servicer: FlashVSR, server: grpc.Server) -> None:
    pb2_grpc.add_FlashVSRServicer_to_server(servicer, server)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="FlashVSR gRPC server")
    parser.add_argument(
        "--model_path",
        default=_default_model_cache_path(),
        help=(
            "Compatibility/status hint for migrated clients. FlashVSR "
            "checkpoints are resolved by flashvsr.config and cached through "
            "HuggingFace/FLASHDREAMS_CACHE_DIR (default: %(default)s)."
        ),
    )
    parser.add_argument("--model_name", default="FlashVSR-v1.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--workers", type=int, default=4, help="gRPC thread pool size")
    parser.add_argument("--max_message_mb", type=int, default=DEFAULT_MAX_MESSAGE_MB)
    parser.add_argument(
        "--default_H", type=int, default=704, help="Default input height"
    )
    parser.add_argument(
        "--default_W", type=int, default=1280, help="Default input width"
    )
    parser.add_argument("--default_scale", type=int, default=2, choices=[2, 4])
    parser.add_argument(
        "--default_sparse_ratio",
        "--sparse_ratio",
        dest="default_sparse_ratio",
        type=float,
        default=1.5,
        help=(
            "Default block-sparse attention ratio. 1.5 is faster; 2.0 is "
            "more stable. Ignored by --attention_mode full. --sparse_ratio "
            "is accepted for runner parity."
        ),
    )
    parser.add_argument(
        "--attention_mode",
        choices=["sparse", "full", "auto"],
        default="sparse",
        help=(
            "Attention backend for the FlashVSR DiT. sparse requires the "
            "block_sparse_attn CUDA extension (a hard dependency of the "
            "FlashVSR integration); if it cannot be imported the server "
            "will fail loudly at startup. Pass --attention_mode auto to fall "
            "back to dense attention, or --attention_mode full to opt "
            "into dense attention instead (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for projector / DiT / decoder.",
    )
    parser.add_argument(
        "--cuda_graph",
        action="store_true",
        help=(
            "Capture the steady-state DiT call into a CUDA graph. Implies --compile."
        ),
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"]
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--stream_inbound_queue_depth",
        type=int,
        default=DEFAULT_STREAM_INBOUND_QUEUE_DEPTH,
        help=(
            "upscale_video only: max incoming request chunks buffered per stream "
            "before gRPC backpressure reaches the client (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--stream_chunk_queue_depth",
        type=int,
        default=DEFAULT_STREAM_CHUNK_QUEUE_DEPTH,
        help=(
            "upscale_video only: max UpscaleChunkResponse messages queued between GPU "
            "completion and gRPC send (FIFO). Larger values overlap send with the next "
            "chunk on the GPU (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--no_combine_8_frame_chunks",
        action="store_true",
        help=(
            "Run FlashVSR on every incoming 8-frame request directly, instead "
            "of the default behavior of coalescing two 8-frame requests into "
            "one 16-frame FlashVSR call (with a 13-frame cold start on the "
            "first call). Smaller chunks reduce per-request latency because "
            "the server does not wait for a second 8-frame request before "
            "running the model, but usually hurt throughput since FlashVSR "
            "is most efficient at 16-frame chunks."
        ),
    )
    parser.add_argument(
        "--viewer_port",
        type=int,
        default=0,
        help=(
            "Start an HTTP MJPEG viewer on this port for upsampled frames. "
            "Disabled when 0 (default)."
        ),
    )
    parser.add_argument(
        "--viewer_host",
        default="0.0.0.0",
        help="HTTP viewer bind host when --viewer_port is set (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--viewer_jpeg_quality",
        type=int,
        default=DEFAULT_VIEWER_JPEG_QUALITY,
        help="JPEG quality for the HTTP viewer stream (default: %(default)s).",
    )
    parser.add_argument(
        "--viewer_jpeg_backend",
        choices=["auto", "cuda", "pillow"],
        default=DEFAULT_VIEWER_JPEG_BACKEND,
        help=(
            "JPEG encoder for the HTTP viewer. auto uses torchvision CUDA JPEG "
            "when available and falls back to Pillow (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--viewer_chunk_queue_depth",
        type=int,
        default=DEFAULT_VIEWER_CHUNK_QUEUE_DEPTH,
        help=(
            "Max completed chunks queued for paced HTTP viewer playback "
            "(default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--viewer_max_fps",
        type=float,
        default=DEFAULT_VIEWER_MAX_FPS,
        help=(
            "Upper FPS cap for HTTP viewer playback. The viewer otherwise uses "
            "the measured FlashVSR generation speed (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--viewer_frame_stride",
        type=int,
        default=DEFAULT_VIEWER_FRAME_STRIDE,
        help=(
            "Publish every Nth upsampled frame to the HTTP viewer. Higher values "
            "reduce CPU copy/JPEG cost and browser bandwidth. gRPC metadata still "
            "reports the full model output frame count (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--viewer_return_grpc_frames",
        action="store_true",
        help=(
            "When the HTTP viewer is enabled, still include raw RGB frames in gRPC "
            "responses. By default viewer mode omits them to save bandwidth."
        ),
    )
    args = parser.parse_args()
    if not 1 <= args.viewer_jpeg_quality <= 100:
        parser.error("--viewer_jpeg_quality must be between 1 and 100")
    if args.viewer_frame_stride < 1:
        parser.error("--viewer_frame_stride must be at least 1")
    attention_mode = _resolve_attention_mode(args.attention_mode)

    compile_network = bool(args.compile or args.cuda_graph)
    if args.cuda_graph and not args.compile:
        logger.info("--cuda_graph implies --compile; enabling compile too.")

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    max_bytes = args.max_message_mb * 1024 * 1024
    options = [
        ("grpc.max_send_message_length", max_bytes),
        ("grpc.max_receive_message_length", max_bytes),
    ]
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.workers), options=options
    )
    addr = f"[::]:{args.port}"
    try:
        bound_port = server.add_insecure_port(addr)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Failed to bind gRPC server to {addr}; another process is likely "
            "already listening on that port."
        ) from exc
    if bound_port == 0:
        raise RuntimeError(
            f"Failed to bind gRPC server to {addr}; another process is likely "
            "already listening on that port."
        )

    viewer: StreamingViewer | None = None
    if args.viewer_port:
        viewer = StreamingViewer(
            host=args.viewer_host,
            port=args.viewer_port,
            jpeg_quality=args.viewer_jpeg_quality,
            jpeg_backend=args.viewer_jpeg_backend,
            chunk_queue_depth=args.viewer_chunk_queue_depth,
            max_fps=args.viewer_max_fps,
            frame_stride=args.viewer_frame_stride,
        )
        viewer.start()

    servicer = FlashVSR(
        model_path=args.model_path,
        model_name=args.model_name,
        default_H=args.default_H,
        default_W=args.default_W,
        default_scale=args.default_scale,
        default_sparse_ratio=args.default_sparse_ratio,
        attention_mode=attention_mode,
        compile_network=compile_network,
        use_cuda_graph=args.cuda_graph,
        dtype=dtype_map[args.dtype],
        device=args.device,
        stream_inbound_queue_depth=args.stream_inbound_queue_depth,
        stream_chunk_queue_depth=args.stream_chunk_queue_depth,
        combine_8_frame_chunks=not args.no_combine_8_frame_chunks,
        viewer=viewer,
        omit_grpc_frames_when_viewing=not args.viewer_return_grpc_frames,
    )

    _add_flash_vsr_servicer_to_server(servicer, server)
    server.start()
    logger.info(
        "Server listening on [::]:{} (max msg {} MB)",
        bound_port,
        args.max_message_mb,
    )

    stop_requested = threading.Event()

    def _on_signal(signum: int, _frame: object) -> None:
        if stop_requested.is_set():
            logger.warning("Second signal {} received; forcing exit.", signum)
            os._exit(130)
        logger.info("Signal {} received; requesting shutdown.", signum)
        stop_requested.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        while not stop_requested.wait(timeout=0.5):
            pass
    finally:
        logger.info("Shutting down ...")
        stopped = server.stop(grace=5)
        if not stopped.wait(timeout=10):
            logger.warning(
                "gRPC server.stop did not complete within 10s; exiting anyway."
            )
        if viewer is not None:
            try:
                viewer.stop()
            except Exception:
                logger.exception("viewer.stop() raised; continuing shutdown")
        sys.exit(0)


if __name__ == "__main__":
    main()
