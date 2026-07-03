#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real-ESRGAN implementation of the FlashVSR uplift gRPC protocol."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
import uuid
from concurrent import futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# On Windows, PyTorch/BLAS worker pools can spin several CPU cores while this
# GPU-bound server is mostly waiting on CUDA. Keep the default lean unless the
# caller explicitly asks for more CPU parallelism.
if os.name == "nt":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import grpc
import numpy as np
import torch
from loguru import logger

from flashdreams.serving.uplift.protos import uplift_pb2 as pb2
from flashdreams.serving.uplift.protos import uplift_pb2_grpc as pb2_grpc
from flashdreams.serving.uplift.streaming_view import (
    DEFAULT_VIEWER_CHUNK_QUEUE_DEPTH,
    DEFAULT_VIEWER_FRAME_STRIDE,
    DEFAULT_VIEWER_JPEG_BACKEND,
    DEFAULT_VIEWER_JPEG_QUALITY,
    DEFAULT_VIEWER_MAX_FPS,
    DEFAULT_VIEWER_PLAYBACK_FPS,
    StreamingViewer,
    decode_jpeg_rgb,
    encode_jpeg_cuda_tensor,
)
from realesrgan.upsampler import RealESRGANUpsampler, default_model_name

DEFAULT_PORT = 50052
DEFAULT_MAX_MESSAGE_MB = 512
DEFAULT_STREAM_QUEUE_DEPTH = 16
Scale = Literal[2, 4]


def _positive_int_env(*names: str) -> int | None:
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        try:
            parsed = int(value)
        except ValueError:
            continue
        if parsed > 0:
            return parsed
    return None


if os.name == "nt":
    torch.set_num_threads(
        _positive_int_env("TORCH_NUM_THREADS", "OMP_NUM_THREADS") or 1
    )


@dataclass
class _Session:
    scale: Scale
    model_name: str
    created_at: float = field(default_factory=time.time)


@dataclass(frozen=True)
class _RunChunkResult:
    frames_rgb: bytes
    frames_out: np.ndarray | None
    num_frames: int
    height: int
    width: int
    elapsed_ms: float


@dataclass(frozen=True)
class _FrameRunResult:
    output: torch.Tensor
    model_ms: float | None


class _StaticShapeExecutor:
    """Run Real-ESRGAN with stable CUDA input storage for one chunk shape."""

    def __init__(
        self,
        upsampler: RealESRGANUpsampler,
        num_frames: int,
        height: int,
        width: int,
    ) -> None:
        if upsampler.device.type != "cuda":
            raise ValueError("Static Real-ESRGAN executor requires a CUDA device")
        if upsampler.tile > 0:
            raise ValueError("Static Real-ESRGAN executor does not support tiling")
        self.upsampler = upsampler
        self.num_frames = int(num_frames)
        self.height = int(height)
        self.width = int(width)
        self.scale = int(upsampler.scale)
        self.device = upsampler.device
        self.dtype = upsampler.dtype
        self._input = torch.empty(
            (self.num_frames, 3, self.height, self.width),
            device=self.device,
            dtype=self.dtype,
        )
        total_pad_h, total_pad_w = self._total_padding()
        self._crop = (
            total_pad_h * upsampler.model_scale,
            total_pad_w * upsampler.model_scale,
        )
        self._model_input = (
            torch.empty(
                (
                    self.num_frames,
                    3,
                    self.height + total_pad_h,
                    self.width + total_pad_w,
                ),
                device=self.device,
                dtype=self.dtype,
            )
            if total_pad_h or total_pad_w
            else self._input
        )
        self._total_pad_h = total_pad_h
        self._total_pad_w = total_pad_w

    @property
    def output_height(self) -> int:
        return self.height * self.scale

    @property
    def output_width(self) -> int:
        return self.width * self.scale

    @torch.no_grad()
    def run_frames(
        self, frames_rgb: np.ndarray, *, profile_model_timing: bool = False
    ) -> _FrameRunResult:
        self._copy_frames(frames_rgb)
        self._fill_model_input()

        start: torch.cuda.Event | None = None
        end: torch.cuda.Event | None = None
        if self.upsampler.compile_model:
            torch.compiler.cudagraph_mark_step_begin()
        if profile_model_timing:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        output = self.upsampler.model(self._model_input)
        if end is not None:
            end.record()
        output = self.upsampler._crop_output(output, self._crop)
        return _FrameRunResult(output=output, model_ms=_elapsed_ms(start, end))

    def _total_padding(self) -> tuple[int, int]:
        pad_h = int(self.upsampler.pre_pad)
        pad_w = int(self.upsampler.pre_pad)
        if self.upsampler.model_scale == 2:
            pad_h += (2 - (self.height + pad_h) % 2) % 2
            pad_w += (2 - (self.width + pad_w) % 2) % 2
        return pad_h, pad_w

    def _copy_frames(self, frames_rgb: np.ndarray) -> None:
        expected = (self.num_frames, self.height, self.width, 3)
        if frames_rgb.shape != expected:
            raise ValueError(
                f"Static executor expected {expected}, "
                f"got {frames_rgb.shape}"
            )
        cpu = torch.from_numpy(np.ascontiguousarray(frames_rgb))
        self._input.copy_(cpu.permute(0, 3, 1, 2), non_blocking=False)
        self._input.div_(255.0)

    def _fill_model_input(self) -> None:
        if self._model_input is self._input:
            return
        h = self.height
        w = self.width
        pad_h = self._total_pad_h
        pad_w = self._total_pad_w
        self._model_input[:, :, :h, :w].copy_(self._input)
        if pad_w:
            self._model_input[:, :, :h, w:].copy_(
                torch.flip(
                    self._model_input[:, :, :h, w - pad_w - 1 : w - 1],
                    dims=[3],
                )
            )
        if pad_h:
            self._model_input[:, :, h:, :].copy_(
                torch.flip(
                    self._model_input[:, :, h - pad_h - 1 : h - 1, :],
                    dims=[2],
                )
            )


def _elapsed_ms(
    start: torch.cuda.Event | None, end: torch.cuda.Event | None
) -> float | None:
    if start is None or end is None:
        return None
    end.synchronize()
    return float(start.elapsed_time(end))


def _output_tensor_to_rgb(output: torch.Tensor) -> np.ndarray:
    """Convert direct Real-ESRGAN model output [1, 3, H, W] in [0, 1] to RGB."""
    return np.ascontiguousarray(
        output[0]
        .detach()
        .float()
        .clamp(0, 1)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .contiguous()
        .cpu()
        .numpy()
    )


def _output_tensor_batch_to_rgb(output: torch.Tensor) -> np.ndarray:
    """Convert direct Real-ESRGAN model output [T, 3, H, W] in [0, 1] to RGB."""
    return np.ascontiguousarray(
        output.detach()
        .float()
        .clamp(0, 1)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
        .cpu()
        .numpy()
    )


def _resolve_scale(scale: int) -> Scale:
    if scale == 2:
        return 2
    if scale == 4:
        return 4
    raise ValueError(f"Real-ESRGAN scale must be 2 or 4, got {scale}")


def _synchronize_device(device: torch.device | str) -> None:
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _context_set_error(context: grpc.ServicerContext | None, details: str) -> None:
    if context is None:
        return
    context.set_code(grpc.StatusCode.INTERNAL)
    context.set_details(details)


class RealESRGANUplift(pb2_grpc.VideoUpliftServicer):
    """Serve Real-ESRGAN behind the existing uplift gRPC surface."""

    def __init__(
        self,
        *,
        default_scale: int = 2,
        model_name: str | None = None,
        model_path: str | Path | None = None,
        tile: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 10,
        half: bool = True,
        compile_model: bool = False,
        compile_mode: str = "reduce-overhead",
        device: str | torch.device | None = None,
        stream_queue_depth: int = DEFAULT_STREAM_QUEUE_DEPTH,
        static_batch_frames: int = 1,
        defer_static_sync: bool = False,
        viewer: StreamingViewer | None = None,
        omit_grpc_frames_when_viewing: bool = False,
        warmup_height: int = 64,
        warmup_width: int = 64,
        warmup_frames: int = 3,
        warmup: bool = True,
        profile_model_timing: bool = True,
        load_checkpoint: bool = True,
    ) -> None:
        self._default_scale = _resolve_scale(default_scale)
        self._default_model_name = model_name
        self._model_path = model_path
        self._tile = int(tile)
        self._tile_pad = int(tile_pad)
        self._pre_pad = int(pre_pad)
        self._half = bool(half)
        self._compile_model = bool(compile_model)
        self._compile_mode = compile_mode
        self._device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._stream_queue_depth = max(1, int(stream_queue_depth))
        self._static_batch_frames = max(1, int(static_batch_frames))
        self._defer_static_sync = bool(defer_static_sync)
        self._viewer = viewer
        self._profile_model_timing = bool(profile_model_timing)
        self._omit_grpc_frames_when_viewing = bool(
            omit_grpc_frames_when_viewing and viewer is not None
        )
        self._load_checkpoint = bool(load_checkpoint)

        self._upsampler_pool: dict[tuple[Scale, str], RealESRGANUpsampler] = {}
        self._executor_pool: dict[
            tuple[Scale, str, int, int, int], _StaticShapeExecutor
        ] = {}
        self._pool_lock = threading.Lock()
        self._sessions: dict[str, _Session] = {}
        self._stream_sessions: set[str] = set()
        self._sessions_lock = threading.Lock()
        self._gpu_lock = threading.Lock()
        self._warned_cuda_viewer_fallback = False

        logger.info(
            "Loading Real-ESRGAN model scale={} model={} device={} half={} "
            "compile={} tile={} static_batch_frames={} defer_static_sync={} "
            "profile_model_timing={} torch_threads={} omp_threads={} ...",
            self._default_scale,
            self._model_name_for_scale(self._default_scale),
            self._device,
            self._half,
            self._compile_model,
            self._tile,
            self._static_batch_frames,
            self._defer_static_sync,
            self._profile_model_timing,
            torch.get_num_threads(),
            os.environ.get("OMP_NUM_THREADS"),
        )
        default_upsampler = self._get_upsampler(self._default_scale)
        if warmup:
            self._warm_up(
                default_upsampler,
                self._default_scale,
                warmup_height,
                warmup_width,
                warmup_frames,
            )
        logger.info("Real-ESRGAN uplift server ready.")

    def _model_name_for_scale(self, scale: Scale) -> str:
        return self._default_model_name or default_model_name(scale)

    def _get_upsampler(self, scale: int) -> RealESRGANUpsampler:
        resolved_scale = _resolve_scale(scale)
        model_name = self._model_name_for_scale(resolved_scale)
        key = (resolved_scale, model_name)
        with self._pool_lock:
            if key not in self._upsampler_pool:
                logger.info(
                    "Creating Real-ESRGAN upsampler scale={} model={} device={}",
                    resolved_scale,
                    model_name,
                    self._device,
                )
                self._upsampler_pool[key] = RealESRGANUpsampler(
                    scale=resolved_scale,
                    model_name=model_name,
                    model_path=self._model_path,
                    tile=self._tile,
                    tile_pad=self._tile_pad,
                    pre_pad=self._pre_pad,
                    half=self._half,
                    compile_model=self._compile_model,
                    compile_mode=self._compile_mode,
                    device=self._device,
                    load_checkpoint=self._load_checkpoint,
                )
            return self._upsampler_pool[key]

    def _get_executor(
        self,
        upsampler: RealESRGANUpsampler,
        scale: int,
        num_frames: int,
        height: int,
        width: int,
    ) -> _StaticShapeExecutor | None:
        if self._device.type != "cuda" or self._tile > 0:
            return None
        if not hasattr(upsampler, "model"):
            return None
        resolved_scale = _resolve_scale(scale)
        model_name = self._model_name_for_scale(resolved_scale)
        key = (resolved_scale, model_name, int(num_frames), int(height), int(width))
        with self._pool_lock:
            if key not in self._executor_pool:
                logger.info(
                    "Creating static Real-ESRGAN executor for T={} {}x{} "
                    "scale={} model={} compile_mode={}",
                    num_frames,
                    height,
                    width,
                    resolved_scale,
                    model_name,
                    self._compile_mode if self._compile_model else "disabled",
                )
                self._executor_pool[key] = _StaticShapeExecutor(
                    upsampler=upsampler,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                )
            return self._executor_pool[key]

    def _warm_up(
        self,
        upsampler: RealESRGANUpsampler,
        scale: int,
        height: int,
        width: int,
        frames: int,
    ) -> None:
        height = max(8, int(height))
        width = max(8, int(width))
        frames = max(1, int(frames))
        logger.info(
            "Warming up Real-ESRGAN with {} {}×{} frame(s) ...",
            frames,
            height,
            width,
        )
        warmup_chunk = np.zeros((frames, height, width, 3), dtype=np.uint8)
        start = time.perf_counter()
        with self._gpu_lock:
            batch_frames = min(self._static_batch_frames, frames)
            executor = self._get_executor(
                upsampler, scale, batch_frames, height, width
            )
            if executor is not None:
                for offset in range(0, frames, batch_frames):
                    batch = warmup_chunk[offset : offset + batch_frames]
                    batch_executor = self._get_executor(
                        upsampler,
                        scale,
                        int(batch.shape[0]),
                        height,
                        width,
                    )
                    if batch_executor is None:
                        raise RuntimeError("Static executor disappeared during warmup")
                    batch_executor.run_frames(batch)
            else:
                for frame in warmup_chunk:
                    self._upsample_frame_rgb(upsampler, frame)
        _synchronize_device(self._device)
        logger.info(
            "Real-ESRGAN warmup complete in {:.0f} ms.",
            (time.perf_counter() - start) * 1000.0,
        )

    def _request_to_frames_rgb(self, request: pb2.UpscaleChunkRequest) -> np.ndarray:
        total = int(request.num_frames)
        height = int(request.height)
        width = int(request.width)
        if total <= 0:
            raise ValueError("num_frames must be positive")
        if request.frame_encoding == pb2.FRAME_ENCODING_JPEG:
            if len(request.frames_jpeg) != total:
                raise ValueError(
                    f"frame_encoding=JPEG requires {total} frames_jpeg payloads; "
                    f"got {len(request.frames_jpeg)}"
                )
            frames = [decode_jpeg_rgb(frame) for frame in request.frames_jpeg]
            arr = np.stack(frames, axis=0)
            if height and width and arr.shape[1:3] != (height, width):
                raise ValueError(
                    f"JPEG payload dimensions {arr.shape[1]}×{arr.shape[2]} "
                    f"do not match request height/width {height}×{width}"
                )
            return np.ascontiguousarray(arr)

        expected = total * height * width * 3
        if len(request.frames_rgb) != expected:
            raise ValueError(
                f"RAW_RGB payload has {len(request.frames_rgb)} bytes; "
                f"expected {expected} for {total}×{height}×{width}×3"
            )
        return np.ascontiguousarray(
            np.frombuffer(request.frames_rgb, dtype=np.uint8).reshape(
                total, height, width, 3
            )
        )

    def _upsample_frame_rgb(
        self, upsampler: RealESRGANUpsampler, frame: np.ndarray
    ) -> np.ndarray:
        tensor = torch.from_numpy(
            np.ascontiguousarray(frame).astype(np.float32).transpose(2, 0, 1)
        )
        tensor = tensor / 127.5 - 1.0
        output = upsampler.upsample_frame_tensor(tensor)
        arr = (
            ((output + 1.0) * 127.5)
            .clamp(0, 255)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .contiguous()
            .numpy()
        )
        return np.ascontiguousarray(arr)

    def _should_omit_response_frames(self, request: pb2.UpscaleChunkRequest) -> bool:
        return bool(request.display_only or self._omit_grpc_frames_when_viewing)

    def _run_chunk(
        self,
        upsampler: RealESRGANUpsampler,
        request: pb2.UpscaleChunkRequest,
        *,
        return_frames: bool,
    ) -> _RunChunkResult:
        decode_t0 = time.perf_counter()
        frames = self._request_to_frames_rgb(request)
        decode_ms = (time.perf_counter() - decode_t0) * 1000.0

        scale = request.scale or self._default_scale
        batch_frames = min(self._static_batch_frames, int(frames.shape[0]))
        executor = self._get_executor(
            upsampler,
            scale,
            batch_frames,
            int(frames.shape[1]),
            int(frames.shape[2]),
        )
        if executor is not None:
            return self._run_chunk_static(
                upsampler,
                scale,
                request,
                frames,
                decode_ms=decode_ms,
                return_frames=return_frames,
            )
        return self._run_chunk_dynamic(
            upsampler,
            request,
            frames,
            decode_ms=decode_ms,
            return_frames=return_frames,
        )

    def _run_chunk_dynamic(
        self,
        upsampler: RealESRGANUpsampler,
        request: pb2.UpscaleChunkRequest,
        frames: np.ndarray,
        *,
        decode_ms: float,
        return_frames: bool,
    ) -> _RunChunkResult:
        infer_t0 = time.perf_counter()
        with self._gpu_lock:
            output_frames = [
                self._upsample_frame_rgb(upsampler, frame) for frame in frames
            ]
        _synchronize_device(self._device)
        infer_ms = (time.perf_counter() - infer_t0) * 1000.0

        frames_out = np.ascontiguousarray(np.stack(output_frames, axis=0))
        num_frames, height, width, _channels = frames_out.shape
        if self._viewer is not None:
            self._viewer.enqueue_original_chunk(frames, elapsed_ms=infer_ms)
            self._viewer.enqueue_upscaled_chunk(frames_out, elapsed_ms=infer_ms)

        encode_t0 = time.perf_counter()
        frames_rgb = frames_out.tobytes() if return_frames else b""
        encode_ms = (time.perf_counter() - encode_t0) * 1000.0
        logger.info(
            "  chunk {} timing: decode_in={:.0f} ms infer={:.0f} ms "
            "encode_out={:.0f} ms total={:.0f} ms (out {}×{})",
            request.chunk_index,
            decode_ms,
            infer_ms,
            encode_ms,
            decode_ms + infer_ms + encode_ms,
            width,
            height,
        )
        return _RunChunkResult(
            frames_rgb=frames_rgb,
            frames_out=frames_out if return_frames else None,
            num_frames=num_frames,
            height=height,
            width=width,
            elapsed_ms=infer_ms,
        )

    def _run_chunk_static(
        self,
        upsampler: RealESRGANUpsampler,
        scale: int,
        request: pb2.UpscaleChunkRequest,
        frames: np.ndarray,
        *,
        decode_ms: float,
        return_frames: bool,
    ) -> _RunChunkResult:
        viewer = self._viewer
        wants_viewer = viewer is not None
        device = torch.device(upsampler.device)
        wants_cuda_jpegs = (
            wants_viewer
            and viewer.jpeg_backend in ("auto", "cuda")
            and device.type == "cuda"
        )
        upscaled_jpegs: list[bytes] = []
        output_chunks: list[np.ndarray] = []
        frames_out: np.ndarray | None = None
        need_cpu = return_frames or (wants_viewer and not wants_cuda_jpegs)
        model_ms_total = 0.0
        batch_frames = min(self._static_batch_frames, int(frames.shape[0]))
        infer_t0 = time.perf_counter()
        with self._gpu_lock:
            for offset in range(0, int(frames.shape[0]), batch_frames):
                batch = frames[offset : offset + batch_frames]
                executor = self._get_executor(
                    upsampler,
                    scale,
                    int(batch.shape[0]),
                    int(batch.shape[1]),
                    int(batch.shape[2]),
                )
                if executor is None:
                    raise RuntimeError("Static executor disappeared during inference")
                result = executor.run_frames(
                    batch,
                    profile_model_timing=self._profile_model_timing,
                )
                if result.model_ms is not None:
                    model_ms_total += result.model_ms
                if wants_cuda_jpegs:
                    try:
                        upscaled_jpegs.extend(
                            self._encode_viewer_cuda_jpeg(result.output)
                        )
                    except Exception:
                        if viewer is not None and viewer.jpeg_backend == "cuda":
                            raise
                        if not self._warned_cuda_viewer_fallback:
                            logger.opt(exception=True).warning(
                                "CUDA viewer JPEG encode unavailable; falling "
                                "back to CPU/Pillow encoding"
                            )
                            self._warned_cuda_viewer_fallback = True
                        wants_cuda_jpegs = False
                        need_cpu = True
                if need_cpu:
                    output_chunks.append(_output_tensor_batch_to_rgb(result.output))
                elif not wants_cuda_jpegs and not self._defer_static_sync:
                    _synchronize_device(device)
            if not need_cpu and not upscaled_jpegs and self._defer_static_sync:
                # Keep the model op ordered before the response when neither
                # gRPC nor viewer consumes the output tensor.
                _synchronize_device(device)
        infer_ms = (time.perf_counter() - infer_t0) * 1000.0
        if model_ms_total > 0.0:
            infer_ms = model_ms_total
        if output_chunks:
            frames_out = (
                output_chunks[0]
                if len(output_chunks) == 1
                else np.ascontiguousarray(np.concatenate(output_chunks, axis=0))
            )
        if viewer is not None:
            viewer.enqueue_original_chunk(frames, elapsed_ms=infer_ms)
            if upscaled_jpegs:
                viewer.enqueue_upscaled_jpeg_chunk(
                    upscaled_jpegs,
                    elapsed_ms=infer_ms,
                )
            elif frames_out is not None:
                viewer.enqueue_upscaled_chunk(frames_out, elapsed_ms=infer_ms)

        encode_t0 = time.perf_counter()
        frames_rgb = frames_out.tobytes() if return_frames and frames_out is not None else b""
        encode_ms = (time.perf_counter() - encode_t0) * 1000.0
        output_height = int(frames.shape[1]) * int(scale)
        output_width = int(frames.shape[2]) * int(scale)
        logger.info(
            "  chunk {} timing: decode_in={:.0f} ms infer={:.0f} ms "
            "encode_out={:.0f} ms total={:.0f} ms (out {}×{} static)",
            request.chunk_index,
            decode_ms,
            infer_ms,
            encode_ms,
            decode_ms + infer_ms + encode_ms,
            output_width,
            output_height,
        )
        return _RunChunkResult(
            frames_rgb=frames_rgb,
            frames_out=frames_out if return_frames else None,
            num_frames=int(frames.shape[0]),
            height=output_height,
            width=output_width,
            elapsed_ms=infer_ms,
        )

    def _encode_viewer_cuda_jpeg(self, output: torch.Tensor) -> list[bytes]:
        # encode_jpeg_cuda_tensor expects [1, 3, T, H, W] in [-1, 1].
        video = output.mul(2.0).sub(1.0).permute(1, 0, 2, 3).unsqueeze(0)
        return encode_jpeg_cuda_tensor(
            video,
            quality=self._viewer.jpeg_quality if self._viewer is not None else 90,
            frame_stride=1,
        )

    def get_status(self, request, context):
        with self._sessions_lock:
            active = sorted([*self._sessions.keys(), *self._stream_sessions])
        return pb2.StatusResponse(
            ready=True,
            device=str(self._device),
            model_name=f"RealESRGAN/{self._model_name_for_scale(self._default_scale)}",
            active_sessions=active,
        )

    def start_session(self, request, context):
        session_id = request.session_id or str(uuid.uuid4())
        scale = _resolve_scale(request.scale or self._default_scale)
        model_name = self._model_name_for_scale(scale)
        with self._sessions_lock:
            if session_id in self._sessions:
                return pb2.StartSessionResponse(
                    session_id=session_id,
                    success=False,
                    error=f"session '{session_id}' already exists; call end_session first",
                )
        try:
            self._get_upsampler(scale)
            with self._sessions_lock:
                self._sessions[session_id] = _Session(scale=scale, model_name=model_name)
            logger.info("start_session {} (scale={} model={})", session_id, scale, model_name)
            return pb2.StartSessionResponse(session_id=session_id, success=True)
        except Exception as exc:
            logger.exception("start_session failed")
            return pb2.StartSessionResponse(success=False, error=str(exc))

    def end_session(self, request, context):
        with self._sessions_lock:
            self._sessions.pop(request.session_id, None)
        logger.info("end_session {}", request.session_id)
        return pb2.EndSessionResponse(success=True)

    def upscale_chunk(self, request, context):
        with self._sessions_lock:
            session = self._sessions.get(request.session_id)
        if session is None:
            message = f"session '{request.session_id}' not found; call start_session first"
            if context is not None:
                context.set_code(grpc.StatusCode.NOT_FOUND)
                context.set_details(message)
            return pb2.UpscaleChunkResponse(error=message)

        try:
            upsampler = self._get_upsampler(session.scale)
            omit_frames = self._should_omit_response_frames(request)
            result = self._run_chunk(
                upsampler,
                request,
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
            _context_set_error(context, str(exc))
            return pb2.UpscaleChunkResponse(error=str(exc))

    def upscale_video(self, request_iterator, context):
        session_id = str(uuid.uuid4())
        stream_started = False
        scale = self._default_scale
        upsampler: RealESRGANUpsampler | None = None
        try:
            for request in request_iterator:
                if not stream_started:
                    session_id = request.session_id or session_id
                    scale = _resolve_scale(request.scale or self._default_scale)
                    upsampler = self._get_upsampler(scale)
                    with self._sessions_lock:
                        self._stream_sessions.add(session_id)
                    stream_started = True
                    logger.info(
                        "upscale_video stream {} ({}×{} scale={}) queue_depth={}",
                        session_id,
                        request.height or request.input_height,
                        request.width or request.input_width,
                        scale,
                        self._stream_queue_depth,
                    )
                assert upsampler is not None
                omit_frames = self._should_omit_response_frames(request)
                result = self._run_chunk(
                    upsampler,
                    request,
                    return_frames=not omit_frames,
                )
                logger.info(
                    "upscale_video {} chunk={} T={} infer={:.0f} ms",
                    session_id,
                    request.chunk_index,
                    request.num_frames,
                    result.elapsed_ms,
                )
                yield pb2.UpscaleChunkResponse(
                    session_id=session_id,
                    frames_rgb=result.frames_rgb,
                    num_frames=result.num_frames,
                    height=result.height,
                    width=result.width,
                    chunk_index=request.chunk_index,
                    elapsed_ms=result.elapsed_ms,
                    frames_omitted=omit_frames,
                )
        except Exception as exc:
            logger.exception("upscale_video failed")
            yield pb2.UpscaleChunkResponse(session_id=session_id, error=str(exc))
        finally:
            with self._sessions_lock:
                self._stream_sessions.discard(session_id)
            logger.info("upscale_video stream {}: completed", session_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-ESRGAN uplift gRPC server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--workers", type=int, default=4, help="gRPC thread pool size")
    parser.add_argument("--max_message_mb", type=int, default=DEFAULT_MAX_MESSAGE_MB)
    parser.add_argument("--default_scale", type=int, choices=[2, 4], default=2)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--tile", type=int, default=0)
    parser.add_argument("--tile_pad", type=int, default=10)
    parser.add_argument("--pre_pad", type=int, default=10)
    parser.add_argument("--fp32", action="store_true", help="Disable fp16 on CUDA.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile().")
    parser.add_argument("--compile_mode", default="reduce-overhead")
    parser.add_argument(
        "--profile_model_timing",
        dest="profile_model_timing",
        action="store_true",
        default=True,
        help=(
            "Use CUDA events for model-only timing. This synchronizes each chunk "
            "and is enabled by default because it avoids excess CPU spin on "
            "Windows for this workload."
        ),
    )
    parser.add_argument(
        "--no_profile_model_timing",
        dest="profile_model_timing",
        action="store_false",
        help=(
            "Disable CUDA event timing. This reduces explicit synchronization "
            "but can increase CPU usage on Windows."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--stream_queue_depth",
        type=int,
        default=DEFAULT_STREAM_QUEUE_DEPTH,
        help="Reported per-stream queue depth; GPU work is serialized by a device lock.",
    )
    parser.add_argument(
        "--static_batch_frames",
        type=int,
        default=1,
        help=(
            "Number of same-size frames per static CUDA model call. Default 1 "
            "keeps Real-ESRGAN's fastest observed path on Windows."
        ),
    )
    parser.add_argument(
        "--defer_static_sync",
        action="store_true",
        help=(
            "Synchronize once after all static microbatches instead of after "
            "each microbatch. This can raise CPU usage on Windows."
        ),
    )
    parser.add_argument("--no_warmup", action="store_true")
    parser.add_argument("--warmup_height", type=int, default=64)
    parser.add_argument("--warmup_width", type=int, default=64)
    parser.add_argument(
        "--warmup_frames",
        type=int,
        default=3,
        help=(
            "Number of same-shape frames to run during warmup. "
            "TorchInductor reduce-overhead may need multiple calls before "
            "the CUDA graph path is fully settled."
        ),
    )
    parser.add_argument(
        "--viewer_port",
        type=int,
        default=0,
        help="Start an HTTP MJPEG viewer on this port. Disabled when 0.",
    )
    parser.add_argument("--viewer_host", default="0.0.0.0")
    parser.add_argument("--viewer_jpeg_quality", type=int, default=DEFAULT_VIEWER_JPEG_QUALITY)
    parser.add_argument(
        "--viewer_jpeg_backend",
        choices=["auto", "cuda", "pillow"],
        default=DEFAULT_VIEWER_JPEG_BACKEND,
    )
    parser.add_argument(
        "--viewer_chunk_queue_depth",
        type=int,
        default=DEFAULT_VIEWER_CHUNK_QUEUE_DEPTH,
    )
    parser.add_argument("--viewer_max_fps", type=float, default=DEFAULT_VIEWER_MAX_FPS)
    parser.add_argument(
        "--viewer_playback_fps",
        type=float,
        default=DEFAULT_VIEWER_PLAYBACK_FPS,
        help=(
            "Steady HTTP viewer playback FPS. If completed chunks back up, "
            "the viewer gradually catches up, capped by --viewer_max_fps. "
            "Set to 0 to pace from model elapsed time."
        ),
    )
    parser.add_argument("--viewer_frame_stride", type=int, default=DEFAULT_VIEWER_FRAME_STRIDE)
    parser.add_argument(
        "--viewer_return_grpc_frames",
        action="store_true",
        help="When viewer is enabled, still include raw RGB frames in gRPC responses.",
    )
    args = parser.parse_args()

    if not 1 <= args.viewer_jpeg_quality <= 100:
        parser.error("--viewer_jpeg_quality must be between 1 and 100")
    if args.viewer_frame_stride < 1:
        parser.error("--viewer_frame_stride must be at least 1")
    if args.viewer_playback_fps < 0:
        parser.error("--viewer_playback_fps must be non-negative")
    if args.warmup_frames < 1:
        parser.error("--warmup_frames must be at least 1")
    if args.static_batch_frames < 1:
        parser.error("--static_batch_frames must be at least 1")

    max_bytes = args.max_message_mb * 1024 * 1024
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.workers),
        options=[
            ("grpc.max_send_message_length", max_bytes),
            ("grpc.max_receive_message_length", max_bytes),
        ],
    )
    addr = f"[::]:{args.port}"
    bound_port = server.add_insecure_port(addr)
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
            playback_fps=args.viewer_playback_fps,
            frame_stride=args.viewer_frame_stride,
        )
        viewer.start()

    servicer = RealESRGANUplift(
        default_scale=args.default_scale,
        model_name=args.model_name,
        model_path=args.model_path,
        tile=args.tile,
        tile_pad=args.tile_pad,
        pre_pad=args.pre_pad,
        half=not args.fp32,
        compile_model=args.compile,
        compile_mode=args.compile_mode,
        device=args.device,
        stream_queue_depth=args.stream_queue_depth,
        static_batch_frames=args.static_batch_frames,
        defer_static_sync=args.defer_static_sync,
        viewer=viewer,
        omit_grpc_frames_when_viewing=not args.viewer_return_grpc_frames,
        warmup_height=args.warmup_height,
        warmup_width=args.warmup_width,
        warmup_frames=args.warmup_frames,
        warmup=not args.no_warmup,
        profile_model_timing=args.profile_model_timing,
    )
    pb2_grpc.add_VideoUpliftServicer_to_server(servicer, server)
    server.start()
    logger.info("Server listening on [::]:{} (max msg {} MB)", bound_port, args.max_message_mb)

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
            logger.warning("gRPC server.stop did not complete within 10s; exiting anyway.")
        if viewer is not None:
            try:
                viewer.stop()
            except Exception:
                logger.exception("viewer.stop() raised; continuing shutdown")
        sys.exit(0)


if __name__ == "__main__":
    main()
