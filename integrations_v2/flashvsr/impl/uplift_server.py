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

"""Standalone gRPC uplift server for FlashVSR video super-resolution."""

from __future__ import annotations

import argparse
import importlib.util
import io
import logging
import secrets
import threading
import time
import uuid
from concurrent import futures
from dataclasses import dataclass, field
from typing import Any, Literal

import grpc
import numpy as np
import torch
from flashvsr.config import build_flashvsr_v1_1
from flashvsr.impl.pipeline import FlashVSRPipeline, FlashVSRPipelineCache
from google.protobuf import descriptor_pool
from google.protobuf.internal import builder

DEFAULT_PORT = 50051
DEFAULT_BIND_HOST = "127.0.0.1"
DEFAULT_MAX_MESSAGE_MB = 512
DEFAULT_INPUT_HEIGHT = 704
DEFAULT_INPUT_WIDTH = 1280
DEFAULT_SCALE = 2
DEFAULT_SPARSE_RATIO = 2.0

AttentionMode = Literal["sparse", "full"]
RequestedAttentionMode = Literal["sparse", "full", "auto"]
Scale = Literal[2, 4]

_PROTO_DESCRIPTOR = b'\n#flashvsr/grpc/protos/flashvsr.proto\x12\x08flashvsr"\x0f\n\rStatusRequest"\\\n\x0eStatusResponse\x12\r\n\x05ready\x18\x01 \x01(\x08\x12\x0e\n\x06device\x18\x02 \x01(\t\x12\x12\n\nmodel_name\x18\x03 \x01(\t\x12\x17\n\x0factive_sessions\x18\x04 \x03(\t"y\n\x13StartSessionRequest\x12\x12\n\nsession_id\x18\x01 \x01(\t\x12\x14\n\x0cinput_height\x18\x02 \x01(\x05\x12\x13\n\x0binput_width\x18\x03 \x01(\x05\x12\r\n\x05scale\x18\x04 \x01(\x05\x12\x14\n\x0csparse_ratio\x18\x05 \x01(\x02"a\n\x14StartSessionResponse\x12\x12\n\nsession_id\x18\x01 \x01(\t\x12\x0f\n\x07success\x18\x02 \x01(\x08\x12\r\n\x05error\x18\x03 \x01(\t\x12\x15\n\rsession_token\x18\x04 \x01(\t">\n\x11EndSessionRequest\x12\x12\n\nsession_id\x18\x01 \x01(\t\x12\x15\n\rsession_token\x18\x02 \x01(\t"%\n\x12EndSessionResponse\x12\x0f\n\x07success\x18\x01 \x01(\x08"\xc8\x02\n\x13UpscaleChunkRequest\x12\x12\n\nsession_id\x18\x01 \x01(\t\x12\x15\n\rsession_token\x18\x0e \x01(\t\x12\x14\n\x0cinput_height\x18\x02 \x01(\x05\x12\x13\n\x0binput_width\x18\x03 \x01(\x05\x12\r\n\x05scale\x18\x04 \x01(\x05\x12\x14\n\x0csparse_ratio\x18\x05 \x01(\x02\x12\x12\n\nframes_rgb\x18\x06 \x01(\x0c\x12\x12\n\nnum_frames\x18\x07 \x01(\x05\x12\x0e\n\x06height\x18\x08 \x01(\x05\x12\r\n\x05width\x18\t \x01(\x05\x12\x13\n\x0bchunk_index\x18\n \x01(\x05\x12/\n\x0eframe_encoding\x18\x0b \x01(\x0e2\x17.flashvsr.FrameEncoding\x12\x13\n\x0bframes_jpeg\x18\x0c \x03(\x0c\x12\x14\n\x0cdisplay_only\x18\r \x01(\x08"\xc1\x01\n\x14UpscaleChunkResponse\x12\x12\n\nsession_id\x18\x01 \x01(\t\x12\x12\n\nframes_rgb\x18\x02 \x01(\x0c\x12\x12\n\nnum_frames\x18\x03 \x01(\x05\x12\x0e\n\x06height\x18\x04 \x01(\x05\x12\r\n\x05width\x18\x05 \x01(\x05\x12\x13\n\x0bchunk_index\x18\x06 \x01(\x05\x12\x12\n\nelapsed_ms\x18\x07 \x01(\x02\x12\r\n\x05error\x18\x08 \x01(\t\x12\x16\n\x0eframes_omitted\x18\t \x01(\x08*D\n\rFrameEncoding\x12\x1a\n\x16FRAME_ENCODING_RAW_RGB\x10\x00\x12\x17\n\x13FRAME_ENCODING_JPEG\x10\x012\x89\x03\n\x08FlashVSR\x12?\n\nget_status\x12\x17.flashvsr.StatusRequest\x1a\x18.flashvsr.StatusResponse\x12N\n\rstart_session\x12\x1d.flashvsr.StartSessionRequest\x1a\x1e.flashvsr.StartSessionResponse\x12H\n\x0bend_session\x12\x1b.flashvsr.EndSessionRequest\x1a\x1c.flashvsr.EndSessionResponse\x12N\n\rupscale_chunk\x12\x1d.flashvsr.UpscaleChunkRequest\x1a\x1e.flashvsr.UpscaleChunkResponse\x12R\n\rupscale_video\x12\x1d.flashvsr.UpscaleChunkRequest\x1a\x1e.flashvsr.UpscaleChunkResponse(\x010\x01b\x06proto3'
"""Serialized historical FlashVSR uplift protocol kept local to this server."""

DESCRIPTOR = descriptor_pool.Default().AddSerializedFile(_PROTO_DESCRIPTOR)
builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, __name__, globals())

# Protobuf's builder creates these symbols dynamically. Bind them explicitly so
# static analysis can follow the service implementation below.
FRAME_ENCODING_JPEG: int = globals()["FRAME_ENCODING_JPEG"]
StatusRequest: Any = globals()["StatusRequest"]
StatusResponse: Any = globals()["StatusResponse"]
StartSessionRequest: Any = globals()["StartSessionRequest"]
StartSessionResponse: Any = globals()["StartSessionResponse"]
EndSessionRequest: Any = globals()["EndSessionRequest"]
EndSessionResponse: Any = globals()["EndSessionResponse"]
UpscaleChunkRequest: Any = globals()["UpscaleChunkRequest"]
UpscaleChunkResponse: Any = globals()["UpscaleChunkResponse"]


@dataclass
class _Session:
    """Per-video pipeline state for unary uplift calls."""

    key: tuple[int, int, Scale, float]
    """Pipeline-pool key for this session."""

    cache: FlashVSRPipelineCache
    """Autoregressive model state for this video."""

    token: str
    """Unforgeable credential required by unary session calls."""

    next_chunk_index: int = 0
    """Next accepted chunk index."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    """Lock preventing concurrent updates to the autoregressive cache."""


@dataclass(frozen=True)
class _RunResult:
    """Encoded model output and response metadata for one chunk."""

    frames_rgb: bytes
    """Raw RGB output, or empty bytes for display-only requests."""

    num_frames: int
    """Number of frames in the output chunk."""

    height: int
    """Output frame height."""

    width: int
    """Output frame width."""

    elapsed_ms: float
    """Model and device-synchronization latency in milliseconds."""


def _resolve_scale(scale: int) -> Scale:
    if scale == 2:
        return 2
    if scale == 4:
        return 4
    raise ValueError(f"FlashVSR scale must be 2 or 4, got {scale}")


def _resolve_attention_mode(mode: RequestedAttentionMode) -> AttentionMode:
    if mode == "full":
        return "full"
    if mode == "auto" and importlib.util.find_spec("triton") is None:
        logging.warning("Triton is unavailable; using full attention")
        return "full"
    return "sparse"


class FlashVSR:
    """Serve the historical FlashVSR uplift protocol without app registration."""

    def __init__(
        self,
        *,
        default_height: int,
        default_width: int,
        default_scale: int,
        default_sparse_ratio: float,
        attention_mode: AttentionMode,
        compile_network: bool,
        use_cuda_graph: bool,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        self._default_height = default_height
        self._default_width = default_width
        self._default_scale = default_scale
        self._default_sparse_ratio = default_sparse_ratio
        self._attention_mode = attention_mode
        self._compile_network = compile_network
        self._use_cuda_graph = use_cuda_graph
        self._dtype = dtype
        self._device = device
        self._pipelines: dict[tuple[int, int, Scale, float], FlashVSRPipeline] = {}
        self._pipeline_lock = threading.Lock()
        self._sessions: dict[str, _Session] = {}
        self._sessions_lock = threading.Lock()
        # ponytail: A global lock serializes one GPU; shard by device and CUDA
        # stream if concurrent inference becomes a requirement.
        self._gpu_lock = threading.Lock()

    def warm_up(self) -> None:
        """Load the default pipeline before accepting requests."""
        self._get_pipeline(
            self._default_height,
            self._default_width,
            self._default_scale,
            self._default_sparse_ratio,
        )

    def _get_pipeline(
        self, height: int, width: int, scale: int, sparse_ratio: float
    ) -> FlashVSRPipeline:
        resolved_scale = _resolve_scale(scale)
        key = (height, width, resolved_scale, float(sparse_ratio))
        with self._pipeline_lock:
            pipeline = self._pipelines.get(key)
            if pipeline is None:
                logging.info(
                    "Loading FlashVSR for %sx%s, scale=%s, sparse_ratio=%s, "
                    "attention=%s",
                    height,
                    width,
                    resolved_scale,
                    sparse_ratio,
                    self._attention_mode,
                )
                config = build_flashvsr_v1_1(
                    input_H=height,
                    input_W=width,
                    scale=resolved_scale,
                    sparse_ratio=sparse_ratio,
                    attention_mode=self._attention_mode,
                    compile_network=self._compile_network,
                    use_cuda_graph=self._use_cuda_graph,
                    dtype=self._dtype,
                )
                pipeline = config.setup().to(device=self._device).eval()
                self._pipelines[key] = pipeline
            return pipeline

    def _request_frames(self, request: Any, height: int, width: int) -> np.ndarray:
        if request.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if request.height != height or request.width != width:
            raise ValueError(
                f"chunk dimensions {request.height}x{request.width} do not match "
                f"session dimensions {height}x{width}"
            )
        if request.frame_encoding == FRAME_ENCODING_JPEG:
            if len(request.frames_jpeg) != request.num_frames:
                raise ValueError(
                    f"expected {request.num_frames} JPEG frames, got "
                    f"{len(request.frames_jpeg)}"
                )
            try:
                from PIL import Image
            except ImportError as exc:
                raise RuntimeError(
                    "JPEG requests require Pillow; install it or send RAW_RGB"
                ) from exc
            frames = [
                np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
                for data in request.frames_jpeg
            ]
            array = np.stack(frames)
            if array.shape[1:3] != (height, width):
                raise ValueError(
                    f"JPEG frames have shape {array.shape[1:3]}, expected "
                    f"{(height, width)}"
                )
            return np.ascontiguousarray(array)

        expected_bytes = request.num_frames * height * width * 3
        if len(request.frames_rgb) != expected_bytes:
            raise ValueError(
                f"RAW_RGB payload has {len(request.frames_rgb)} bytes; "
                f"expected {expected_bytes}"
            )
        return (
            np.frombuffer(request.frames_rgb, dtype=np.uint8)
            .reshape(request.num_frames, height, width, 3)
            .copy()
        )

    def _run_chunk(
        self,
        pipeline: FlashVSRPipeline,
        cache: FlashVSRPipelineCache,
        request: Any,
        *,
        height: int,
        width: int,
    ) -> _RunResult:
        frames = self._request_frames(request, height, width)
        video = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)
        video = (video.to(device=self._device, dtype=torch.float32) / 127.5 - 1.0).to(
            dtype=self._dtype
        )

        started = time.perf_counter()
        with self._gpu_lock, torch.inference_mode():
            output = pipeline.generate(request.chunk_index, cache, video)
            pipeline.finalize(request.chunk_index, cache)
            if output.is_cuda:
                torch.cuda.synchronize(output.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        num_frames, output_height, output_width = output.shape[2:]
        frames_rgb = b""
        if not request.display_only:
            output = ((output.float() + 1.0) * 127.5).clamp_(0, 255).to(torch.uint8)
            array = output[0].permute(1, 2, 3, 0).cpu().contiguous().numpy()
            frames_rgb = array.tobytes()
        return _RunResult(
            frames_rgb=frames_rgb,
            num_frames=int(num_frames),
            height=int(output_height),
            width=int(output_width),
            elapsed_ms=elapsed_ms,
        )

    def get_status(self, request: Any, context: grpc.ServicerContext) -> Any:
        """Return server readiness and active unary sessions."""
        del request, context
        with self._sessions_lock:
            active_sessions = list(self._sessions)
        return StatusResponse(
            ready=True,
            device=self._device,
            model_name=f"FlashVSR-v1.1/{self._attention_mode}",
            active_sessions=active_sessions,
        )

    def start_session(self, request: Any, context: grpc.ServicerContext) -> Any:
        """Allocate pipeline state for unary chunk calls."""
        del context
        session_id = request.session_id or str(uuid.uuid4())
        height = request.input_height or self._default_height
        width = request.input_width or self._default_width
        scale = request.scale or self._default_scale
        sparse_ratio = request.sparse_ratio or self._default_sparse_ratio
        token = secrets.token_urlsafe(32)
        with self._sessions_lock:
            if session_id in self._sessions:
                return StartSessionResponse(
                    session_id=session_id,
                    success=False,
                    error=f"session {session_id!r} already exists",
                )
        try:
            pipeline = self._get_pipeline(height, width, scale, sparse_ratio)
            session = _Session(
                key=(height, width, _resolve_scale(scale), float(sparse_ratio)),
                cache=pipeline.initialize_cache(),
                token=token,
            )
            with self._sessions_lock:
                if session_id in self._sessions:
                    return StartSessionResponse(
                        session_id=session_id,
                        success=False,
                        error=f"session {session_id!r} already exists",
                    )
                self._sessions[session_id] = session
            return StartSessionResponse(
                session_id=session_id, success=True, session_token=token
            )
        except Exception as exc:
            logging.exception("Could not start uplift session %s", session_id)
            return StartSessionResponse(success=False, error=str(exc))

    def end_session(self, request: Any, context: grpc.ServicerContext) -> Any:
        """Release one unary uplift session."""
        with self._sessions_lock:
            session = self._sessions.get(request.session_id)
            if session is None:
                return EndSessionResponse(success=True)
            if not secrets.compare_digest(request.session_token, session.token):
                context.set_code(grpc.StatusCode.PERMISSION_DENIED)
                context.set_details("invalid session token")
                return EndSessionResponse(success=False)
            del self._sessions[request.session_id]
        return EndSessionResponse(success=True)

    def upscale_chunk(self, request: Any, context: grpc.ServicerContext) -> Any:
        """Upscale one chunk in an existing unary session."""
        with self._sessions_lock:
            session = self._sessions.get(request.session_id)
        if session is None:
            return self._error_response(request, "unknown session")
        if not secrets.compare_digest(request.session_token, session.token):
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
            context.set_details("invalid session token")
            return self._error_response(request, "invalid session token")
        height, width, scale, sparse_ratio = session.key
        pipeline = self._get_pipeline(height, width, scale, sparse_ratio)
        try:
            with session.lock:
                if request.chunk_index != session.next_chunk_index:
                    raise ValueError(
                        f"expected chunk_index {session.next_chunk_index}, got "
                        f"{request.chunk_index}"
                    )
                result = self._run_chunk(
                    pipeline,
                    session.cache,
                    request,
                    height=height,
                    width=width,
                )
                session.next_chunk_index += 1
            return self._result_response(request, result)
        except Exception as exc:
            logging.exception("Could not upscale unary chunk")
            return self._error_response(request, str(exc))

    def upscale_video(
        self, request_iterator: Any, context: grpc.ServicerContext
    ) -> Any:
        """Upscale a bidirectional stream with cache state scoped to the RPC."""
        del context
        session_id = str(uuid.uuid4())
        pipeline: FlashVSRPipeline | None = None
        cache: FlashVSRPipelineCache | None = None
        height = width = 0
        expected_chunk_index = 0
        for request in request_iterator:
            try:
                if pipeline is None:
                    session_id = request.session_id or session_id
                    height = (
                        request.input_height or request.height or self._default_height
                    )
                    width = request.input_width or request.width or self._default_width
                    scale = request.scale or self._default_scale
                    sparse_ratio = request.sparse_ratio or self._default_sparse_ratio
                    pipeline = self._get_pipeline(height, width, scale, sparse_ratio)
                    cache = pipeline.initialize_cache()
                if request.chunk_index != expected_chunk_index:
                    raise ValueError(
                        f"expected chunk_index {expected_chunk_index}, got "
                        f"{request.chunk_index}"
                    )
                assert cache is not None
                result = self._run_chunk(
                    pipeline, cache, request, height=height, width=width
                )
                expected_chunk_index += 1
                response = self._result_response(request, result)
                response.session_id = session_id
                yield response
            except Exception as exc:
                logging.exception("Could not upscale streaming chunk")
                yield self._error_response(request, str(exc), session_id=session_id)
                return

    @staticmethod
    def _result_response(request: Any, result: _RunResult) -> Any:
        return UpscaleChunkResponse(
            session_id=request.session_id,
            frames_rgb=result.frames_rgb,
            num_frames=result.num_frames,
            height=result.height,
            width=result.width,
            chunk_index=request.chunk_index,
            elapsed_ms=result.elapsed_ms,
            frames_omitted=request.display_only,
        )

    @staticmethod
    def _error_response(
        request: Any, error: str, *, session_id: str | None = None
    ) -> Any:
        return UpscaleChunkResponse(
            session_id=session_id or request.session_id,
            chunk_index=request.chunk_index,
            error=error,
        )


def _add_flash_vsr_servicer_to_server(servicer: FlashVSR, server: grpc.Server) -> None:
    handlers = {
        "get_status": grpc.unary_unary_rpc_method_handler(
            servicer.get_status,
            request_deserializer=StatusRequest.FromString,
            response_serializer=StatusResponse.SerializeToString,
        ),
        "start_session": grpc.unary_unary_rpc_method_handler(
            servicer.start_session,
            request_deserializer=StartSessionRequest.FromString,
            response_serializer=StartSessionResponse.SerializeToString,
        ),
        "end_session": grpc.unary_unary_rpc_method_handler(
            servicer.end_session,
            request_deserializer=EndSessionRequest.FromString,
            response_serializer=EndSessionResponse.SerializeToString,
        ),
        "upscale_chunk": grpc.unary_unary_rpc_method_handler(
            servicer.upscale_chunk,
            request_deserializer=UpscaleChunkRequest.FromString,
            response_serializer=UpscaleChunkResponse.SerializeToString,
        ),
        "upscale_video": grpc.stream_stream_rpc_method_handler(
            servicer.upscale_video,
            request_deserializer=UpscaleChunkRequest.FromString,
            response_serializer=UpscaleChunkResponse.SerializeToString,
        ),
    }
    server.add_generic_rpc_handlers(
        (grpc.method_handlers_generic_handler("flashvsr.FlashVSR", handlers),)
    )


def serve(
    *,
    host: str = DEFAULT_BIND_HOST,
    port: int = DEFAULT_PORT,
    default_height: int = DEFAULT_INPUT_HEIGHT,
    default_width: int = DEFAULT_INPUT_WIDTH,
    default_scale: int = DEFAULT_SCALE,
    default_sparse_ratio: float = DEFAULT_SPARSE_RATIO,
    attention_mode: RequestedAttentionMode = "auto",
    compile_network: bool = False,
    use_cuda_graph: bool = False,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> None:
    """Run the standalone uplift server until interrupted."""
    if use_cuda_graph and not compile_network:
        compile_network = True
    service = FlashVSR(
        default_height=default_height,
        default_width=default_width,
        default_scale=default_scale,
        default_sparse_ratio=default_sparse_ratio,
        attention_mode=_resolve_attention_mode(attention_mode),
        compile_network=compile_network,
        use_cuda_graph=use_cuda_graph,
        dtype=dtype,
        device=device,
    )
    service.warm_up()
    message_bytes = DEFAULT_MAX_MESSAGE_MB * 1024 * 1024
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=8),
        options=(
            ("grpc.max_receive_message_length", message_bytes),
            ("grpc.max_send_message_length", message_bytes),
        ),
    )
    _add_flash_vsr_servicer_to_server(service, server)
    address = f"{host}:{port}"
    if server.add_insecure_port(address) == 0:
        raise RuntimeError(f"could not bind gRPC server to {address}")
    server.start()
    logging.info("Standalone FlashVSR uplift server listening on %s", address)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logging.info("Stopping standalone FlashVSR uplift server")
        server.stop(grace=5).wait()


def main() -> None:
    """Parse command-line arguments and run the server."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=DEFAULT_BIND_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--input-height", type=int, default=DEFAULT_INPUT_HEIGHT)
    parser.add_argument("--input-width", type=int, default=DEFAULT_INPUT_WIDTH)
    parser.add_argument("--scale", type=int, choices=(2, 4), default=DEFAULT_SCALE)
    parser.add_argument("--sparse-ratio", type=float, default=DEFAULT_SPARSE_RATIO)
    parser.add_argument(
        "--attention-mode", choices=("auto", "sparse", "full"), default="auto"
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    serve(
        host=args.host,
        port=args.port,
        default_height=args.input_height,
        default_width=args.input_width,
        default_scale=args.scale,
        default_sparse_ratio=args.sparse_ratio,
        attention_mode=args.attention_mode,
        compile_network=args.compile,
        use_cuda_graph=args.cuda_graph,
        dtype=getattr(torch, args.dtype),
        device=args.device,
    )


if __name__ == "__main__":
    main()
