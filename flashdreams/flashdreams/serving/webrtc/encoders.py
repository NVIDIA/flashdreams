# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Video encoder backends for the WebRTC serving path.

Thread-affine WebRTC runtimes call :func:`select_encoder` during shared runtime
initialization. Runtimes that do not opt in pick up :class:`DefaultRTCEncoder`
transparently through :meth:`BaseWebRTCSessionManager._resolve_video_encoder`.

**This module deliberately does not import** ``PyNvVideoCodec``. The
hardware encoder lives in a sibling module (:mod:`nvenc`) that
:func:`select_encoder` imports only when a hardware backend is about to
be constructed. Callers that only ever end up on the software path (any
integration that has not opted in to hardware encoding, tests forcing
``backend="default"``) never trigger the PyNvVideoCodec import and never
pay its process-global side effects.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

import torch
from aiortc import MediaStreamTrack
from loguru import logger

from flashdreams.runtime import StepResult

if TYPE_CHECKING:
    from flashdreams.serving.webrtc.media import BufferedVideoTrack, NVENCVideoTrack


EncoderBackend = Literal["auto", "nvenc", "default"]


class EncoderInitError(RuntimeError):
    """Raised when a forced encoder backend cannot be initialized."""


@dataclass(slots=True, frozen=True)
class ChunkDeliveryResult:
    """Uniform result shape for :meth:`VideoEncoder.deliver_chunk`.

    Callers do not need to know which encoder produced the chunk; the
    fields here cover both paths.
    """

    backend: str
    num_frames: int
    num_keyframes: int
    encode_ms: float


@runtime_checkable
class VideoEncoder(Protocol):
    """Encoder backend paired with a compatible :class:`MediaStreamTrack`.

    Each backend owns two responsibilities: creating a fresh media track
    sized for one session (:meth:`create_track`), and encoding + enqueueing
    one chunk of frames onto that track (:meth:`deliver_chunk`). Callers
    pick one backend at startup and branch nowhere else.
    """

    fps: int
    backend: str
    prefers_codec: str | None

    def create_track(self, *, maxsize: int) -> BufferedVideoTrack | NVENCVideoTrack: ...

    def prepare_chunk_payload(
        self,
        result: StepResult,
        track: MediaStreamTrack,
    ) -> object: ...

    async def deliver_prepared_chunk(
        self,
        payload: object,
        track: MediaStreamTrack,
        *,
        force_keyframe: bool = False,
    ) -> ChunkDeliveryResult: ...

    async def deliver_chunk(
        self,
        result: StepResult,
        track: MediaStreamTrack,
        *,
        force_keyframe: bool = False,
    ) -> ChunkDeliveryResult: ...

    def close(self) -> None: ...


class DefaultRTCEncoder:
    """Software encoder that piggybacks on aiortc's built-in H.264 / VP8
    encoding via :class:`~flashdreams.serving.webrtc.media.BufferedVideoTrack`.

    Runtime-lightweight — no hardware handle to hold, no NVENC session to
    allocate. The base manager falls back to this class whenever the
    runtime does not expose ``video_encoder`` or when the
    ``select_encoder`` factory decides NVENC is not usable.
    """

    prefers_codec: str | None = None
    backend = "aiortc"

    def __init__(self, *, fps: int) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be > 0, got {fps}")
        self.fps = fps
        logger.info(
            "Video encoder ready: backend={} fps={}",
            self.backend,
            fps,
        )

    def create_track(self, *, maxsize: int) -> BufferedVideoTrack:
        # Lazy import — avoids ``encoders`` ↔ ``media`` cycle at module
        # load time.
        from flashdreams.serving.webrtc.media import BufferedVideoTrack

        return BufferedVideoTrack(fps=self.fps, maxsize=maxsize)

    def prepare_chunk_payload(
        self,
        result: StepResult,
        track: MediaStreamTrack,
    ) -> tuple[object, ...]:
        from flashdreams.serving.webrtc.media import BufferedVideoTrack

        if not isinstance(track, BufferedVideoTrack):
            raise TypeError(
                "DefaultRTCEncoder requires a BufferedVideoTrack; got "
                f"{type(track).__name__}. Create it via encoder.create_track()."
            )
        return track.prepare_result_frames(result)

    async def deliver_prepared_chunk(
        self,
        payload: object,
        track: MediaStreamTrack,
        *,
        force_keyframe: bool = False,
    ) -> ChunkDeliveryResult:
        # aiortc's software encoder decides its own keyframe cadence and
        # responds to receiver PLI/FIR feedback, so we don't need to
        # forward the flag.
        del force_keyframe
        from flashdreams.serving.webrtc.media import BufferedVideoTrack

        if not isinstance(track, BufferedVideoTrack):
            raise TypeError(
                "DefaultRTCEncoder requires a BufferedVideoTrack; got "
                f"{type(track).__name__}. Create it via encoder.create_track()."
            )
        if not isinstance(payload, tuple):
            raise TypeError("DefaultRTCEncoder payload must be a tuple of RGB frames.")
        enqueued = await track.enqueue_frames(cast(Any, payload))
        return ChunkDeliveryResult(
            backend=self.backend,
            num_frames=enqueued,
            num_keyframes=0,
            encode_ms=0.0,
        )

    async def deliver_chunk(
        self,
        result: StepResult,
        track: MediaStreamTrack,
        *,
        force_keyframe: bool = False,
    ) -> ChunkDeliveryResult:
        return await self.deliver_prepared_chunk(
            self.prepare_chunk_payload(result, track),
            track,
            force_keyframe=force_keyframe,
        )

    def close(self) -> None:
        return


def _pynvvideocodec_installed() -> bool:
    """Fast, side-effect-free check that the ``PyNvVideoCodec`` package is
    present on ``sys.path``.

    ``importlib.util.find_spec`` walks the import machinery to locate the
    package but does not execute its ``__init__.py``, so this probe has no
    side effects on the calling process's CUDA / shared-library state.

    **Necessary but not sufficient.** A ``True`` return means the Python
    package is discoverable, not that it can actually be loaded — the
    library's ``__init__.py`` performs its own driver-library probe
    (``libnvidia-encode.so.1`` via ``dlopen``) and raises ``RuntimeError``
    when that library is absent, which is the common shape on hosts that
    have the ``PyNvVideoCodec`` wheel installed without the NVIDIA driver
    (CPU CI runners, some devcontainers). The definitive gate is therefore
    :func:`select_encoder`'s deferred import of
    :mod:`flashdreams.serving.webrtc.nvenc`, which catches both
    ``ImportError`` and ``RuntimeError`` from that load.

    Kept as a module-scope function (rather than inlined) so tests can
    monkey-patch it to force the library-missing branch without needing
    to touch the actual on-disk install.
    """
    return importlib.util.find_spec("PyNvVideoCodec") is not None


def select_encoder(
    *,
    backend: EncoderBackend,
    width: int,
    height: int,
    fps: int,
    bitrate: int,
    gpu_id: int,
    gop: int = 30,
) -> VideoEncoder:
    """Choose an encoder implementation.

    ``backend == "default"`` returns a :class:`DefaultRTCEncoder` and does
    not touch the NVENC probe path at all — importantly, it also does not
    import :mod:`flashdreams.serving.webrtc.nvenc`, so ``PyNvVideoCodec``
    is never loaded on this call.

    ``backend == "nvenc"`` requires the hardware path; any probe failure
    raises :class:`EncoderInitError` so misconfigured deployments surface
    at startup rather than silently degrading.

    ``backend == "auto"`` prefers hardware when the environment supports
    it. Selection uses two stages with two different failure semantics:

    * **Stage 1** (package probe + deferred import + ``GetEncoderCaps`` +
      resolution bounds): a "no" here means the environment cannot do
      this, which is expected on hosts without NVENC. Fall back silently
      to :class:`DefaultRTCEncoder`, logging the reason at INFO. The
      deferred import can raise ``RuntimeError`` when the package is
      installed but its native driver library is absent (common on CPU
      CI hosts); that is Stage-1-shaped and is treated the same way.
    * **Stage 2** (``CreateEncoder``): a raise here means Stage 1 said
      the environment supports NVENC but session allocation failed anyway
      — driver bug, session-pool exhaustion, hardware fault, or
      misconfiguration. Log with traceback and re-raise; do not silently
      degrade, because doing so would hide a real problem.

    PyNvVideoCodec may replace the CUDA context current on the calling
    thread, including when ``GetEncoderCaps`` fails. Preserve an already
    initialized PyTorch device context so subsequent model work does not
    inherit the encoder probe's context.
    """
    if backend == "default":
        return DefaultRTCEncoder(fps=fps)

    restore_device = (
        torch.cuda.current_device() if torch.cuda.is_initialized() else None
    )
    try:
        return _select_hardware_encoder(
            backend=backend,
            width=width,
            height=height,
            fps=fps,
            bitrate=bitrate,
            gpu_id=gpu_id,
            gop=gop,
        )
    finally:
        if restore_device is not None:
            torch.cuda.set_device(restore_device)


def _select_hardware_encoder(
    *,
    backend: Literal["auto", "nvenc"],
    width: int,
    height: int,
    fps: int,
    bitrate: int,
    gpu_id: int,
    gop: int,
) -> VideoEncoder:
    if not _pynvvideocodec_installed():
        reason = "PyNvVideoCodec library is not installed"
        if backend == "nvenc":
            raise EncoderInitError(
                f"encoder_backend='nvenc' requested but NVENC H.264 is not "
                f"supported on this system: {reason}"
            )
        logger.info(
            "NVENC H.264 not supported on this system ({}); using default "
            "aiortc backend for video encoder.",
            reason,
        )
        return DefaultRTCEncoder(fps=fps)

    # Deferred import — the process only loads ``PyNvVideoCodec`` at this
    # point, when a hardware backend is actually about to be constructed.
    # The package's ``__init__.py`` performs its own driver-library probe
    # and raises ``RuntimeError`` when ``libnvidia-encode.so.1`` is absent
    # (installed-but-driverless hosts); treat that Stage-1-shaped failure
    # the same way as ``find_spec`` returning False.
    try:
        from flashdreams.serving.webrtc.nvenc import PyNvHardwareEncoder
    except (ImportError, RuntimeError) as exc:
        reason = f"PyNvVideoCodec not loadable: {type(exc).__name__}: {exc}"
        if backend == "nvenc":
            raise EncoderInitError(
                f"encoder_backend='nvenc' requested but NVENC H.264 is not "
                f"supported on this system: {reason}"
            ) from exc
        logger.info(
            "NVENC H.264 not supported on this system ({}); using default "
            "aiortc backend for video encoder.",
            reason,
        )
        return DefaultRTCEncoder(fps=fps)

    supported, reason = PyNvHardwareEncoder.is_supported(
        gpu_id=gpu_id,
        width=width,
        height=height,
    )
    if not supported:
        if backend == "nvenc":
            raise EncoderInitError(
                f"encoder_backend='nvenc' requested but NVENC H.264 is not "
                f"supported on this system: {reason}"
            )
        logger.info(
            "NVENC H.264 not supported on this system ({}); using default "
            "aiortc backend for video encoder.",
            reason,
        )
        return DefaultRTCEncoder(fps=fps)

    try:
        return PyNvHardwareEncoder(
            width=width,
            height=height,
            fps=fps,
            bitrate=bitrate,
            gpu_id=gpu_id,
            gop=gop,
        )
    except Exception:
        logger.exception(
            "GetEncoderCaps reported NVENC H.264 supported, but CreateEncoder "
            "failed. Not silently falling back to the software encoder — the "
            "underlying failure needs to be diagnosed.",
        )
        raise
