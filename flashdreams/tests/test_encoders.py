# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``VideoEncoder`` abstraction and ``select_encoder`` factory.

Covers:

- Protocol conformance for the software-path adapter.
- ``select_encoder`` branch coverage under ``backend={"default","auto","nvenc"}``.
- The two failure semantics of the capability probe: a Stage-1 no is a
  silent fallback; a Stage-2 ``CreateEncoder`` raise is a hard error
  (never a silent fallback). The latter is the regression guard that
  keeps a future refactor from masking driver problems.
- The cross-chunk packet-ordering invariant: sequential ``await
  deliver_chunk(...)`` calls must produce monotonically-ordered packets
  on the paired track. A ``create_task``-style refactor would silently
  break this — see :class:`TestDeliverChunkOrdering`.
- Compatibility guards for the two upstream libraries we couple to
  (``aiortc`` and ``PyNvVideoCodec``).

The tests never import ``PyNvVideoCodec`` for real: :func:`_install_fake_nvc`
injects a stand-in module into ``sys.modules`` so ``nvenc``'s top-level
``import PyNvVideoCodec as nvc`` binds to the mock. That mirrors the split
in production — any process that has not opted in to hardware encoding
never loads the real library.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import Callable, Sequence
from fractions import Fraction
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from av.packet import Packet

pytestmark = pytest.mark.ci_cpu

from flashdreams.runtime import StepResult
from flashdreams.serving.webrtc import encoders as enc_mod
from flashdreams.serving.webrtc.encoders import (
    ChunkDeliveryResult,
    DefaultRTCEncoder,
    EncoderInitError,
    VideoEncoder,
    select_encoder,
)
from flashdreams.serving.webrtc.media import NVENCVideoTrack

_SELECT_KW = dict(
    width=1280,
    height=704,
    fps=30,
    bitrate=6_000_000,
    gpu_id=0,
    gop=30,
)


def _install_fake_nvc(monkeypatch: pytest.MonkeyPatch, fake_nvc: MagicMock):
    """Make ``nvenc`` importable with a fake ``PyNvVideoCodec`` and return
    the ``nvenc`` module with its ``nvc`` symbol pointing at the fake.

    Injecting the fake into ``sys.modules`` first ensures that if the
    ``nvenc`` module has not been imported yet in this test session, its
    top-level ``import PyNvVideoCodec as nvc`` binds to the mock rather
    than requiring the real library. If ``nvenc`` was already imported
    (a prior test), we still overwrite the ``nvc`` attribute directly so
    every test sees the same fake.
    """
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", fake_nvc)
    monkeypatch.setattr(enc_mod, "_pynvvideocodec_installed", lambda: True)
    from flashdreams.serving.webrtc import nvenc as nvenc_mod

    monkeypatch.setattr(nvenc_mod, "nvc", fake_nvc)
    return nvenc_mod


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestVideoEncoderProtocol:
    def test_default_encoder_satisfies_protocol(self) -> None:
        assert isinstance(DefaultRTCEncoder(fps=30), VideoEncoder)

    def test_default_encoder_advertises_no_codec_preference(self) -> None:
        assert DefaultRTCEncoder(fps=30).prefers_codec is None

    def test_default_encoder_rejects_bad_fps(self) -> None:
        with pytest.raises(ValueError, match="fps"):
            DefaultRTCEncoder(fps=0)


# ---------------------------------------------------------------------------
# select_encoder: "default" backend never touches NVENC
# ---------------------------------------------------------------------------


class TestSelectDefaultBackend:
    def test_returns_default_encoder(self) -> None:
        enc = select_encoder(backend="default", **_SELECT_KW)
        assert isinstance(enc, DefaultRTCEncoder)
        assert enc.fps == 30

    def test_does_not_probe_or_import_nvenc(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Even when the library exists, "default" must not probe it and
        # must not import the sibling ``nvenc`` module — the whole point
        # of the module split is that a process on the software path
        # never loads ``PyNvVideoCodec``.
        probe_calls: list[bool] = []

        def _probe_spy() -> bool:
            probe_calls.append(True)
            return True

        monkeypatch.setattr(enc_mod, "_pynvvideocodec_installed", _probe_spy)
        select_encoder(backend="default", **_SELECT_KW)
        assert probe_calls == [], (
            "select_encoder(backend='default') must short-circuit before the "
            "PyNvVideoCodec availability probe."
        )

    def test_works_even_when_library_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(enc_mod, "_pynvvideocodec_installed", lambda: False)
        enc = select_encoder(backend="default", **_SELECT_KW)
        assert isinstance(enc, DefaultRTCEncoder)


# ---------------------------------------------------------------------------
# select_encoder: Stage 1 (GetEncoderCaps) failure paths
# ---------------------------------------------------------------------------


class TestSelectStage1Failure:
    """Stage 1 says "environment can't do this" — auto silently falls back;
    nvenc raises loudly."""

    def test_library_missing_auto_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(enc_mod, "_pynvvideocodec_installed", lambda: False)
        enc = select_encoder(backend="auto", **_SELECT_KW)
        assert isinstance(enc, DefaultRTCEncoder)

    def test_library_missing_nvenc_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(enc_mod, "_pynvvideocodec_installed", lambda: False)
        with pytest.raises(EncoderInitError, match="not installed"):
            select_encoder(backend="nvenc", **_SELECT_KW)

    @pytest.mark.parametrize(
        "caps_effect, expected_reason_frag",
        [
            (RuntimeError("driver comms error"), "driver comms error"),
            ({}, "no capabilities"),
            (
                {
                    "width_max": 640,
                    "height_max": 480,
                    "width_min": 32,
                    "height_min": 32,
                },
                "exceeds driver maximum",
            ),
            (
                {
                    "width_max": 8192,
                    "height_max": 8192,
                    "width_min": 1920,
                    "height_min": 1088,
                },
                "below driver minimum",
            ),
        ],
        ids=["caps_raise", "caps_empty", "caps_max_too_small", "caps_min_too_big"],
    )
    def test_caps_failure_auto_falls_back(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caps_effect,
        expected_reason_frag,
    ) -> None:
        fake_nvc = MagicMock()
        if isinstance(caps_effect, BaseException):
            fake_nvc.GetEncoderCaps.side_effect = caps_effect
        else:
            fake_nvc.GetEncoderCaps.return_value = caps_effect
        _install_fake_nvc(monkeypatch, fake_nvc)
        enc = select_encoder(backend="auto", **_SELECT_KW)
        assert isinstance(enc, DefaultRTCEncoder)
        # ``CreateEncoder`` must not be reached when Stage 1 fails.
        fake_nvc.CreateEncoder.assert_not_called()

    def test_caps_failure_nvenc_raises_with_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_nvc = MagicMock()
        fake_nvc.GetEncoderCaps.side_effect = RuntimeError("driver comms error")
        _install_fake_nvc(monkeypatch, fake_nvc)
        with pytest.raises(EncoderInitError, match="driver comms error"):
            select_encoder(backend="nvenc", **_SELECT_KW)

    def test_caps_probe_restores_initialized_torch_cuda_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_nvc = MagicMock()
        fake_nvc.GetEncoderCaps.side_effect = RuntimeError("unsupported GPU")
        _install_fake_nvc(monkeypatch, fake_nvc)
        set_device = MagicMock()
        monkeypatch.setattr(enc_mod.torch.cuda, "is_initialized", lambda: True)
        monkeypatch.setattr(enc_mod.torch.cuda, "current_device", lambda: 3)
        monkeypatch.setattr(enc_mod.torch.cuda, "set_device", set_device)

        enc = select_encoder(backend="auto", **_SELECT_KW)

        assert isinstance(enc, DefaultRTCEncoder)
        set_device.assert_called_once_with(3)


# ---------------------------------------------------------------------------
# select_encoder: Stage-1 deferred-import failure (package present but
# not loadable — e.g. driverless host)
# ---------------------------------------------------------------------------


class _RaisingNvencModule(ModuleType):
    """Stand-in for ``flashdreams.serving.webrtc.nvenc`` whose attribute
    access raises. Mirrors the shape ``PyNvVideoCodec`` induces on hosts
    where the wheel is installed but the NVIDIA driver library is
    missing: ``find_spec`` returns non-None, but importing symbols from
    ``nvenc`` fires the package's driver probe and raises."""

    def __init__(self, name: str, exc: BaseException) -> None:
        super().__init__(name)
        self._exc = exc

    def __getattr__(self, name: str):  # noqa: ANN204 - stub, raises unconditionally
        raise self._exc


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("simulated: libnvidia-encode.so.1 missing"),
        ImportError("simulated: cannot import PyNvHardwareEncoder"),
    ],
    ids=["runtime_error_driverless_host", "import_error"],
)
class TestSelectStage1DeferredImportFailure:
    """Cover the ``find_spec`` positive / import negative gap.

    ``_pynvvideocodec_installed`` returns True when the Python package is
    on ``sys.path``, but importing from ``nvenc`` may still raise if the
    package's own driver probe fails (``RuntimeError`` on driverless
    hosts) or if the module cannot resolve the target symbol
    (``ImportError``). Both must be treated as Stage-1 failures — silent
    fallback on ``auto``, hard error on ``nvenc`` — otherwise a
    driverless production host running ``backend="auto"`` crashes session
    init instead of degrading to software.
    """

    def _install_raising_nvenc(
        self, monkeypatch: pytest.MonkeyPatch, exc: BaseException
    ) -> None:
        monkeypatch.setattr(enc_mod, "_pynvvideocodec_installed", lambda: True)
        monkeypatch.setitem(
            sys.modules,
            "flashdreams.serving.webrtc.nvenc",
            _RaisingNvencModule("flashdreams.serving.webrtc.nvenc", exc),
        )

    def test_auto_falls_back(
        self, monkeypatch: pytest.MonkeyPatch, exc: BaseException
    ) -> None:
        self._install_raising_nvenc(monkeypatch, exc)
        enc = select_encoder(backend="auto", **_SELECT_KW)
        assert isinstance(enc, DefaultRTCEncoder)

    def test_nvenc_raises_with_reason(
        self, monkeypatch: pytest.MonkeyPatch, exc: BaseException
    ) -> None:
        self._install_raising_nvenc(monkeypatch, exc)
        with pytest.raises(EncoderInitError, match=type(exc).__name__):
            select_encoder(backend="nvenc", **_SELECT_KW)


# ---------------------------------------------------------------------------
# select_encoder: Stage 2 (CreateEncoder) failure — HARD ERROR
# ---------------------------------------------------------------------------


class TestSelectStage2HardError:
    """The key regression guard: if Stage 1 says supported but Stage 2's
    ``CreateEncoder`` raises, ``select_encoder`` must re-raise. Silently
    falling back to the software encoder here would mask a real bug
    (driver, session pool exhaustion, hardware fault). Any future refactor
    that makes this test fail is almost certainly regressing that semantic.
    """

    def _fake_nvc_caps_ok(self) -> MagicMock:
        fake = MagicMock()
        fake.GetEncoderCaps.return_value = {
            "width_max": 8192,
            "height_max": 8192,
            "width_min": 32,
            "height_min": 32,
        }
        fake.FORCEIDR = 0x1
        return fake

    def test_construct_failure_reraises_under_auto(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_nvc = self._fake_nvc_caps_ok()
        fake_nvc.CreateEncoder.side_effect = RuntimeError(
            "NVENC session pool exhausted"
        )
        _install_fake_nvc(monkeypatch, fake_nvc)
        with pytest.raises(RuntimeError, match="session pool exhausted"):
            select_encoder(backend="auto", **_SELECT_KW)

    def test_construct_failure_reraises_under_nvenc(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_nvc = self._fake_nvc_caps_ok()
        fake_nvc.CreateEncoder.side_effect = RuntimeError("hardware fault")
        _install_fake_nvc(monkeypatch, fake_nvc)
        with pytest.raises(RuntimeError, match="hardware fault"):
            select_encoder(backend="nvenc", **_SELECT_KW)


# ---------------------------------------------------------------------------
# ChunkDeliveryResult
# ---------------------------------------------------------------------------


class TestChunkDeliveryResult:
    def test_is_frozen_dataclass(self) -> None:
        result = ChunkDeliveryResult(
            backend="fake",
            num_frames=4,
            num_keyframes=1,
            encode_ms=1.5,
        )
        with pytest.raises((AttributeError, Exception)):
            result.backend = "other"  # ty:ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# DefaultRTCEncoder.deliver_chunk delegates to track.enqueue_result
# ---------------------------------------------------------------------------


class _FakeBufferedVideoTrack:
    """Minimal stand-in that ``isinstance(track, BufferedVideoTrack)``
    treats as a real track (subclassing lets us bypass MediaStreamTrack's
    aiortc runtime dependencies without breaking the isinstance check)."""

    def __init__(self) -> None:
        self.enqueued_results: list[StepResult] = []
        self.enqueued_frames: list[object] = []

    def prepare_result_frames(self, result: StepResult) -> tuple[object, ...]:
        self.enqueued_results.append(result)
        return tuple(object() for _ in range(result.frame_count))

    async def enqueue_frames(self, frames: Sequence[object]) -> int:
        self.enqueued_frames.extend(frames)
        return len(frames)

    async def enqueue_result(self, result: StepResult) -> int:
        return await self.enqueue_frames(self.prepare_result_frames(result))


class TestDefaultRTCEncoderDeliver:
    @pytest.mark.parametrize(
        ("layout", "shape"),
        [("tchw", (4, 3, 8, 8)), ("bvtchw", (1, 1, 4, 3, 8, 8))],
    )
    @pytest.mark.asyncio
    async def test_deliver_chunk_returns_frames_from_track(
        self, layout: str, shape: tuple[int, ...]
    ) -> None:
        from flashdreams.serving.webrtc import media as media_mod

        fake_track = _FakeBufferedVideoTrack()
        step_result = StepResult.from_video_chunk(
            step_index=0,
            video_chunk=torch.zeros(shape, dtype=torch.uint8),
            layout=layout,  # ty:ignore[invalid-argument-type]
        )
        # Patch the isinstance check inside deliver_chunk to accept our fake.
        with patch.object(media_mod, "BufferedVideoTrack", _FakeBufferedVideoTrack):
            enc = DefaultRTCEncoder(fps=30)
            result = await enc.deliver_chunk(
                step_result,
                fake_track,  # ty:ignore[invalid-argument-type]
            )
        assert result.backend == "aiortc"
        assert result.num_frames == 4
        assert result.num_keyframes == 0
        assert fake_track.enqueued_results == [step_result]
        assert len(fake_track.enqueued_frames) == 4

    @pytest.mark.parametrize(
        ("layout", "shape"),
        [("tchw", (3, 3, 2, 2)), ("bvtchw", (1, 1, 3, 3, 2, 2))],
    )
    @pytest.mark.asyncio
    async def test_software_conversion_uses_declared_layout(
        self, layout: str, shape: tuple[int, ...]
    ) -> None:
        enc = DefaultRTCEncoder(fps=30)
        track = enc.create_track(maxsize=3)
        step_result = StepResult.from_video_chunk(
            step_index=0,
            video_chunk=torch.zeros(shape, dtype=torch.uint8),
            layout=layout,  # ty:ignore[invalid-argument-type]
        )

        delivery = await enc.deliver_chunk(step_result, track)

        assert delivery.num_frames == 3
        assert track.qsize() == 3
        await track.close()

    @pytest.mark.asyncio
    async def test_software_path_prepares_host_frames_with_track(self) -> None:
        from flashdreams.serving.webrtc.media import BufferedVideoTrack

        source = torch.zeros((2, 3, 2, 2), dtype=torch.uint8)
        step_result = StepResult.from_video_chunk(
            step_index=0,
            video_chunk=source,
            layout="tchw",
        )
        seen: list[StepResult] = []

        def _converter(delivered: StepResult) -> list[np.ndarray]:
            seen.append(delivered)
            assert delivered is step_result
            assert delivered.video_chunk.data_ptr() == source.data_ptr()
            return [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(2)]

        track = BufferedVideoTrack(fps=30, maxsize=2, frame_converter=_converter)
        encoder = DefaultRTCEncoder(fps=30)
        payload = encoder.prepare_chunk_payload(step_result, track)
        delivery = await encoder.deliver_prepared_chunk(payload, track)

        assert delivery.num_frames == 2
        assert seen == [step_result]
        await track.close()

    @pytest.mark.asyncio
    async def test_deliver_chunk_rejects_wrong_track_type(self) -> None:
        enc = DefaultRTCEncoder(fps=30)
        step_result = StepResult.from_video_chunk(
            step_index=0,
            video_chunk=torch.zeros((1, 3, 2, 2), dtype=torch.uint8),
            layout="tchw",
        )
        with pytest.raises(TypeError, match="BufferedVideoTrack"):
            await enc.deliver_chunk(
                step_result,
                SimpleNamespace(),  # ty:ignore[invalid-argument-type]
            )


# ---------------------------------------------------------------------------
# Compatibility guards for upstream libraries
# ---------------------------------------------------------------------------


class TestCompatGuards:
    """These tests exist to fail loudly when an upstream dependency
    changes its public surface underneath us. They are cheap and
    catch dependency drift far earlier than a runtime failure would."""

    def test_aiortc_h264_encoder_has_pack_and_encode(self) -> None:
        from aiortc.codecs.h264 import H264Encoder

        assert callable(getattr(H264Encoder, "encode", None)), (
            "aiortc H264Encoder.encode disappeared — the software encoder "
            "contract this design assumes has changed."
        )
        assert callable(getattr(H264Encoder, "pack", None)), (
            "aiortc H264Encoder.pack disappeared — the pre-encoded packet "
            "path this design relies on has changed."
        )

    def test_aiortc_sender_module_importable(self) -> None:
        # Any structural change to rtcrtpsender that breaks import will
        # break the runtime; catch it here before the first RTP packet.
        import aiortc.rtcrtpsender  # noqa: F401


class TestNvencResultConversion:
    @pytest.mark.parametrize(
        ("layout", "shape"),
        [("tchw", (2, 3, 2, 3)), ("bvtchw", (1, 1, 2, 3, 2, 3))],
    )
    def test_conversion_uses_declared_layout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        layout: str,
        shape: tuple[int, ...],
    ) -> None:
        nvenc_mod = _install_fake_nvc(monkeypatch, MagicMock())
        video = torch.empty(shape, dtype=torch.uint8)
        channel_dim = 1 if layout == "tchw" else 3
        video.select(channel_dim, 0).fill_(10)
        video.select(channel_dim, 1).fill_(20)
        video.select(channel_dim, 2).fill_(30)
        result = StepResult.from_video_chunk(
            step_index=0,
            video_chunk=video,
            layout=layout,  # ty:ignore[invalid-argument-type]
        )

        frames = nvenc_mod._result_to_abgr_frames(result)

        assert frames.shape == (2, 2, 3, 4)
        assert torch.equal(frames[0, 0, 0], torch.tensor([10, 20, 30, 255]))

    def test_getencodercaps_callable_when_library_available(self) -> None:
        # This guard exercises the *real* PyNvVideoCodec surface via
        # ``nvenc``. ``PyNvVideoCodec`` raises ``RuntimeError`` (not
        # ``ImportError``) when the NVIDIA driver library is absent, so
        # ``importorskip`` alone wouldn't be enough — the ``nvenc``
        # import would still raise.
        try:
            from flashdreams.serving.webrtc import nvenc as nvenc_mod
        except (ImportError, RuntimeError) as exc:
            pytest.skip(f"PyNvVideoCodec not loadable: {type(exc).__name__}: {exc}")

        assert callable(getattr(nvenc_mod.nvc, "GetEncoderCaps", None)), (
            "PyNvVideoCodec.GetEncoderCaps renamed or removed — Stage 1 "
            "capability probe cannot function."
        )


# ---------------------------------------------------------------------------
# Cross-chunk packet-ordering invariant
# ---------------------------------------------------------------------------


class _OrderingFakeEncoder:
    """Minimal encoder that mimics :class:`PyNvHardwareEncoder`'s async
    marshaling contract without needing CUDA or PyNvVideoCodec.

    Encoding runs on an ``asyncio.to_thread`` worker and each emitted
    packet is delivered to the track via ``run_coroutine_threadsafe``.
    That mirrors the real hardware encoder's backpressured per-packet
    handoff without needing any GPU dependency.

    The pts counter is guarded by a lock because the fire-and-forget
    test dispatches multiple ``deliver_chunk`` calls concurrently, and
    ``self._pts_counter += 1`` is not atomic across Python bytecodes.
    """

    backend = "fake"
    prefers_codec: str | None = "h264"

    def __init__(self, *, fps: int, frames_per_chunk: int) -> None:
        self.fps = fps
        self._frames_per_chunk = frames_per_chunk
        self._pts_counter = 0
        self._pts_counter_lock = threading.Lock()
        self._time_base = Fraction(1, 90_000)

    def create_track(self, *, maxsize: int) -> NVENCVideoTrack:
        return NVENCVideoTrack(fps=self.fps, maxsize=maxsize)

    async def deliver_chunk(
        self,
        result: StepResult,
        track: NVENCVideoTrack,
        *,
        force_keyframe: bool = False,
    ) -> ChunkDeliveryResult:
        del result, force_keyframe
        loop = asyncio.get_running_loop()
        frames = self._frames_per_chunk

        def _encode_worker() -> None:
            # One packet per frame, monotonically-increasing pts. The
            # The per-iteration run_coroutine_threadsafe hand-off mirrors
            # what PyNvHardwareEncoder does with its on_packet callback.
            for _ in range(frames):
                packet = Packet(b"\x00\x00\x00\x01\x67")
                with self._pts_counter_lock:
                    pts_frame_index = self._pts_counter
                    self._pts_counter += 1
                packet.pts = (pts_frame_index * 90_000) // self.fps
                packet.time_base = self._time_base
                future = asyncio.run_coroutine_threadsafe(
                    track.enqueue_encoded_packet(packet),
                    loop,
                )
                future.result()

        await asyncio.to_thread(_encode_worker)
        return ChunkDeliveryResult(
            backend=self.backend,
            num_frames=frames,
            num_keyframes=0,
            encode_ms=0.0,
        )

    def close(self) -> None:
        return


# 30 chunks × 8 frames = 240 packets — realistic scale for an interactive
# session (about 8 seconds of 30 fps video, or a few minutes of chunked
# generation at typical omnidreams cadence). At the encoder's real fps=30
# the track's recv() pacing would make draining take ~8 s per test, so we
# use a much higher fps here to keep the pacing throttle negligible while
# preserving distinct, monotonic pts values (pts = i at fps=90_000 since
# ``(i * 90_000) // 90_000 == i``).
_ORDERING_NUM_CHUNKS = 30
_ORDERING_FRAMES_PER_CHUNK = 8
_ORDERING_TOTAL_FRAMES = _ORDERING_NUM_CHUNKS * _ORDERING_FRAMES_PER_CHUNK
_ORDERING_FPS = 90_000


class TestDeliverChunkOrdering:
    """Regression guard for the manager's sequential await pattern.

    The manager's ``_generation_worker`` does ``await deliver_chunk(...)``
    in a single loop, which forces chunk N's packets to land on the track
    before chunk N+1 starts encoding. If a future refactor swaps that for
    ``asyncio.create_task(deliver_chunk(...))`` — or introduces a
    producer-consumer queue between generation and encoding without a
    reorder buffer — chunks could complete out of order and packets
    would land on the track with non-monotonic ``pts``.

    These tests replay the manager's sequential-await pattern against a
    fake encoder that mirrors :class:`PyNvHardwareEncoder`'s async
    marshaling. They pass under the current ``await``-based design and
    would fail under a fire-and-forget rewrite.
    """

    @pytest.mark.asyncio
    async def test_hardware_deliver_streams_before_chunk_encode_returns(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """First packet should reach the track while later frames encode.

        This catches whole-chunk buffering regressions where the NVENC
        callback merely appends packets to a local list and ``deliver_chunk``
        does not enqueue anything until ``encode_chunk_sync`` has returned.
        """
        nvenc_mod = _install_fake_nvc(monkeypatch, MagicMock())
        encoder = nvenc_mod.PyNvHardwareEncoder.__new__(nvenc_mod.PyNvHardwareEncoder)
        loop = asyncio.get_running_loop()
        first_packet_enqueued = asyncio.Event()
        finish_encode = threading.Event()
        encode_returned = threading.Event()

        def _packet(pts: int) -> Packet:
            packet = Packet(b"\x00\x00\x00\x01\x67")
            packet.pts = pts
            packet.time_base = Fraction(1, 90_000)
            return packet

        def _fake_encode_chunk_sync(
            result: StepResult,
            *,
            force_keyframe: bool = False,
            on_packet: Callable[[Packet], None] | None = None,
        ) -> tuple[int, int, float]:
            del result, force_keyframe
            assert on_packet is not None
            on_packet(_packet(0))
            loop.call_soon_threadsafe(first_packet_enqueued.set)
            assert finish_encode.wait(timeout=1.0)
            on_packet(_packet(1))
            encode_returned.set()
            return (2, 0, 12.5)

        setattr(encoder, "encode_chunk_sync", _fake_encode_chunk_sync)
        track = NVENCVideoTrack(fps=_ORDERING_FPS, maxsize=4)
        deliver_task = asyncio.create_task(
            encoder.deliver_chunk(
                StepResult.from_video_chunk(
                    step_index=0,
                    video_chunk=torch.zeros((2, 3, 2, 2)),
                    layout="tchw",
                ),
                track,
            )
        )

        try:
            await asyncio.wait_for(first_packet_enqueued.wait(), timeout=1.0)
            assert not encode_returned.is_set()
            assert track.qsize() == 1
            first = await asyncio.wait_for(track.recv(), timeout=1.0)
            assert first.pts == 0
        finally:
            finish_encode.set()

        result = await asyncio.wait_for(deliver_task, timeout=1.0)
        assert result.backend == "pynvvideocodec"
        assert result.num_frames == 2
        assert result.num_keyframes == 0
        assert result.encode_ms == 12.5
        second = await asyncio.wait_for(track.recv(), timeout=1.0)
        assert second.pts == 1
        await track.close()

    @pytest.mark.asyncio
    async def test_sequential_await_produces_monotonic_pts(self) -> None:
        encoder = _OrderingFakeEncoder(
            fps=_ORDERING_FPS,
            frames_per_chunk=_ORDERING_FRAMES_PER_CHUNK,
        )
        track = encoder.create_track(maxsize=_ORDERING_TOTAL_FRAMES)

        for step_index in range(_ORDERING_NUM_CHUNKS):
            await encoder.deliver_chunk(
                StepResult.from_video_chunk(
                    step_index=step_index,
                    video_chunk=torch.zeros((_ORDERING_FRAMES_PER_CHUNK, 3, 1, 1)),
                    layout="tchw",
                ),
                track,
            )

        seen_pts: list[int] = []
        for _ in range(_ORDERING_TOTAL_FRAMES):
            packet = await asyncio.wait_for(track.recv(), timeout=1.0)
            # ``_OrderingFakeEncoder`` always sets pts before enqueueing; the
            # ``av.Packet.pts`` field is nullable at the type level, so narrow.
            assert packet.pts is not None
            seen_pts.append(int(packet.pts))

        # Strictly monotonic and matches the exact expected pts sequence.
        expected = [
            (i * 90_000) // _ORDERING_FPS for i in range(_ORDERING_TOTAL_FRAMES)
        ]
        assert seen_pts == expected, (
            "packets emitted out of order across chunks: "
            f"first 16 got={seen_pts[:16]} expected={expected[:16]}"
        )
        assert seen_pts == sorted(seen_pts)

    @pytest.mark.asyncio
    async def test_fire_and_forget_pattern_would_break_ordering(self) -> None:
        """Counter-check: if the manager ever spawns deliver_chunk via
        ``asyncio.create_task`` (fire-and-forget), packets *can* interleave
        across chunks. This test documents that failure mode so a future
        refactor that reintroduces the anti-pattern can be caught by
        comparing behaviour against the sequential-await test above.

        The test is not a bug — it is a demonstration that the sequential
        await in the manager is *load-bearing* for ordering.
        """
        encoder = _OrderingFakeEncoder(
            fps=_ORDERING_FPS,
            frames_per_chunk=_ORDERING_FRAMES_PER_CHUNK,
        )
        track = encoder.create_track(maxsize=_ORDERING_TOTAL_FRAMES)

        # Bias interleaving: create_task, then gather. Individual chunks
        # each finish quickly; scheduler order within the loop does not
        # guarantee packets arrive in the same order the tasks were spawned.
        tasks = [
            asyncio.create_task(
                encoder.deliver_chunk(
                    StepResult.from_video_chunk(
                        step_index=step_index,
                        video_chunk=torch.zeros((_ORDERING_FRAMES_PER_CHUNK, 3, 1, 1)),
                        layout="tchw",
                    ),
                    track,
                )
            )
            for step_index in range(_ORDERING_NUM_CHUNKS)
        ]
        await asyncio.gather(*tasks)

        seen_pts: list[int] = []
        for _ in range(_ORDERING_TOTAL_FRAMES):
            packet = await asyncio.wait_for(track.recv(), timeout=1.0)
            # ``_OrderingFakeEncoder`` always sets pts before enqueueing; the
            # ``av.Packet.pts`` field is nullable at the type level, so narrow.
            assert packet.pts is not None
            seen_pts.append(int(packet.pts))

        # Every pts value must be present exactly once; that part is a
        # correctness invariant of the fake encoder (guarded by the lock
        # around ``_pts_counter``). Duplicate pts here would be a bug in
        # the fake, not in the code under test.
        expected_set = {
            (i * 90_000) // _ORDERING_FPS for i in range(_ORDERING_TOTAL_FRAMES)
        }
        assert set(seen_pts) == expected_set
        assert len(seen_pts) == _ORDERING_TOTAL_FRAMES

        # The full sequence, however, is not necessarily monotonic — the
        # fire-and-forget pattern permits interleave. We do NOT assert
        # monotonicity here; on the contrary, the moment this test starts
        # asserting monotonic pts, the manager's sequential-await guard
        # has become unnecessary (and the test above becomes redundant).
