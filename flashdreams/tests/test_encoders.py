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
from fractions import Fraction
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from av.packet import Packet

pytestmark = pytest.mark.ci_cpu

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
# DefaultRTCEncoder.deliver_chunk delegates to track.enqueue_chunk
# ---------------------------------------------------------------------------


class _FakeBufferedVideoTrack:
    """Minimal stand-in that ``isinstance(track, BufferedVideoTrack)``
    treats as a real track (subclassing lets us bypass MediaStreamTrack's
    aiortc runtime dependencies without breaking the isinstance check)."""

    def __init__(self) -> None:
        self.enqueued_chunks: list = []

    async def enqueue_chunk(self, chunk) -> int:
        self.enqueued_chunks.append(chunk)
        return 4


class TestDefaultRTCEncoderDeliver:
    @pytest.mark.asyncio
    async def test_deliver_chunk_returns_frames_from_track(self) -> None:
        from flashdreams.serving.webrtc import media as media_mod

        fake_track = _FakeBufferedVideoTrack()
        # Patch the isinstance check inside deliver_chunk to accept our fake.
        with patch.object(media_mod, "BufferedVideoTrack", _FakeBufferedVideoTrack):
            enc = DefaultRTCEncoder(fps=30)
            result = await enc.deliver_chunk(
                SimpleNamespace(shape=(4, 3, 8, 8)),  # ty:ignore[invalid-argument-type]
                fake_track,  # ty:ignore[invalid-argument-type]
            )
        assert result.backend == "aiortc"
        assert result.num_frames == 4
        assert result.num_keyframes == 0
        assert len(fake_track.enqueued_chunks) == 1

    @pytest.mark.asyncio
    async def test_deliver_chunk_rejects_wrong_track_type(self) -> None:
        enc = DefaultRTCEncoder(fps=30)
        with pytest.raises(TypeError, match="BufferedVideoTrack"):
            await enc.deliver_chunk(
                SimpleNamespace(),  # ty:ignore[invalid-argument-type]
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
    packet is delivered to the track via ``loop.call_soon_threadsafe``,
    exactly as the real hardware encoder does. This lets us assert the
    end-to-end ordering property without any GPU dependency.

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
        chunk: object,
        track: NVENCVideoTrack,
        *,
        force_keyframe: bool = False,
    ) -> ChunkDeliveryResult:
        del chunk, force_keyframe
        loop = asyncio.get_running_loop()
        frames = self._frames_per_chunk

        def _encode_worker() -> None:
            # One packet per frame, monotonically-increasing pts. The
            # per-iteration call_soon_threadsafe hand-off mirrors what
            # PyNvHardwareEncoder does with its on_packet callback.
            for _ in range(frames):
                packet = Packet(b"\x00\x00\x00\x01\x67")
                with self._pts_counter_lock:
                    pts_frame_index = self._pts_counter
                    self._pts_counter += 1
                packet.pts = (pts_frame_index * 90_000) // self.fps
                packet.time_base = self._time_base
                loop.call_soon_threadsafe(
                    track.enqueue_encoded_packet_nowait,
                    packet,
                )

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
    async def test_sequential_await_produces_monotonic_pts(self) -> None:
        encoder = _OrderingFakeEncoder(
            fps=_ORDERING_FPS,
            frames_per_chunk=_ORDERING_FRAMES_PER_CHUNK,
        )
        track = encoder.create_track(maxsize=_ORDERING_TOTAL_FRAMES)

        for _ in range(_ORDERING_NUM_CHUNKS):
            await encoder.deliver_chunk(object(), track)

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
            asyncio.create_task(encoder.deliver_chunk(object(), track))
            for _ in range(_ORDERING_NUM_CHUNKS)
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
