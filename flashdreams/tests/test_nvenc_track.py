# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for :class:`NVENCVideoTrack` — pre-encoded H.264 packet plumbing.

The track is a thin queue in front of aiortc's ``RTCRtpSender``. What
matters is that:

- ``recv()`` returns exactly the enqueued :class:`av.Packet` (aiortc's
  ``H264Encoder.pack()`` reads ``payload``, ``pts`` and ``time_base``
  directly), and
- overflow drops the *oldest* packet — a stale I-frame is more useful than
  falling behind wall-clock on an interactive stream.
"""

from __future__ import annotations

import asyncio
from fractions import Fraction

import pytest
from aiortc.mediastreams import MediaStreamError
from av.packet import Packet

pytestmark = pytest.mark.ci_cpu

from flashdreams.serving.webrtc.media import NVENCVideoTrack


def _mk_packet(pts: int, payload: bytes = b"\x00\x00\x00\x01\x67") -> Packet:
    pkt = Packet(payload)
    pkt.pts = pts
    pkt.time_base = Fraction(1, 90_000)
    return pkt


class TestConstructorValidation:
    def test_rejects_bad_fps(self) -> None:
        with pytest.raises(ValueError, match="fps"):
            NVENCVideoTrack(fps=0, maxsize=8)

    def test_rejects_bad_maxsize(self) -> None:
        with pytest.raises(ValueError, match="maxsize"):
            NVENCVideoTrack(fps=30, maxsize=0)

    def test_exposes_fps_and_maxsize(self) -> None:
        track = NVENCVideoTrack(fps=30, maxsize=8)
        assert track.fps == 30
        assert track.maxsize == 8
        assert track.qsize() == 0
        assert track.dropped_packets == 0


class TestEnqueue:
    def test_enqueue_returns_true_on_open_track(self) -> None:
        track = NVENCVideoTrack(fps=30, maxsize=8)
        assert track.enqueue_encoded_packet_nowait(_mk_packet(0)) is True
        assert track.qsize() == 1

    @pytest.mark.asyncio
    async def test_enqueue_returns_false_on_closed_track(self) -> None:
        track = NVENCVideoTrack(fps=30, maxsize=8)
        await track.close()
        assert track.enqueue_encoded_packet_nowait(_mk_packet(0)) is False


class TestRecv:
    @pytest.mark.asyncio
    async def test_recv_returns_enqueued_packet_unchanged(self) -> None:
        # aiortc's H264Encoder.pack() consumes bytes(packet), packet.pts
        # and packet.time_base directly, so recv() must not mutate them.
        track = NVENCVideoTrack(fps=30, maxsize=8)
        original = _mk_packet(pts=1234, payload=b"\x00\x00\x00\x01\x25\xaa")
        track.enqueue_encoded_packet_nowait(original)
        got = await asyncio.wait_for(track.recv(), timeout=1.0)
        assert bytes(got) == b"\x00\x00\x00\x01\x25\xaa"
        assert got.pts == 1234
        assert got.time_base == Fraction(1, 90_000)

    @pytest.mark.asyncio
    async def test_recv_on_closed_track_raises_mediastreamerror(self) -> None:
        track = NVENCVideoTrack(fps=30, maxsize=8)
        await track.close()
        with pytest.raises(MediaStreamError):
            await asyncio.wait_for(track.recv(), timeout=1.0)


class TestOverflow:
    """When the queue fills up, drop the oldest packet, not the newest.

    Real-time streams tolerate a stale I-frame worse than they tolerate
    stale motion, and aiortc will re-request a keyframe via PLI/FIR if
    the oldest dropped packet was a keyframe.
    """

    def test_full_queue_drops_oldest(self) -> None:
        track = NVENCVideoTrack(fps=30, maxsize=2)
        p0 = _mk_packet(pts=0)
        p1 = _mk_packet(pts=1)
        p2 = _mk_packet(pts=2)
        assert track.enqueue_encoded_packet_nowait(p0) is True
        assert track.enqueue_encoded_packet_nowait(p1) is True
        # This must drop p0, not refuse p2.
        assert track.enqueue_encoded_packet_nowait(p2) is True
        assert track.qsize() == 2
        assert track.dropped_packets == 1

    @pytest.mark.asyncio
    async def test_overflow_preserves_newest_packet(self) -> None:
        track = NVENCVideoTrack(fps=30, maxsize=2)
        track.enqueue_encoded_packet_nowait(_mk_packet(pts=0))
        track.enqueue_encoded_packet_nowait(_mk_packet(pts=1))
        track.enqueue_encoded_packet_nowait(_mk_packet(pts=2))
        # Queue should now hold pts=1 and pts=2 (p0 dropped).
        first = await asyncio.wait_for(track.recv(), timeout=1.0)
        second = await asyncio.wait_for(track.recv(), timeout=1.0)
        assert first.pts == 1
        assert second.pts == 2
