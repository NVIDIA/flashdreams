# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the EncodedPacketVideoTrack (Phase 1 shared media component).

These tests verify the pre-encoded packet track behavior without requiring
a GPU or a real WebRTC connection.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
from av.packet import Packet

from flashdreams.serving.webrtc.encode import EncodedVideoPacket
from flashdreams.serving.webrtc.media import EncodedPacketVideoTrack

pytestmark = pytest.mark.ci_cpu


# ---------------------------------------------------------------------------
# EncodedPacketVideoTrack — basic recv
# ---------------------------------------------------------------------------


class TestEncodedPacketVideoTrackRecv:
    @pytest.mark.asyncio
    async def test_recv_returns_av_packet(self) -> None:
        track = EncodedPacketVideoTrack(fps=30)
        payload = b"\x00\x00\x00\x01\x65\x88\x84"
        try:
            await track.enqueue_encoded_packets([EncodedVideoPacket(payload=payload)])
            packet = await track.recv()
            assert isinstance(packet, Packet)
            assert bytes(packet) == payload
        finally:
            await track.close()

    @pytest.mark.asyncio
    async def test_pts_increments(self) -> None:
        track = EncodedPacketVideoTrack(fps=30)
        try:
            await track.enqueue_encoded_packets([
                EncodedVideoPacket(payload=b"\x00\x00\x01\x67"),
                EncodedVideoPacket(payload=b"\x00\x00\x01\x68"),
                EncodedVideoPacket(payload=b"\x00\x00\x01\x65"),
            ])
            p0 = await track.recv()
            p1 = await track.recv()
            p2 = await track.recv()
            assert p0.pts == 0
            assert p1.pts == 1
            assert p2.pts == 2
        finally:
            await track.close()

    @pytest.mark.asyncio
    async def test_time_base_matches_fps(self) -> None:
        track = EncodedPacketVideoTrack(fps=30)
        try:
            await track.enqueue_encoded_packets([EncodedVideoPacket(payload=b"\x00")])
            packet = await track.recv()
            assert packet.time_base == Fraction(1, 30)
        finally:
            await track.close()

    @pytest.mark.asyncio
    async def test_close_signals_end(self) -> None:
        from aiortc.mediastreams import MediaStreamError

        track = EncodedPacketVideoTrack(fps=30)
        await track.close()
        with pytest.raises(MediaStreamError):
            await track.recv()


# ---------------------------------------------------------------------------
# EncodedPacketVideoTrack — bounded queue with drop-oldest
# ---------------------------------------------------------------------------


class TestEncodedPacketVideoTrackDropOldest:
    @pytest.mark.asyncio
    async def test_drops_oldest_when_full(self) -> None:
        track = EncodedPacketVideoTrack(fps=30, maxsize=1)
        try:
            await track.enqueue_encoded_packets([
                EncodedVideoPacket(payload=b"old"),
                EncodedVideoPacket(payload=b"new"),
            ])
            assert track.dropped_packets == 1
            packet = await track.recv()
            assert isinstance(packet, Packet)
            assert bytes(packet) == b"new"
        finally:
            await track.close()

    @pytest.mark.asyncio
    async def test_no_drops_when_unbounded(self) -> None:
        track = EncodedPacketVideoTrack(fps=30, maxsize=0)
        try:
            await track.enqueue_encoded_packets([
                EncodedVideoPacket(payload=b"a"),
                EncodedVideoPacket(payload=b"b"),
                EncodedVideoPacket(payload=b"c"),
            ])
            assert track.dropped_packets == 0
            assert track.qsize() == 3
        finally:
            await track.close()

    @pytest.mark.asyncio
    async def test_multiple_drops(self) -> None:
        track = EncodedPacketVideoTrack(fps=30, maxsize=1)
        try:
            await track.enqueue_encoded_packets([
                EncodedVideoPacket(payload=b"a"),
                EncodedVideoPacket(payload=b"b"),
                EncodedVideoPacket(payload=b"c"),
            ])
            assert track.dropped_packets == 2
            packet = await track.recv()
            assert bytes(packet) == b"c"
        finally:
            await track.close()


# ---------------------------------------------------------------------------
# EncodedPacketVideoTrack — properties and edge cases
# ---------------------------------------------------------------------------


class TestEncodedPacketVideoTrackProperties:
    def test_fps_property(self) -> None:
        track = EncodedPacketVideoTrack(fps=60)
        assert track.fps == 60

    def test_maxsize_property(self) -> None:
        track = EncodedPacketVideoTrack(fps=30, maxsize=100)
        assert track.maxsize == 100

    def test_invalid_fps_rejected(self) -> None:
        with pytest.raises(ValueError, match="fps"):
            EncodedPacketVideoTrack(fps=0)

    def test_negative_maxsize_rejected(self) -> None:
        with pytest.raises(ValueError, match="maxsize"):
            EncodedPacketVideoTrack(fps=30, maxsize=-1)

    @pytest.mark.asyncio
    async def test_enqueue_after_close_returns_zero(self) -> None:
        track = EncodedPacketVideoTrack(fps=30)
        await track.close()
        count = await track.enqueue_encoded_packets([
            EncodedVideoPacket(payload=b"x"),
        ])
        assert count == 0

    @pytest.mark.asyncio
    async def test_qsize_tracks_depth(self) -> None:
        track = EncodedPacketVideoTrack(fps=30)
        try:
            assert track.qsize() == 0
            await track.enqueue_encoded_packets([
                EncodedVideoPacket(payload=b"a"),
                EncodedVideoPacket(payload=b"b"),
            ])
            assert track.qsize() == 2
            await track.recv()
            assert track.qsize() == 1
        finally:
            await track.close()
