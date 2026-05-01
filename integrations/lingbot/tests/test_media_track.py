from __future__ import annotations

from fractions import Fraction

import pytest
from av.packet import Packet

from lingbot.webrtc.media import (
    EncodedVideoPacket,
    LingbotVideoTrack,
)


@pytest.mark.asyncio
async def test_video_track_recv_returns_packet_for_encoded_input() -> None:
    track = LingbotVideoTrack(fps=16)
    payload = b"\x00\x00\x00\x01\x65\x88\x84"
    try:
        await track.enqueue_encoded_packets([EncodedVideoPacket(payload=payload)])
        packet = await track.recv()
        assert isinstance(packet, Packet)
        assert bytes(packet) == payload
        assert packet.pts == 0
        assert packet.time_base == Fraction(1, 16)
    finally:
        await track.close()

@pytest.mark.asyncio
async def test_video_track_drops_oldest_when_queue_full() -> None:
    track = LingbotVideoTrack(fps=16, queue_max_size=1)
    try:
        await track.enqueue_encoded_packets(
            [
                EncodedVideoPacket(payload=b"old"),
                EncodedVideoPacket(payload=b"new"),
            ]
        )
        assert track.dropped_units == 1
        packet = await track.recv()
        assert isinstance(packet, Packet)
        assert bytes(packet) == b"new"
    finally:
        await track.close()
