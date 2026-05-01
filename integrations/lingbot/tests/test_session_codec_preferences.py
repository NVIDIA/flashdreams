from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from lingbot.webrtc.session import LingbotWebRTCSessionManager


@dataclass(slots=True)
class _FakeCodec:
    mimeType: str


class _FakeTransceiver:
    def __init__(self) -> None:
        self.preferences: list[_FakeCodec] | None = None

    def setCodecPreferences(self, codecs: list[_FakeCodec]) -> None:
        self.preferences = codecs


class _FakeSenderWithH264:
    @staticmethod
    def getCapabilities(kind: str) -> SimpleNamespace:
        assert kind == "video"
        return SimpleNamespace(
            codecs=[
                _FakeCodec("video/VP8"),
                _FakeCodec("video/H264"),
                _FakeCodec("video/H264"),
                _FakeCodec("video/rtx"),
            ]
        )


class _FakeSenderWithoutH264:
    @staticmethod
    def getCapabilities(kind: str) -> SimpleNamespace:
        assert kind == "video"
        return SimpleNamespace(codecs=[_FakeCodec("video/VP8"), _FakeCodec("video/rtx")])


def test_prefer_h264_video_codec_sets_only_h264_plus_rtx() -> None:
    transceiver = _FakeTransceiver()
    LingbotWebRTCSessionManager._prefer_h264_video_codec(
        transceiver=transceiver,
        rtp_sender_cls=_FakeSenderWithH264,
    )
    assert transceiver.preferences is not None
    assert [codec.mimeType for codec in transceiver.preferences] == [
        "video/H264",
        "video/H264",
        "video/rtx",
    ]


def test_prefer_h264_video_codec_keeps_defaults_when_unavailable() -> None:
    transceiver = _FakeTransceiver()
    LingbotWebRTCSessionManager._prefer_h264_video_codec(
        transceiver=transceiver,
        rtp_sender_cls=_FakeSenderWithoutH264,
    )
    assert transceiver.preferences is None
