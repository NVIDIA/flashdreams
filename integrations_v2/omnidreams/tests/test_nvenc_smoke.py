# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU smoke test for :class:`PyNvHardwareEncoder`.

Requires a host with NVENC-capable hardware and ``PyNvVideoCodec``
installed. Skips cleanly on any other host so the ``ci_gpu`` job runs
elsewhere aren't confused by a hard failure.

What we assert (kept small on purpose):

- The capability probe reports supported.
- Encoding a modest CUDA chunk produces at least one packet.
- The first emitted packet is an Annex-B keyframe carrying SPS + PPS +
  IDR NAL units. SPS/PPS presence verifies ``repeatspspps=1`` reached
  the hardware — without it, receivers cannot lock on.
- Packet ``pts`` / ``time_base`` are set to the RTP 90 kHz clock, which
  is what aiortc's ``H264Encoder.pack()`` expects.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
import torch

from flashdreams.runtime import StepResult

pytestmark = pytest.mark.ci_gpu

# ``PyNvVideoCodec`` probes for the NVIDIA driver library at import time
# and raises ``RuntimeError`` when it is absent (e.g. CPU CI runners with
# the package installed but no driver). ``pytestmark`` gates test
# selection, not import, so guard collection here — otherwise
# ``pytest -m ci_cpu`` aborts before the marker filter runs.
try:
    from flashdreams.serving.webrtc.encoders import (  # noqa: E402
        _pynvvideocodec_installed,
    )
    from flashdreams.serving.webrtc.nvenc import (  # noqa: E402
        PyNvHardwareEncoder,
        _payload_contains_nal_type,
    )
except (ImportError, RuntimeError) as exc:
    pytest.skip(
        f"NVENC imports unavailable ({type(exc).__name__}: {exc})",
        allow_module_level=True,
    )

_H264_NAL_TYPE_IDR = 5
_H264_NAL_TYPE_SPS = 7
_H264_NAL_TYPE_PPS = 8


def _has_annex_b_start_code(payload: bytes) -> bool:
    return payload.startswith(b"\x00\x00\x00\x01") or payload.startswith(
        b"\x00\x00\x01"
    )


@pytest.fixture(scope="module")
def nvenc_available() -> None:
    if not _pynvvideocodec_installed():
        pytest.skip("PyNvVideoCodec is not installed on this host")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available on this host")
    supported, reason = PyNvHardwareEncoder.is_supported(
        gpu_id=0,
        width=512,
        height=288,
    )
    if not supported:
        pytest.skip(f"NVENC H.264 not supported on this host: {reason}")


def test_probe_reports_supported(nvenc_available) -> None:
    supported, reason = PyNvHardwareEncoder.is_supported(
        gpu_id=0,
        width=512,
        height=288,
    )
    assert supported, reason


def test_encode_chunk_produces_annex_b_keyframe_with_sps_pps(
    nvenc_available,
) -> None:
    encoder = PyNvHardwareEncoder(
        width=512,
        height=288,
        fps=30,
        bitrate=2_000_000,
        gpu_id=0,
        gop=15,
    )
    try:
        # 4-frame CUDA chunk in the omnidreams runtime's output layout.
        chunk = torch.randint(
            0,
            255,
            (4, 3, 288, 512),
            dtype=torch.uint8,
            device="cuda",
        )
        packets: list = []
        num_frames, num_keyframes, encode_ms = encoder.encode_chunk_sync(
            StepResult.from_video_chunk(
                step_index=0,
                video_chunk=chunk,
                layout="tchw",
            ),
            force_keyframe=True,
            on_packet=packets.append,
        )
        assert num_frames == 4
        assert num_keyframes >= 1
        assert encode_ms > 0.0
        assert len(packets) >= 1

        first = bytes(packets[0])
        assert _has_annex_b_start_code(first), (
            "first packet does not start with an Annex-B start code"
        )
        assert _payload_contains_nal_type(first, _H264_NAL_TYPE_IDR), (
            "first packet does not carry an IDR NAL (nal_type=5) despite "
            "force_keyframe=True"
        )
        assert _payload_contains_nal_type(first, _H264_NAL_TYPE_SPS), (
            "SPS NAL missing from keyframe packet — repeatspspps=1 not "
            "reaching the hardware; receivers will fail to lock on"
        )
        assert _payload_contains_nal_type(first, _H264_NAL_TYPE_PPS), (
            "PPS NAL missing from keyframe packet — repeatspspps=1 not "
            "reaching the hardware; receivers will fail to lock on"
        )

        # PTS uses the RTP 90 kHz clock so aiortc's H264Encoder.pack()
        # rescales cleanly. First packet's pts must be 0.
        assert packets[0].pts == 0
        assert packets[0].time_base == Fraction(1, 90_000)
        # Second packet, if produced, must be exactly 90_000/fps ticks later.
        if len(packets) >= 2:
            assert packets[1].pts == 90_000 // 30
    finally:
        encoder.close()
