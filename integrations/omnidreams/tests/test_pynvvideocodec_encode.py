# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared PyNvVideoCodec encoder and ABGR conversion (Phase 1).

Encoder tests require a CUDA GPU with NVENC support and PyNvVideoCodec installed.
ABGR conversion tests require only CUDA.
"""

from __future__ import annotations


import pytest
import torch

from flashdreams.serving.webrtc.encode import (
    ChunkEncodingResult,
    EncodedVideoPacket,
    PyNvHardwareEncoder,
    tensor_chunk_to_abgr_cuda_frames,
)

pytestmark = pytest.mark.ci_gpu


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_float_chunk(
    *, frames: int, width: int, height: int, seed: int = 0
) -> torch.Tensor:
    """Create a [1, 1, T, 3, H, W] float tensor in [-1, 1] on CUDA."""
    torch.manual_seed(seed)
    return (torch.rand((1, 1, frames, 3, height, width), device="cuda") * 2.0) - 1.0


def _make_uint8_chunk(
    *, frames: int, width: int, height: int, seed: int = 0
) -> torch.Tensor:
    """Create a [1, 1, T, 3, H, W] uint8 tensor in [0, 255] on CUDA."""
    torch.manual_seed(seed)
    return torch.randint(0, 256, (1, 1, frames, 3, height, width), device="cuda", dtype=torch.uint8)


# ---------------------------------------------------------------------------
# tensor_chunk_to_abgr_cuda_frames — uint8 path
# ---------------------------------------------------------------------------


class TestAbgrConversionUint8:
    def test_output_shape_and_dtype(self) -> None:
        chunk = _make_uint8_chunk(frames=4, width=64, height=48)
        frames = tensor_chunk_to_abgr_cuda_frames(chunk)
        assert len(frames) == 4
        for frame in frames:
            assert frame.is_cuda
            assert frame.dtype == torch.uint8
            assert tuple(frame.shape) == (48, 64, 4)

    def test_channel_ordering(self) -> None:
        """Verify ABGR surface format: memory bytes are R, G, B, A (little-endian)."""
        # Create a 1x1 pixel: R=100, G=150, B=200
        pixel = torch.tensor(
            [[[[[[100]], [[150]], [[200]]]]]],
            dtype=torch.uint8,
            device="cuda",
        )
        frames = tensor_chunk_to_abgr_cuda_frames(pixel)
        assert len(frames) == 1
        abgr = frames[0][0, 0].tolist()
        assert abgr == [100, 150, 200, 255]  # R, G, B, A

    def test_omnidreams_resolution(self) -> None:
        """Verify conversion works at OmniDreams native resolution."""
        chunk = _make_uint8_chunk(frames=8, width=1280, height=704)
        frames = tensor_chunk_to_abgr_cuda_frames(chunk)
        assert len(frames) == 8
        assert tuple(frames[0].shape) == (704, 1280, 4)


# ---------------------------------------------------------------------------
# tensor_chunk_to_abgr_cuda_frames — float path
# ---------------------------------------------------------------------------


class TestAbgrConversionFloat:
    def test_output_shape_and_dtype(self) -> None:
        chunk = _make_float_chunk(frames=4, width=64, height=48)
        frames = tensor_chunk_to_abgr_cuda_frames(chunk)
        assert len(frames) == 4
        for frame in frames:
            assert frame.is_cuda
            assert frame.dtype == torch.uint8
            assert tuple(frame.shape) == (48, 64, 4)

    def test_channel_ordering(self) -> None:
        """Float [-1,1] → ABGR surface format: verify known pixel values."""
        # R=1.0→255, G=0.0→128, B=-1.0→0
        pixel = torch.tensor(
            [[[[[[1.0]], [[0.0]], [[-1.0]]]]]],
            dtype=torch.float32,
            device="cuda",
        )
        frames = tensor_chunk_to_abgr_cuda_frames(pixel)
        assert len(frames) == 1
        abgr = frames[0][0, 0].tolist()
        # R=255 (from 1.0), G=128 (from 0.0), B=0 (from -1.0), A=255
        assert abgr == [255, 128, 0, 255]


# ---------------------------------------------------------------------------
# Dtype equivalence: uint8 and float paths produce identical output
# ---------------------------------------------------------------------------


class TestAbgrDtypeEquivalence:
    def test_equivalent_values_produce_identical_abgr(self) -> None:
        """A uint8 tensor and its float [-1,1] equivalent must yield the same ABGR."""
        uint8_chunk = torch.tensor(
            [[[[[[0]], [[128]], [[255]]]]]],
            dtype=torch.uint8,
            device="cuda",
        )
        # Equivalent float values: 0→-1.0, 128→~0.004, 255→1.0
        # Use the exact inverse of the quantization formula: float = uint8/127.5 - 1.0
        float_chunk = uint8_chunk.to(torch.float32) / 127.5 - 1.0

        uint8_frames = tensor_chunk_to_abgr_cuda_frames(uint8_chunk)
        float_frames = tensor_chunk_to_abgr_cuda_frames(float_chunk)

        assert len(uint8_frames) == len(float_frames)
        for u8_frame, f_frame in zip(uint8_frames, float_frames):
            assert torch.equal(u8_frame, f_frame)


# ---------------------------------------------------------------------------
# tensor_chunk_to_abgr_cuda_frames — error cases
# ---------------------------------------------------------------------------


class TestAbgrConversionErrors:
    def test_wrong_ndim(self) -> None:
        with pytest.raises(ValueError, match="6 dimensions"):
            tensor_chunk_to_abgr_cuda_frames(torch.zeros(3, 3, device="cuda"))

    def test_cpu_tensor_rejected(self) -> None:
        with pytest.raises(ValueError, match="CUDA"):
            tensor_chunk_to_abgr_cuda_frames(torch.zeros(1, 1, 1, 3, 4, 4))

    def test_empty_batch_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            tensor_chunk_to_abgr_cuda_frames(torch.zeros(0, 1, 1, 3, 4, 4, device="cuda"))


# ---------------------------------------------------------------------------
# PyNvHardwareEncoder
# ---------------------------------------------------------------------------


class TestPyNvVideoCodecEncoder:
    def test_encoder_produces_packets(self) -> None:
        encoder = PyNvHardwareEncoder(
            width=512, height=288, fps=30, bitrate=4_000_000, gpu_id=0,
        )
        try:
            produced_packets = 0
            last_result = None
            for step in range(4):
                result = encoder.encode_chunk(
                    _make_float_chunk(frames=8, width=512, height=288),
                    force_keyframe=(step == 0),
                )
                last_result = result
                produced_packets += len(result.packets)
                if produced_packets > 0:
                    break

            assert last_result is not None
            assert isinstance(last_result, ChunkEncodingResult)
            assert last_result.backend == "pynvvideocodec"
            assert last_result.num_input_frames == 8
            assert last_result.encode_ms >= 0
            assert produced_packets > 0
            assert all(isinstance(p, EncodedVideoPacket) for p in last_result.packets)
            assert all(p.payload for p in last_result.packets)
        finally:
            encoder.close()

    def test_encoder_accepts_uint8_input(self) -> None:
        encoder = PyNvHardwareEncoder(
            width=512, height=288, fps=30, bitrate=4_000_000, gpu_id=0,
        )
        try:
            produced_packets = 0
            for step in range(4):
                result = encoder.encode_chunk(
                    _make_uint8_chunk(frames=8, width=512, height=288),
                    force_keyframe=(step == 0),
                )
                produced_packets += len(result.packets)
                if produced_packets > 0:
                    break
            assert produced_packets > 0
        finally:
            encoder.close()

    def test_keyframe_detected_in_output(self) -> None:
        """When force_keyframe=True, at least one emitted packet contains an IDR NAL."""
        encoder = PyNvHardwareEncoder(
            width=512, height=288, fps=30, bitrate=4_000_000, gpu_id=0,
        )
        try:
            all_packets: list[EncodedVideoPacket] = []
            for step in range(4):
                result = encoder.encode_chunk(
                    _make_float_chunk(frames=4, width=512, height=288),
                    force_keyframe=(step == 0),
                )
                all_packets.extend(result.packets)
                if any(p.keyframe for p in all_packets):
                    break

            keyframe_packets = [p for p in all_packets if p.keyframe]
            assert len(keyframe_packets) >= 1
            # Verify the keyframe packet actually contains an IDR NAL (type 5)
            for p in keyframe_packets:
                assert b"\x00\x00\x01" in p.payload
        finally:
            encoder.close()

    def test_encoder_invalid_params(self) -> None:
        with pytest.raises(ValueError, match="width and height"):
            PyNvHardwareEncoder(
                width=0, height=288, fps=30, bitrate=4_000_000, gpu_id=0,
            )
        with pytest.raises(ValueError, match="fps"):
            PyNvHardwareEncoder(
                width=512, height=288, fps=0, bitrate=4_000_000, gpu_id=0,
            )
        with pytest.raises(ValueError, match="bitrate"):
            PyNvHardwareEncoder(
                width=512, height=288, fps=30, bitrate=-1, gpu_id=0,
            )
