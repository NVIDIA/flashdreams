# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the shared PyNvVideoCodec encoder and ABGR conversion (Phase 1).

Encoder tests require a CUDA GPU with NVENC support and PyNvVideoCodec installed.
ABGR conversion tests require only CUDA.
"""

from __future__ import annotations

from fractions import Fraction

import pytest
import torch

from flashdreams.serving.webrtc.encode import (
    ChunkEncodingResult,
    EncodedVideoPacket,
    PyNvVideoCodecH264ChunkEncoder,
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
# PyNvVideoCodecH264ChunkEncoder
# ---------------------------------------------------------------------------


class TestPyNvVideoCodecEncoder:
    def test_encoder_produces_packets(self) -> None:
        encoder = PyNvVideoCodecH264ChunkEncoder(
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
        encoder = PyNvVideoCodecH264ChunkEncoder(
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
        encoder = PyNvVideoCodecH264ChunkEncoder(
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

    @pytest.mark.parametrize("num_encoders", [1, 2, 4])
    def test_sustained_encode_throughput(self, num_encoders: int) -> None:
        """Encode 600s of video at native resolution to measure sustained throughput.

        Parametrized over 1/2/4 parallel encoder instances to saturate
        all NVENC engines on the GPU.  Monitor utilization with:
            nvidia-smi dmon -s u --gpm-metrics 30,31,166,167,168,169
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        width, height, fps = 1280, 704, 30
        frames_per_chunk = 8
        duration_s = 600
        total_chunks = (duration_s * fps + frames_per_chunk - 1) // frames_per_chunk
        chunks_per_encoder = (total_chunks + num_encoders - 1) // num_encoders

        encoders = [
            PyNvVideoCodecH264ChunkEncoder(
                width=width, height=height, fps=fps, bitrate=10_000_000, gpu_id=0,
            )
            for _ in range(num_encoders)
        ]

        def _encode_worker(encoder_idx: int) -> tuple[int, int, int]:
            enc = encoders[encoder_idx]
            packets = 0
            nbytes = 0
            keyframes = 0
            for i in range(chunks_per_encoder):
                chunk = _make_uint8_chunk(
                    frames=frames_per_chunk, width=width, height=height,
                    seed=encoder_idx * chunks_per_encoder + i,
                )
                result = enc.encode_chunk(chunk, force_keyframe=(i % 30 == 0))
                packets += len(result.packets)
                nbytes += sum(len(p.payload) for p in result.packets)
                keyframes += result.num_keyframes
            return packets, nbytes, keyframes

        try:
            wall_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=num_encoders) as pool:
                futures = [
                    pool.submit(_encode_worker, idx)
                    for idx in range(num_encoders)
                ]
                total_packets = 0
                total_bytes = 0
                total_keyframes = 0
                for fut in as_completed(futures):
                    packets, nbytes, keyframes = fut.result()
                    total_packets += packets
                    total_bytes += nbytes
                    total_keyframes += keyframes
            wall_s = time.perf_counter() - wall_start

            total_frames = chunks_per_encoder * num_encoders * frames_per_chunk
            encode_fps = total_frames / wall_s
            print(
                f"\nSustained encode ({num_encoders} encoder(s)): "
                f"{total_frames} frames "
                f"({width}x{height} @ {fps}fps, {duration_s}s) "
                f"in {wall_s:.2f}s → {encode_fps:.1f} fps, "
                f"{total_bytes / 1024:.0f} KiB, "
                f"{total_keyframes} keyframes"
            )
            assert total_packets == total_frames
            assert total_bytes > 0
            assert encode_fps > fps, (
                f"Encoder too slow for real-time: {encode_fps:.1f} fps < {fps} fps"
            )
        finally:
            for enc in encoders:
                enc.close()

    def test_encoder_invalid_params(self) -> None:
        with pytest.raises(ValueError, match="width and height"):
            PyNvVideoCodecH264ChunkEncoder(
                width=0, height=288, fps=30, bitrate=4_000_000, gpu_id=0,
            )
        with pytest.raises(ValueError, match="fps"):
            PyNvVideoCodecH264ChunkEncoder(
                width=512, height=288, fps=0, bitrate=4_000_000, gpu_id=0,
            )
        with pytest.raises(ValueError, match="bitrate"):
            PyNvVideoCodecH264ChunkEncoder(
                width=512, height=288, fps=30, bitrate=-1, gpu_id=0,
            )


# ---------------------------------------------------------------------------
# Bitstream dump: GPU (NVENC) vs CPU (PyAV/libav) H.264 output
# ---------------------------------------------------------------------------


_DUMP_WIDTH = 512
_DUMP_HEIGHT = 288
_DUMP_FPS = 30
_DUMP_BITRATE = 4_000_000
_DUMP_FRAMES_PER_CHUNK = 8
_DUMP_DURATION_S = 5
_DUMP_NUM_CHUNKS = (_DUMP_DURATION_S * _DUMP_FPS + _DUMP_FRAMES_PER_CHUNK - 1) // _DUMP_FRAMES_PER_CHUNK


def _generate_gradient_chunks(
    num_chunks: int, frames_per_chunk: int, width: int, height: int,
) -> list[torch.Tensor]:
    """Generate deterministic gradient video chunks on CUDA.

    Each chunk is [1, 1, T, 3, H, W] uint8 with a shifting horizontal
    gradient so frames are visually distinct.
    """
    chunks = []
    for chunk_idx in range(num_chunks):
        frames = []
        for f in range(frames_per_chunk):
            t = (chunk_idx * frames_per_chunk + f) / (num_chunks * frames_per_chunk)
            r = torch.linspace(t, 1.0, width, device="cuda").unsqueeze(0).expand(height, -1)
            g = torch.linspace(0.0, t, height, device="cuda").unsqueeze(1).expand(-1, width)
            b = torch.full((height, width), 0.5, device="cuda")
            frame = torch.stack([r, g, b], dim=0)  # [3, H, W]
            frames.append(frame)
        chunk = torch.stack(frames, dim=0).unsqueeze(0).unsqueeze(0)  # [1,1,T,3,H,W]
        chunks.append((chunk * 255).to(torch.uint8))
    return chunks


@pytest.mark.ci_gpu
class TestBitstreamDump:
    """Dump raw .h264 elementary streams from GPU and CPU encoders.

    Run with:
        LOGURU_LEVEL=DEBUG uv run --package flashdreams-omnidreams \\
            pytest integrations/omnidreams/tests/test_pynvvideocodec_encode.py \\
            -v -s -m manual -k TestBitstreamDump

    Output files are written to a temp directory printed in the test output.
    Inspect with:
        ffprobe -show_frames <file>.h264
        ffplay <file>.h264
    """

    def test_dump_gpu_and_cpu_bitstreams(self, tmp_path: pytest.TempPathFactory) -> None:
        import av
        import numpy as np

        chunks = _generate_gradient_chunks(
            _DUMP_NUM_CHUNKS, _DUMP_FRAMES_PER_CHUNK, _DUMP_WIDTH, _DUMP_HEIGHT,
        )

        # --- GPU encode (PyNvVideoCodec / NVENC) ---
        gpu_path = tmp_path / "gpu_nvenc.h264"
        encoder = PyNvVideoCodecH264ChunkEncoder(
            width=_DUMP_WIDTH, height=_DUMP_HEIGHT,
            fps=_DUMP_FPS, bitrate=_DUMP_BITRATE, gpu_id=0,
        )
        gpu_total_bytes = 0
        try:
            with open(gpu_path, "wb") as f:
                for i, chunk in enumerate(chunks):
                    result = encoder.encode_chunk(chunk, force_keyframe=(i == 0))
                    for pkt in result.packets:
                        f.write(pkt.payload)
                        gpu_total_bytes += len(pkt.payload)
        finally:
            encoder.close()

        # --- CPU encode (PyAV / libav software H.264) ---
        cpu_path = tmp_path / "cpu_libx264.h264"
        cpu_total_bytes = 0
        codec = av.CodecContext.create("libx264", "w")
        codec.width = _DUMP_WIDTH
        codec.height = _DUMP_HEIGHT
        codec.pix_fmt = "yuv420p"
        codec.time_base = Fraction(1, _DUMP_FPS)
        codec.bit_rate = _DUMP_BITRATE
        codec.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "bframes": "0",
        }
        codec.open()
        try:
            with open(cpu_path, "wb") as f:
                pts = 0
                for chunk in chunks:
                    # [1,1,T,3,H,W] uint8 CUDA → [T,H,W,3] uint8 numpy
                    frames_np = chunk[0, 0].permute(0, 2, 3, 1).cpu().numpy()
                    for frame_np in frames_np:
                        av_frame = av.VideoFrame.from_ndarray(
                            np.ascontiguousarray(frame_np), format="rgb24",
                        )
                        av_frame.pts = pts
                        pts += 1
                        for pkt in codec.encode(av_frame):
                            f.write(bytes(pkt))
                            cpu_total_bytes += len(bytes(pkt))
                # Flush
                for pkt in codec.encode(None):
                    f.write(bytes(pkt))
                    cpu_total_bytes += len(bytes(pkt))
        finally:
            del codec

        # --- Report ---
        total_frames = _DUMP_NUM_CHUNKS * _DUMP_FRAMES_PER_CHUNK
        duration_s = total_frames / _DUMP_FPS
        print(f"\n{'='*60}")
        print(f"Bitstream dump: {_DUMP_WIDTH}x{_DUMP_HEIGHT} @ {_DUMP_FPS}fps, "
              f"{total_frames} frames ({duration_s:.1f}s), {_DUMP_BITRATE/1e6:.1f} Mbps target")
        print(f"  GPU (NVENC):   {gpu_path}  ({gpu_total_bytes:,} bytes)")
        print(f"  CPU (libx264): {cpu_path}  ({cpu_total_bytes:,} bytes)")
        print(f"Inspect with:")
        print(f"  ffprobe -show_frames {gpu_path}")
        print(f"  ffplay {gpu_path}")
        print(f"{'='*60}")

        assert gpu_path.stat().st_size > 0, "GPU bitstream is empty"
        assert cpu_path.stat().st_size > 0, "CPU bitstream is empty"
