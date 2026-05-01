from __future__ import annotations

import torch

from lingbot.webrtc.media import (
    PyNvVideoCodecH264ChunkEncoder,
    tensor_chunk_to_abgr_cuda_frames,
)


def _make_cuda_video_chunk(
    *,
    frames: int,
    width: int,
    height: int,
) -> torch.Tensor:
    torch.manual_seed(0)
    return (torch.rand((1, 1, frames, 3, height, width), device="cuda") * 2.0) - 1.0


def test_tensor_chunk_to_abgr_cuda_frames_layout_and_values() -> None:
    chunk = torch.tensor(
        [[[[[[1.0]], [[0.0]], [[-1.0]]]]]],
        dtype=torch.float32,
        device="cuda",
    )

    frames = tensor_chunk_to_abgr_cuda_frames(chunk)
    assert len(frames) == 1
    frame = frames[0]
    assert frame.is_cuda
    assert frame.dtype == torch.uint8
    assert tuple(frame.shape) == (1, 1, 4)
    # ABGR for source RGB=(255, 128, 0)
    assert frame[0, 0].tolist() == [255, 0, 128, 255]


def test_pynvvideocodec_h264_chunk_encoder_emits_packets() -> None:
    encoder = PyNvVideoCodecH264ChunkEncoder(
        width=512,
        height=288,
        fps=16,
        bitrate=4_000_000,
        gpu_id=0,
    )
    try:
        produced_packets = 0
        last_result = None
        for step in range(4):
            result = encoder.encode_chunk(
                _make_cuda_video_chunk(frames=8, width=512, height=288),
                force_keyframe=(step == 0),
            )
            last_result = result
            produced_packets += len(result.packets)
            if produced_packets > 0:
                break

        assert last_result is not None
        assert last_result.backend == "pynvvideocodec"
        assert last_result.num_input_frames == 8
        assert last_result.encode_ms >= 0
        assert produced_packets > 0
        assert all(packet.payload for packet in last_result.packets)
    finally:
        encoder.close()
