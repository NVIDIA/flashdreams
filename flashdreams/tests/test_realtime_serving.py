# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading

import numpy as np
import pytest
import torch

from flashdreams.serving.realtime.frame_bus import LatestFrameBus
from flashdreams.serving.realtime.input import (
    ImageRequest,
    KeyboardState,
    PromptRequest,
    ResetRequest,
    SparseInputSnapshot,
)
from flashdreams.serving.realtime.media import (
    encode_rgb_frame_to_jpeg,
    rgb_array_to_uint8_frames,
    tensor_chunk_to_rgb_frames,
)

pytestmark = pytest.mark.ci_cpu


def test_keyboard_state_builds_sparse_snapshot_with_effective_keys() -> None:
    state = KeyboardState()

    assert state.apply_event(event="keydown", key="ArrowLeft")
    assert state.apply_event(event="keydown", key="d")

    snapshot = state.sparse_snapshot(timestamp_s=12.5)

    assert snapshot == SparseInputSnapshot(
        timestamp_s=12.5,
        pressed_keys=frozenset({"a", "d"}),
        effective_keys=frozenset({"d"}),
    )


def test_input_request_containers_are_transport_neutral() -> None:
    snapshot = SparseInputSnapshot(
        timestamp_s=1.0,
        reset=ResetRequest(reason="new_session", request_id="reset-1"),
        prompt=PromptRequest(prompt="drive forward", request_id="prompt-1"),
        image=ImageRequest(
            data=b"image-bytes",
            content_type="image/jpeg",
            request_id="image-1",
        ),
    )

    assert snapshot.reset is not None
    assert snapshot.reset.reason == "new_session"
    assert snapshot.prompt is not None
    assert snapshot.prompt.prompt == "drive forward"
    assert snapshot.image is not None
    assert snapshot.image.content_type == "image/jpeg"


def test_latest_frame_bus_publishes_single_slot_frames() -> None:
    bus = LatestFrameBus[bytes]()

    assert bus.latest() is None
    first_count = bus.publish(b"first")
    second_count = bus.publish(b"second")

    assert first_count == 1
    assert second_count == 2
    latest = bus.latest()
    assert latest is not None
    assert latest.payload == b"second"
    assert latest.count == 2
    waited = bus.wait_for_frame(last_seen_count=1, timeout_s=0.01)
    assert waited is not None
    assert waited.payload == b"second"
    assert waited.count == 2


def test_latest_frame_bus_close_wakes_waiters() -> None:
    bus = LatestFrameBus[bytes]()
    waiter_blocking = threading.Event()
    results: list[object] = []
    original_wait = bus._condition.wait

    def wait_after_ready(timeout: float | None = None) -> bool:
        waiter_blocking.set()
        return original_wait(timeout=timeout)

    setattr(bus._condition, "wait", wait_after_ready)

    def wait_for_frame() -> None:
        results.append(bus.wait_for_frame(last_seen_count=0, timeout_s=5.0))

    thread = threading.Thread(target=wait_for_frame)
    thread.start()
    assert waiter_blocking.wait(timeout=1.0)

    bus.close()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert results == [None]
    with pytest.raises(RuntimeError, match="closed LatestFrameBus"):
        bus.publish(b"late")


def test_realtime_media_matches_legacy_tensor_chunk_pixels() -> None:
    chunk = torch.tensor(
        [
            [
                [[-1.0, 0.0], [1.0, 2.0]],
                [[-2.0, 0.5], [0.0, 1.0]],
                [[1.0, -1.0], [0.0, 0.0]],
            ],
        ],
        dtype=torch.float32,
    )

    shared_frames = tensor_chunk_to_rgb_frames(chunk)

    assert len(shared_frames) == 1
    np.testing.assert_array_equal(
        shared_frames[0],
        np.array(
            [[[0, 0, 255], [127, 191, 0]], [[255, 127, 127], [255, 255, 127]]],
            dtype=np.uint8,
        ),
    )


def test_realtime_media_supports_omnidreams_uint8_layout() -> None:
    chunk = torch.zeros((1, 1, 2, 3, 4, 5), dtype=torch.uint8)
    chunk[0, 0, 1, 0] = 255

    frames = tensor_chunk_to_rgb_frames(chunk)

    assert len(frames) == 2
    assert frames[0].shape == (4, 5, 3)
    assert frames[0].dtype == np.uint8
    assert frames[1][0, 0, 0] == 255


@pytest.mark.parametrize(
    "chunk",
    [
        torch.full((1, 3, 2, 2), -1.0, dtype=torch.bfloat16),
        torch.full((1, 1, 1, 3, 2, 2), -1.0, dtype=torch.bfloat16),
    ],
)
def test_realtime_media_promotes_bfloat16_tensor_chunks(
    chunk: torch.Tensor,
) -> None:
    frames = tensor_chunk_to_rgb_frames(chunk)

    assert len(frames) == 1
    assert frames[0].shape == (2, 2, 3)
    assert frames[0].dtype == np.uint8
    assert frames[0].max() == 0


def test_realtime_media_rejects_non_rgb_bvtchw_layout() -> None:
    chunk = torch.zeros((1, 1, 2, 4, 5, 6), dtype=torch.uint8)

    with pytest.raises(ValueError, match=r"\[1, 1, T, 3, H, W\]"):
        tensor_chunk_to_rgb_frames(chunk)


def test_realtime_media_rejects_scaled_value_range_for_uint8() -> None:
    chunk = np.zeros((1, 2, 2, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="uint8 inputs require value_range='uint8'"):
        rgb_array_to_uint8_frames(
            chunk,
            layout="thwc",
            value_range="minus_one_one",
        )


def test_realtime_media_converts_array_chunks() -> None:
    chunk = np.array(
        [
            [[[0.0, 0.5, 1.0], [1.0, 0.0, 0.5]]],
            [[[0.25, 0.75, 1.0], [0.0, 0.0, 0.0]]],
        ],
        dtype=np.float32,
    )

    frames = rgb_array_to_uint8_frames(
        chunk,
        layout="thwc",
        value_range="zero_one",
    )

    assert len(frames) == 2
    np.testing.assert_array_equal(
        frames[0],
        np.array([[[0, 127, 255], [255, 0, 127]]], dtype=np.uint8),
    )


def test_realtime_media_encodes_jpeg_bytes() -> None:
    pytest.importorskip("PIL.Image")
    frame = np.full((4, 5, 3), 127, dtype=np.uint8)

    jpeg = encode_rgb_frame_to_jpeg(frame, quality=80)

    assert jpeg.startswith(b"\xff\xd8")
    assert jpeg.endswith(b"\xff\xd9")
