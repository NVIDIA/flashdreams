# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from flashdreams.runtime.keyboard import (
    ImageRequest,
    KeyboardState,
    PromptRequest,
    ResetRequest,
    SparseInputSnapshot,
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
