# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import pytest
from omnidreams.grpc.server import WorldModelService

pytestmark = pytest.mark.ci_gpu


class _DummyWrapper:
    frame_chunk_size = 4
    initial_frame_chunk_size = 8


class _DummyEngine:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.sessions: dict[str, object] = {}
        self.conditioning_wrapper = _DummyWrapper()
        self.seed_for_every_rollout_default = 42
        self.n_cameras = 1

    def _cleanup_session(self, session_id: str) -> None:
        self.calls.append(("cleanup", session_id))

    def open_session_on_all_ranks(self, payload=None) -> None:
        self.calls.append(("open", payload))

    def render_video_chunk_all_ranks(self, payload=None):
        self.calls.append(("render", payload))
        return {"ok": True}

    def finalize_kv_cache_all_ranks(self, session_id=None) -> None:
        self.calls.append(("finalize", session_id))

    def close_session_all_ranks(self, session_id=None) -> None:
        self.calls.append(("close", session_id))


class _DummyRecorder:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_service_delegates_runtime_operations_to_engine() -> None:
    engine = _DummyEngine()
    service = WorldModelService(engine=engine)  # ty:ignore[invalid-argument-type]

    service.open_session_on_all_ranks("payload")  # ty:ignore[invalid-argument-type]
    render_result = service.render_video_chunk_all_ranks("payload")  # ty:ignore[invalid-argument-type]
    service.finalize_kv_cache_all_ranks("session")
    service.close_session_all_ranks("session")

    assert render_result == {"ok": True}
    assert engine.calls[:4] == [
        ("open", "payload"),
        ("render", "payload"),
        ("finalize", "session"),
        ("close", "session"),
    ]


def test_service_cleanup_closes_recorder_and_engine_session() -> None:
    engine = _DummyEngine()
    service = WorldModelService(engine=engine)  # ty:ignore[invalid-argument-type]
    recorder = _DummyRecorder()
    service.recorders["session-a"] = recorder  # ty:ignore[invalid-assignment]

    service._cleanup_session("session-a")

    assert recorder.closed is True
    assert "session-a" not in service.recorders
    assert ("cleanup", "session-a") in engine.calls
