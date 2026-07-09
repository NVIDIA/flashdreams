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

import asyncio
import json
from typing import cast

import pytest
import torch
from lingbot2.webrtc import session
from lingbot2.webrtc.session import (
    Lingbot2RuntimeConfig,
    Lingbot2WebRTCSessionManager,
)

from flashdreams.serving.webrtc.manager import WebRTCStepResult

pytestmark = pytest.mark.ci_cpu


class _FakeCloseable:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _FakeControlChannel:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] = []

    def send(self, payload: str) -> None:
        decoded = json.loads(payload)
        assert isinstance(decoded, dict)
        self.messages.append(decoded)


def _fake_runtime_factory(config: Lingbot2RuntimeConfig) -> object:
    del config
    return object()


def test_session_manager_hooks_are_wired() -> None:
    # Guards against the shared base-class attribute overrides being dropped
    # (e.g. losing their leading underscore), which silently reverts behaviour
    # to the base defaults.
    assert (
        Lingbot2WebRTCSessionManager._busy_message
        == "A Lingbot2 session is already active."
    )
    assert Lingbot2WebRTCSessionManager._warmup_label == "Lingbot2 WebRTC"
    assert Lingbot2WebRTCSessionManager._runtime_error_types == (
        session.Lingbot2RuntimeError,
    )
    # Lingbot2 keeps streaming after a per-chunk failure rather than tearing down.
    assert Lingbot2WebRTCSessionManager._close_session_on_generation_error is False


def test_validate_remote_url_normalizes_github_blob_image_url() -> None:
    image_url = (
        "https://github.com/Robbyant/lingbot-world-v2/blob/main/examples/03/image.jpg"
    )
    assert session._validate_remote_url(image_url, field_name="image") == (
        "https://raw.githubusercontent.com/Robbyant/lingbot-world-v2/main/examples/03/image.jpg"
    )


def test_initial_scene_advertises_text_event_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        _active_event_id = None

        def __init__(self, config: Lingbot2RuntimeConfig) -> None:
            self.config = config

        def _load_default_prompt(self) -> str:
            return "drive through a city"

    monkeypatch.setattr(session, "Lingbot2InferenceRuntime", _FakeRuntime)
    manager = Lingbot2WebRTCSessionManager(
        runtime_config=Lingbot2RuntimeConfig(device="cpu", warmup_chunks=0)
    )

    scene = manager.get_initial_scene()

    assert scene["capabilities"] == {"text_events": True}
    assert scene["active_event_id"] is None
    assert scene["event_catalog"] == [
        event.as_public_dict() for event in session.DEFAULT_TEXT_EVENTS
    ]


@pytest.mark.asyncio
async def test_event_message_dispatches_to_runtime_and_acknowledges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def trigger_event(
            self, *, event_id: str, state: str
        ) -> dict[str, object]:
            self.calls.append((event_id, state))
            return {"active_event_id": event_id}

    monkeypatch.setattr(session, "Lingbot2InferenceRuntime", _fake_runtime_factory)
    manager = Lingbot2WebRTCSessionManager(
        runtime_config=Lingbot2RuntimeConfig(device="cpu", warmup_chunks=0)
    )
    runtime = _FakeRuntime()
    channel = _FakeControlChannel()
    managed_session = session._ManagedLingbot2Session(
        runtime=runtime,
        video_track=_FakeCloseable(),  # ty:ignore[invalid-argument-type]
        peer_connection=_FakeCloseable(),
        resampler=object(),  # ty:ignore[invalid-argument-type]
        control_channel=channel,
    )

    await manager._handle_datachannel_message(
        managed_session=managed_session,
        raw_message='{"type":"event","event_id":"portal","state":"trigger"}',
    )

    assert runtime.calls == [("portal", "trigger")]
    assert channel.messages == [
        {
            "type": "event_ack",
            "event_id": "portal",
            "state": "trigger",
            "active_event_id": "portal",
        }
    ]
    assert managed_session.first_action_received.is_set()


def test_trigger_event_sync_swaps_precomputed_text_embeddings() -> None:
    class _FakeTransformer:
        def __init__(self) -> None:
            self.calls: list[tuple[object, torch.Tensor]] = []

        def replace_text_embeddings(
            self, cache: object, text_embeddings: torch.Tensor
        ) -> None:
            self.calls.append((cache, text_embeddings))

    class _FakeDiffusionModel:
        def __init__(self) -> None:
            self.transformer = _FakeTransformer()

    class _FakePipeline:
        def __init__(self) -> None:
            self.diffusion_model = _FakeDiffusionModel()

    runtime = session.Lingbot2InferenceRuntime(
        config=Lingbot2RuntimeConfig(
            device="cpu",
            warmup_chunks=0,
            text_events=(),
        )
    )
    transformer_cache = object()
    cache = type("_FakeCache", (), {"transformer_cache": transformer_cache})()
    base_text = torch.zeros((1, 2, 3))
    event_text = torch.ones((1, 2, 3))
    runtime._pipeline = _FakePipeline()  # ty:ignore[invalid-assignment]
    runtime._cache = cache
    runtime._base_text_embeddings = base_text
    runtime._event_embeddings = {"portal": event_text}

    result = runtime._trigger_event_sync(event_id="portal", state="trigger")

    transformer = runtime._pipeline.diffusion_model.transformer
    assert result == {"active_event_id": "portal"}
    assert runtime._active_event_id == "portal"
    assert transformer.calls == [(transformer_cache, event_text)]

    result = runtime._trigger_event_sync(event_id="portal", state="clear")

    assert result == {"active_event_id": None}
    assert runtime._active_event_id is None
    assert transformer.calls[-1] == (transformer_cache, base_text)


@pytest.mark.asyncio
async def test_session_manager_preload_runs_loopback_warmup_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        def __init__(self, config: Lingbot2RuntimeConfig) -> None:
            self.config = config
            self.initialize_calls = 0
            self.close_calls = 0

        async def initialize(self) -> None:
            self.initialize_calls += 1

        async def close(self) -> None:
            self.close_calls += 1

    fake_runtime: _FakeRuntime | None = None
    warmup_calls: list[int] = []

    def _fake_runtime_factory(config: Lingbot2RuntimeConfig) -> _FakeRuntime:
        nonlocal fake_runtime
        fake_runtime = _FakeRuntime(config)
        return fake_runtime

    async def _fake_loopback_warmup(
        self: Lingbot2WebRTCSessionManager, *, num_chunks: int
    ) -> None:
        del self
        warmup_calls.append(num_chunks)

    monkeypatch.setattr(session, "Lingbot2InferenceRuntime", _fake_runtime_factory)
    monkeypatch.setattr(
        Lingbot2WebRTCSessionManager,
        "_run_loopback_warmup_session",
        _fake_loopback_warmup,
    )
    manager = Lingbot2WebRTCSessionManager(
        runtime_config=Lingbot2RuntimeConfig(device="cpu", warmup_chunks=2)
    )

    await manager.preload_runtime()
    await manager.preload_runtime()

    assert fake_runtime is not None
    assert fake_runtime.initialize_calls == 1
    assert warmup_calls == [2]
    assert manager.is_runtime_ready()


@pytest.mark.asyncio
async def test_loopback_warmup_drives_session_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        def __init__(self, config: Lingbot2RuntimeConfig) -> None:
            self.config = config
            self.initialize_calls = 0
            self.reset_calls = 0
            self.close_calls = 0
            self.generated_segments: list[
                list[tuple[float, float, frozenset[str]]]
            ] = []

        async def initialize(self) -> None:
            self.initialize_calls += 1

        async def reset_for_new_session(
            self, session_input: session.Lingbot2SessionInput | None = None
        ) -> None:
            del session_input
            self.reset_calls += 1

        def peek_steady_chunk_num_frames(self) -> int:
            return 1

        def peek_next_chunk_num_frames(self) -> int:
            return 1

        async def generate_chunk(
            self,
            *,
            segments: list[tuple[float, float, frozenset[str]]],
            frame_times: list[float],
        ) -> WebRTCStepResult:
            del frame_times
            chunk_index = len(self.generated_segments)
            self.generated_segments.append(segments)
            return WebRTCStepResult(
                chunk_index=chunk_index,
                num_frames=1,
                video_chunk=torch.zeros((1, 1, 1, 3, 2, 2), dtype=torch.uint8),
                stats=None,
            )

        async def close(self) -> None:
            self.close_calls += 1

    fake_runtime: _FakeRuntime | None = None

    def _fake_runtime_factory(config: Lingbot2RuntimeConfig) -> _FakeRuntime:
        nonlocal fake_runtime
        fake_runtime = _FakeRuntime(config)
        return fake_runtime

    monkeypatch.setattr(session, "Lingbot2InferenceRuntime", _fake_runtime_factory)
    manager = Lingbot2WebRTCSessionManager(
        runtime_config=Lingbot2RuntimeConfig(
            device="cpu",
            warmup_chunks=2,
        ),
        fps=30,
    )

    await asyncio.wait_for(manager.preload_runtime(), timeout=10.0)

    assert fake_runtime is not None
    assert fake_runtime.initialize_calls == 1
    assert fake_runtime.reset_calls == 1
    assert len(fake_runtime.generated_segments) == 2
    assert not manager.has_active_session()


@pytest.mark.asyncio
async def test_loopback_warmup_skips_when_configured_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        def __init__(self, config: Lingbot2RuntimeConfig) -> None:
            self.config = config
            self.initialize_calls = 0
            self.reset_calls = 0
            self.close_calls = 0

        async def initialize(self) -> None:
            self.initialize_calls += 1

        async def reset_for_new_session(
            self, session_input: session.Lingbot2SessionInput | None = None
        ) -> None:
            del session_input
            self.reset_calls += 1

        async def close(self) -> None:
            self.close_calls += 1

    fake_runtime: _FakeRuntime | None = None

    def _fake_runtime_factory(config: Lingbot2RuntimeConfig) -> _FakeRuntime:
        nonlocal fake_runtime
        fake_runtime = _FakeRuntime(config)
        return fake_runtime

    monkeypatch.setattr(session, "Lingbot2InferenceRuntime", _fake_runtime_factory)
    manager = Lingbot2WebRTCSessionManager(
        runtime_config=Lingbot2RuntimeConfig(device="cpu", warmup_chunks=0)
    )

    await manager.preload_runtime()

    assert fake_runtime is not None
    assert fake_runtime.initialize_calls == 1
    assert fake_runtime.reset_calls == 0
    assert not manager.has_active_session()


@pytest.mark.asyncio
async def test_create_answer_passes_pending_session_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session, "Lingbot2InferenceRuntime", _fake_runtime_factory)
    manager = Lingbot2WebRTCSessionManager(
        runtime_config=Lingbot2RuntimeConfig(device="cpu", warmup_chunks=0)
    )
    manager._runtime_ready = True
    manager._warmup_complete = True
    session_input = session.Lingbot2SessionInput(prompt="follow a coastal highway")
    manager.set_pending_session_input(session_input)
    captured_inputs: list[session.Lingbot2SessionInput | None] = []

    async def _fake_create_answer_with_runtime_ready_locked(
        **kwargs: object,
    ) -> dict[str, str]:
        captured_inputs.append(
            cast(session.Lingbot2SessionInput | None, kwargs.get("session_input"))
        )
        return {"sdp": "answer-sdp", "type": "answer"}

    monkeypatch.setattr(
        manager,
        "_create_answer_with_runtime_ready_locked",
        _fake_create_answer_with_runtime_ready_locked,
    )

    answer = await manager.create_answer(offer_sdp="offer-sdp", offer_type="offer")

    assert answer == {"sdp": "answer-sdp", "type": "answer"}
    assert captured_inputs == [session_input]
    assert manager._pending_session_input is None


@pytest.mark.asyncio
async def test_heartbeat_message_refreshes_client_liveness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session, "Lingbot2InferenceRuntime", _fake_runtime_factory)
    manager = Lingbot2WebRTCSessionManager(
        runtime_config=Lingbot2RuntimeConfig(device="cpu", warmup_chunks=0)
    )
    managed_session = session._ManagedLingbot2Session(
        runtime=object(),
        video_track=_FakeCloseable(),  # ty:ignore[invalid-argument-type]
        peer_connection=_FakeCloseable(),
        resampler=object(),  # ty:ignore[invalid-argument-type]
        control_channel=object(),
        last_client_message_at=0.0,
    )
    manager._active_session = managed_session

    await manager._handle_datachannel_message(
        managed_session=managed_session,
        raw_message='{"type":"heartbeat"}',
    )

    assert managed_session.last_client_message_at > 0.0
    assert manager.has_active_session()


@pytest.mark.asyncio
async def test_client_liveness_timeout_closes_active_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session, "Lingbot2InferenceRuntime", _fake_runtime_factory)
    manager = Lingbot2WebRTCSessionManager(
        runtime_config=Lingbot2RuntimeConfig(device="cpu", warmup_chunks=0),
        client_liveness_timeout_s=0.01,
    )
    video_track = _FakeCloseable()
    peer_connection = _FakeCloseable()
    managed_session = session._ManagedLingbot2Session(
        runtime=object(),
        video_track=video_track,  # ty:ignore[invalid-argument-type]
        peer_connection=peer_connection,
        resampler=object(),  # ty:ignore[invalid-argument-type]
        last_client_message_at=asyncio.get_running_loop().time() - 1.0,
    )
    manager._active_session = managed_session
    liveness_task = asyncio.create_task(
        manager._client_liveness_watchdog(managed_session=managed_session)
    )
    managed_session.liveness_task = liveness_task

    await asyncio.wait_for(liveness_task, timeout=1.0)

    assert not manager.has_active_session()
    assert managed_session.closed
    assert video_track.closed
    assert peer_connection.closed


@pytest.mark.asyncio
async def test_disconnect_message_closes_active_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session, "Lingbot2InferenceRuntime", _fake_runtime_factory)
    manager = Lingbot2WebRTCSessionManager(
        runtime_config=Lingbot2RuntimeConfig(device="cpu", warmup_chunks=0)
    )
    video_track = _FakeCloseable()
    peer_connection = _FakeCloseable()
    managed_session = session._ManagedLingbot2Session(
        runtime=object(),
        video_track=video_track,  # ty:ignore[invalid-argument-type]
        peer_connection=peer_connection,
        resampler=object(),  # ty:ignore[invalid-argument-type]
        control_channel=object(),
    )
    manager._active_session = managed_session

    await manager._handle_datachannel_message(
        managed_session=managed_session,
        raw_message='{"type":"disconnect"}',
    )

    assert not manager.has_active_session()
    assert managed_session.closed
    assert video_track.closed
    assert peer_connection.closed
