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
import ipaddress
import json
from pathlib import Path
from typing import cast

import pytest
import torch
from lingbot.webrtc import session
from lingbot.webrtc.session import (
    LingbotRuntimeConfig,
    LingbotWebRTCSessionManager,
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


def _fake_runtime_factory(config: LingbotRuntimeConfig) -> object:
    del config
    return object()


def test_session_manager_hooks_are_wired() -> None:
    # Guards against the shared base-class attribute overrides being dropped
    # (e.g. losing their leading underscore), which silently reverts behaviour
    # to the base defaults.
    assert (
        LingbotWebRTCSessionManager._busy_message
        == "A Lingbot session is already active."
    )
    assert LingbotWebRTCSessionManager._warmup_label == "Lingbot WebRTC"
    assert LingbotWebRTCSessionManager._runtime_error_types == (
        session.LingbotRuntimeError,
    )
    # Lingbot keeps streaming after a per-chunk failure rather than tearing down.
    assert LingbotWebRTCSessionManager._close_session_on_generation_error is False


def test_validate_remote_url_normalizes_github_blob_image_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        session,
        "_resolve_remote_host",
        lambda hostname: (ipaddress.ip_address("140.82.112.4"),),
    )
    image_url = (
        "https://github.com/Robbyant/lingbot-world-v2/blob/main/examples/03/image.jpg"
    )
    assert session._validate_remote_url(image_url, field_name="image") == (
        "https://raw.githubusercontent.com/Robbyant/lingbot-world-v2/main/examples/03/image.jpg"
    )


@pytest.mark.parametrize(
    "image_url",
    [
        "http://127.0.0.1/image.jpg",
        "http://10.0.0.5/image.jpg",
        "http://172.16.0.5/image.jpg",
        "http://192.168.1.10/image.jpg",
        "http://169.254.169.254/latest/meta-data",
        "http://[::1]/image.jpg",
        "http://localhost/image.jpg",
    ],
)
def test_validate_remote_url_rejects_non_public_hosts(image_url: str) -> None:
    with pytest.raises(ValueError, match="publicly routable"):
        session._validate_remote_url(image_url, field_name="image")


def test_read_remote_bytes_rejects_non_public_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_read_once(
        url: str, *, max_bytes: int, field_name: str
    ) -> tuple[bytes, str, str | None]:
        del url, max_bytes, field_name
        return b"", "", "http://127.0.0.1/image.jpg"

    with pytest.raises(ValueError, match="publicly routable"):
        monkeypatch.setattr(session, "_read_remote_bytes_once", _fake_read_once)
        session._read_remote_bytes(
            "https://example.test/image.jpg",
            max_bytes=1024,
            field_name="image",
        )


def test_read_remote_bytes_uses_validated_resolved_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved_address = ipaddress.ip_address("93.184.216.34")
    calls: list[dict[str, object]] = []

    class _FakeHeaders:
        @staticmethod
        def get_content_type() -> str:
            return "image/jpeg"

    class _FakeResponse:
        status = 200
        headers = _FakeHeaders()

        @staticmethod
        def getheader(name: str) -> str | None:
            del name
            return None

        @staticmethod
        def read(size: int | None = None) -> bytes:
            del size
            return b"image-bytes"

        @staticmethod
        def close() -> None:
            return

    class _FakeConnection:
        def __init__(
            self,
            host: str,
            *,
            port: int | None,
            timeout: float,
            resolved_address: ipaddress.IPv4Address | ipaddress.IPv6Address,
        ) -> None:
            calls.append(
                {
                    "host": host,
                    "port": port,
                    "timeout": timeout,
                    "resolved_address": resolved_address,
                }
            )

        def request(self, method: str, target: str, *, headers: dict[str, str]) -> None:
            calls[-1]["method"] = method
            calls[-1]["target"] = target
            calls[-1]["headers"] = headers

        @staticmethod
        def getresponse() -> _FakeResponse:
            return _FakeResponse()

        @staticmethod
        def close() -> None:
            return

    monkeypatch.setattr(
        session, "_resolve_remote_host", lambda hostname: (resolved_address,)
    )
    monkeypatch.setattr(session, "_ResolvedHTTPConnection", _FakeConnection)

    data, content_type = session._read_remote_bytes(
        "http://example.test:8080/path/to/image.jpg?token=1",
        max_bytes=1024,
        field_name="image",
    )

    assert data == b"image-bytes"
    assert content_type == "image/jpeg"
    assert calls == [
        {
            "host": "example.test",
            "port": 8080,
            "timeout": session._REMOTE_READ_TIMEOUT_S,
            "resolved_address": resolved_address,
            "method": "GET",
            "target": "/path/to/image.jpg?token=1",
            "headers": {"User-Agent": "flashdreams-lingbot-webrtc/1.0"},
        }
    ]


def test_initial_scene_advertises_text_event_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        _active_event_id = None

        def __init__(self, config: LingbotRuntimeConfig) -> None:
            self.config = config

        def _load_default_prompt(self) -> str:
            return "drive through a city"

    monkeypatch.setattr(session, "LingbotInferenceRuntime", _FakeRuntime)
    manager = LingbotWebRTCSessionManager(
        runtime_config=LingbotRuntimeConfig(device="cpu", warmup_chunks=0)
    )

    scene = manager.get_initial_scene()

    assert scene["capabilities"] == {"text_events": True}
    assert scene["active_event_id"] is None
    assert scene["event_catalog"] == [
        event.as_public_dict() for event in session.DEFAULT_TEXT_EVENTS
    ]


def test_missing_default_prompt_warns_and_resolves_to_empty_string(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use an empty prompt when the example has no ``prompt.txt`` file."""
    runtime = object.__new__(session.LingbotInferenceRuntime)
    runtime.config = LingbotRuntimeConfig(example_data_dir=tmp_path)
    warnings: list[str] = []
    monkeypatch.setattr(
        session.logger,
        "warning",
        lambda message, *args: warnings.append(message.format(*args)),
    )

    assert runtime._load_default_prompt() == ""
    assert warnings == [
        f"LingBot prompt.txt is missing or empty at {tmp_path / 'prompt.txt'}; "
        "proceeding with an empty prompt."
    ]


def test_pending_session_input_overrides_text_event_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        _active_event_id = None

        def __init__(self, config: LingbotRuntimeConfig) -> None:
            self.config = config

        def _load_default_prompt(self) -> str:
            return "drive through a city"

    monkeypatch.setattr(session, "LingbotInferenceRuntime", _FakeRuntime)
    manager = LingbotWebRTCSessionManager(
        runtime_config=LingbotRuntimeConfig(device="cpu", warmup_chunks=0)
    )
    custom_events = (
        session.TextEventSpec(
            event_id="rain",
            label="Rain",
            prompt="Rain begins falling across the street.",
            category="custom",
        ),
    )

    manager.set_pending_session_input(
        session.LingbotSessionInput(text_events=custom_events)
    )
    scene = manager.get_initial_scene()

    assert scene["capabilities"] == {"text_events": True}
    assert scene["event_catalog"] == [custom_events[0].as_public_dict()]


def test_pending_remote_first_frame_is_fetched_once_and_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        _active_event_id = None

        def __init__(self, config: LingbotRuntimeConfig) -> None:
            self.config = config
            self.decoded_images: list[bytes] = []

        def _load_default_prompt(self) -> str:
            return "drive through a city"

        def _load_uploaded_first_frame_rgb(self, image_bytes: bytes) -> object:
            self.decoded_images.append(image_bytes)
            return object()

    fake_runtime: _FakeRuntime | None = None
    read_calls: list[str] = []

    def _fake_runtime_factory(config: LingbotRuntimeConfig) -> _FakeRuntime:
        nonlocal fake_runtime
        fake_runtime = _FakeRuntime(config)
        return fake_runtime

    def _fake_read_remote_bytes(
        url: str, *, max_bytes: int, field_name: str
    ) -> tuple[bytes, str]:
        del max_bytes, field_name
        read_calls.append(url)
        return b"remote-image", "image/png"

    monkeypatch.setattr(session, "LingbotInferenceRuntime", _fake_runtime_factory)
    monkeypatch.setattr(
        session,
        "_resolve_remote_host",
        lambda hostname: (ipaddress.ip_address("93.184.216.34"),),
    )
    monkeypatch.setattr(session, "_read_remote_bytes", _fake_read_remote_bytes)
    manager = LingbotWebRTCSessionManager(
        runtime_config=LingbotRuntimeConfig(device="cpu", warmup_chunks=0)
    )

    manager.set_pending_session_input(
        session.LingbotSessionInput(
            first_frame_image_url="https://example.test/scene.png"
        )
    )
    payload = manager.get_first_frame()

    assert fake_runtime is not None
    assert fake_runtime.decoded_images == [b"remote-image"]
    assert read_calls == ["https://example.test/scene.png"]
    assert payload == session.LingbotImagePayload(
        data=b"remote-image",
        content_type="image/png",
    )
    assert manager._pending_session_input is not None
    assert manager._pending_session_input.first_frame_remote_payload == payload


def test_prepare_session_input_state_uses_cached_remote_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = session.LingbotInferenceRuntime(
        config=LingbotRuntimeConfig(device="cpu", warmup_chunks=0)
    )
    runtime._device = torch.device("cpu")
    decoded_images: list[bytes] = []

    def _fake_load_uploaded_first_frame_rgb(image_bytes: bytes) -> object:
        decoded_images.append(image_bytes)
        return object()

    def _fail_remote_fetch(image_url: str) -> object:
        raise AssertionError(f"unexpected remote fetch: {image_url}")

    monkeypatch.setattr(
        runtime,
        "_load_uploaded_first_frame_rgb",
        _fake_load_uploaded_first_frame_rgb,
    )
    monkeypatch.setattr(runtime, "_load_remote_first_frame_rgb", _fail_remote_fetch)
    monkeypatch.setattr(
        runtime,
        "_first_frame_to_tensor",
        lambda image_rgb: torch.zeros((1, 3, 2, 2)),
    )
    monkeypatch.setattr(
        runtime,
        "_encode_text_embeddings_sync",
        lambda texts: torch.zeros((len(texts), 1, 2)),
    )

    runtime._prepare_session_input_state(
        session.LingbotSessionInput(
            prompt="follow a coastal highway",
            first_frame_image_url="https://example.test/scene.png",
            first_frame_remote_payload=session.LingbotImagePayload(
                data=b"cached-image",
                content_type="image/png",
            ),
        )
    )

    assert decoded_images == [b"cached-image"]
    assert runtime._prompt == "follow a coastal highway"


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

    monkeypatch.setattr(session, "LingbotInferenceRuntime", _fake_runtime_factory)
    manager = LingbotWebRTCSessionManager(
        runtime_config=LingbotRuntimeConfig(device="cpu", warmup_chunks=0)
    )
    runtime = _FakeRuntime()
    channel = _FakeControlChannel()
    managed_session = session._ManagedLingbotSession(
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


@pytest.mark.asyncio
async def test_clear_event_message_does_not_require_event_id_and_preserves_ack_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def trigger_event(
            self, *, event_id: str, state: str
        ) -> dict[str, object]:
            self.calls.append((event_id, state))
            return {
                "type": "not_event_ack",
                "event_id": "overwritten",
                "state": "overwritten",
                "active_event_id": None,
            }

    monkeypatch.setattr(session, "LingbotInferenceRuntime", _fake_runtime_factory)
    manager = LingbotWebRTCSessionManager(
        runtime_config=LingbotRuntimeConfig(device="cpu", warmup_chunks=0)
    )
    runtime = _FakeRuntime()
    channel = _FakeControlChannel()
    managed_session = session._ManagedLingbotSession(
        runtime=runtime,
        video_track=_FakeCloseable(),  # ty:ignore[invalid-argument-type]
        peer_connection=_FakeCloseable(),
        resampler=object(),  # ty:ignore[invalid-argument-type]
        control_channel=channel,
    )

    await manager._handle_datachannel_message(
        managed_session=managed_session,
        raw_message='{"type":"event","state":"clear"}',
    )

    assert runtime.calls == [("", "clear")]
    assert channel.messages == [
        {
            "type": "event_ack",
            "event_id": None,
            "state": "clear",
            "active_event_id": None,
        }
    ]
    assert managed_session.first_action_received.is_set()


@pytest.mark.asyncio
async def test_event_message_without_id_is_rejected_for_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        def __init__(self) -> None:
            self.calls = 0

        async def trigger_event(
            self, *, event_id: str, state: str
        ) -> dict[str, object]:
            del event_id, state
            self.calls += 1
            return {}

    monkeypatch.setattr(session, "LingbotInferenceRuntime", _fake_runtime_factory)
    manager = LingbotWebRTCSessionManager(
        runtime_config=LingbotRuntimeConfig(device="cpu", warmup_chunks=0)
    )
    runtime = _FakeRuntime()
    channel = _FakeControlChannel()
    managed_session = session._ManagedLingbotSession(
        runtime=runtime,
        video_track=_FakeCloseable(),  # ty:ignore[invalid-argument-type]
        peer_connection=_FakeCloseable(),
        resampler=object(),  # ty:ignore[invalid-argument-type]
        control_channel=channel,
    )

    await manager._handle_datachannel_message(
        managed_session=managed_session,
        raw_message='{"type":"event","state":"trigger"}',
    )

    assert runtime.calls == 0
    assert channel.messages == [
        {
            "type": "error",
            "message": (
                "Event payload must include non-empty 'event_id' "
                "unless state clears the active event."
            ),
        }
    ]
    assert not managed_session.first_action_received.is_set()


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

    runtime = session.LingbotInferenceRuntime(
        config=LingbotRuntimeConfig(
            device="cpu",
            warmup_chunks=0,
            text_events=(),
        )
    )
    transformer_cache = object()
    cache = type("_FakeCache", (), {"transformer_cache": transformer_cache})()
    base_text = torch.zeros((1, 2, 3))
    event_text = torch.ones((1, 2, 3))
    runtime._pipeline = _FakePipeline()
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


def test_reset_rollout_precomputes_session_text_events() -> None:
    class _FakePipeline:
        def __init__(self) -> None:
            self.encoded_texts: list[tuple[str, ...]] = []

        def _ensure_oneshot_encoders_loaded(self) -> None:
            return

        def text_encoder(self, texts: list[str]) -> torch.Tensor:
            self.encoded_texts.append(tuple(texts))
            return torch.arange(len(texts) * 2, dtype=torch.float32).reshape(
                len(texts), 1, 2
            )

        def initialize_cache(self, *, text: list[str], image: torch.Tensor) -> object:
            del text, image
            return object()

    runtime = session.LingbotInferenceRuntime(
        config=LingbotRuntimeConfig(
            device="cpu",
            warmup_chunks=0,
            text_events=(),
        )
    )
    pipeline = _FakePipeline()
    runtime._device = torch.device("cpu")
    runtime._pipeline = pipeline

    def _fake_prepare_session_input_state(
        session_input: session.LingbotSessionInput | None,
    ) -> None:
        del session_input
        runtime._first_frames = torch.zeros((1, 3, 2, 2))
        runtime._prompt = "base prompt"
        runtime._base_text_embeddings = torch.zeros((1, 1, 2))

    setattr(
        runtime,
        "_prepare_session_input_state",
        _fake_prepare_session_input_state,
    )
    custom_events = (
        session.TextEventSpec(
            event_id="rain",
            label="Rain",
            prompt="Rain begins falling across the street.",
        ),
    )

    runtime._reset_rollout_sync(session.LingbotSessionInput(text_events=custom_events))

    assert pipeline.encoded_texts == [("Rain begins falling across the street.",)]
    assert set(runtime._event_embeddings) == {"rain"}


@pytest.mark.asyncio
async def test_trigger_event_prevalidates_before_distributed_broadcast() -> None:
    runtime = session.LingbotInferenceRuntime(
        config=LingbotRuntimeConfig(
            device="cpu",
            warmup_chunks=0,
            text_events=(),
        )
    )
    runtime._pipeline = object()
    runtime._cache = object()
    runtime._event_embeddings = {"portal": torch.ones((1, 2, 3))}
    calls = 0

    def _fail_if_called(event_id: str, state: str) -> dict[str, str | None]:
        nonlocal calls
        del event_id, state
        calls += 1
        raise AssertionError("distributed event op should not be invoked")

    runtime._trigger_event_sync_all_ranks = _fail_if_called

    with pytest.raises(ValueError, match="Unknown event_id='unknown'"):
        await runtime.trigger_event(event_id="unknown", state="trigger")

    assert calls == 0


@pytest.mark.asyncio
async def test_trigger_event_waits_for_generation_lock() -> None:
    runtime = session.LingbotInferenceRuntime(
        config=LingbotRuntimeConfig(
            device="cpu",
            warmup_chunks=0,
            text_events=(),
        )
    )
    runtime._pipeline = object()
    runtime._cache = object()
    runtime._event_embeddings = {"portal": torch.ones((1, 2, 3))}
    calls: list[tuple[str, str]] = []

    def _fake_event_op(event_id: str, state: str) -> dict[str, str | None]:
        calls.append((event_id, state))
        return {"active_event_id": event_id}

    runtime._trigger_event_sync_all_ranks = _fake_event_op

    await runtime._step_lock.acquire()
    task = asyncio.create_task(
        runtime.trigger_event(event_id="portal", state="trigger")
    )
    await asyncio.sleep(0)
    assert not task.done()
    assert calls == []

    runtime._step_lock.release()
    result = await asyncio.wait_for(task, timeout=1.0)

    assert result == {"active_event_id": "portal"}
    assert calls == [("portal", "trigger")]


@pytest.mark.asyncio
async def test_session_manager_preload_runs_loopback_warmup_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeRuntime:
        def __init__(self, config: LingbotRuntimeConfig) -> None:
            self.config = config
            self.initialize_calls = 0
            self.close_calls = 0

        async def initialize(self) -> None:
            self.initialize_calls += 1

        async def close(self) -> None:
            self.close_calls += 1

    fake_runtime: _FakeRuntime | None = None
    warmup_calls: list[int] = []

    def _fake_runtime_factory(config: LingbotRuntimeConfig) -> _FakeRuntime:
        nonlocal fake_runtime
        fake_runtime = _FakeRuntime(config)
        return fake_runtime

    async def _fake_loopback_warmup(
        self: LingbotWebRTCSessionManager, *, num_chunks: int
    ) -> None:
        del self
        warmup_calls.append(num_chunks)

    monkeypatch.setattr(session, "LingbotInferenceRuntime", _fake_runtime_factory)
    monkeypatch.setattr(
        LingbotWebRTCSessionManager,
        "_run_loopback_warmup_session",
        _fake_loopback_warmup,
    )
    manager = LingbotWebRTCSessionManager(
        runtime_config=LingbotRuntimeConfig(device="cpu", warmup_chunks=2)
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
        def __init__(self, config: LingbotRuntimeConfig) -> None:
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
            self, session_input: session.LingbotSessionInput | None = None
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

    def _fake_runtime_factory(config: LingbotRuntimeConfig) -> _FakeRuntime:
        nonlocal fake_runtime
        fake_runtime = _FakeRuntime(config)
        return fake_runtime

    monkeypatch.setattr(session, "LingbotInferenceRuntime", _fake_runtime_factory)
    manager = LingbotWebRTCSessionManager(
        runtime_config=LingbotRuntimeConfig(
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
        def __init__(self, config: LingbotRuntimeConfig) -> None:
            self.config = config
            self.initialize_calls = 0
            self.reset_calls = 0
            self.close_calls = 0

        async def initialize(self) -> None:
            self.initialize_calls += 1

        async def reset_for_new_session(
            self, session_input: session.LingbotSessionInput | None = None
        ) -> None:
            del session_input
            self.reset_calls += 1

        async def close(self) -> None:
            self.close_calls += 1

    fake_runtime: _FakeRuntime | None = None

    def _fake_runtime_factory(config: LingbotRuntimeConfig) -> _FakeRuntime:
        nonlocal fake_runtime
        fake_runtime = _FakeRuntime(config)
        return fake_runtime

    monkeypatch.setattr(session, "LingbotInferenceRuntime", _fake_runtime_factory)
    manager = LingbotWebRTCSessionManager(
        runtime_config=LingbotRuntimeConfig(device="cpu", warmup_chunks=0)
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
    monkeypatch.setattr(session, "LingbotInferenceRuntime", _fake_runtime_factory)
    manager = LingbotWebRTCSessionManager(
        runtime_config=LingbotRuntimeConfig(device="cpu", warmup_chunks=0)
    )
    manager._runtime_ready = True
    manager._warmup_complete = True
    session_input = session.LingbotSessionInput(prompt="follow a coastal highway")
    manager.set_pending_session_input(session_input)
    captured_inputs: list[session.LingbotSessionInput | None] = []

    async def _fake_create_answer_with_runtime_ready_locked(
        **kwargs: object,
    ) -> dict[str, str]:
        captured_inputs.append(
            cast(session.LingbotSessionInput | None, kwargs.get("session_input"))
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
    monkeypatch.setattr(session, "LingbotInferenceRuntime", _fake_runtime_factory)
    manager = LingbotWebRTCSessionManager(
        runtime_config=LingbotRuntimeConfig(device="cpu", warmup_chunks=0)
    )
    managed_session = session._ManagedLingbotSession(
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
    monkeypatch.setattr(session, "LingbotInferenceRuntime", _fake_runtime_factory)
    manager = LingbotWebRTCSessionManager(
        runtime_config=LingbotRuntimeConfig(device="cpu", warmup_chunks=0),
        client_liveness_timeout_s=0.01,
    )
    video_track = _FakeCloseable()
    peer_connection = _FakeCloseable()
    managed_session = session._ManagedLingbotSession(
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
    monkeypatch.setattr(session, "LingbotInferenceRuntime", _fake_runtime_factory)
    manager = LingbotWebRTCSessionManager(
        runtime_config=LingbotRuntimeConfig(device="cpu", warmup_chunks=0)
    )
    video_track = _FakeCloseable()
    peer_connection = _FakeCloseable()
    managed_session = session._ManagedLingbotSession(
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
