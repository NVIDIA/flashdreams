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
from pathlib import Path
from typing import cast

import pytest
import torch
from lingbot.input_mapping import (
    KeyboardToCameraCommand,
    LingbotInputMapping,
    TextEventSelection,
)
from lingbot.model_session import LingbotModelSessionCore
from lingbot.webrtc import session
from lingbot.webrtc.session import (
    LingbotRuntimeConfig,
    create_lingbot_webrtc_session_manager,
)

from flashdreams.infra.video_output import VideoOutputStream
from flashdreams.runtime import StepRequest, StepResult
from flashdreams.runtime.canonical import InputCanonicalizer
from flashdreams.runtime.inputs import InferenceInput
from flashdreams.serving.webrtc import runtime as webrtc_runtime
from flashdreams.serving.webrtc.manager import (
    BaseWebRTCSessionManager,
    ManagedWebRTCSession,
)

pytestmark = pytest.mark.ci_cpu


def _attach_model_session(
    runtime: session.LingbotInferenceRuntime,
    pipeline: object,
    *,
    cache: object | None = None,
) -> LingbotModelSessionCore:
    core = LingbotModelSessionCore(
        pipeline=pipeline,
        output_stream_factory=lambda: VideoOutputStream(
            postprocess_stream=None,
            output_layout="tchw",
        ),
    )
    core._cache = cache  # Test seam for already-initialized runtime state.
    runtime._model_session = core
    return core


class _FakeCloseable:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _FakeVideoEncoder:
    """Minimal ``VideoEncoder``-shaped stub for ``ManagedWebRTCSession``
    construction. Enough to satisfy the dataclass field; the tests here do
    not exercise ``create_track`` / ``deliver_chunk`` on it."""

    fps = 30
    backend = "fake"
    prefers_codec: str | None = None

    def close(self) -> None:
        return


def _fake_runtime_factory(config: LingbotRuntimeConfig) -> object:
    del config
    return object()


def test_session_manager_hooks_are_wired() -> None:
    manager = create_lingbot_webrtc_session_manager(
        runtime_config=LingbotRuntimeConfig(device="cpu")
    )

    assert manager.busy_message == "A Lingbot session is already active."
    assert manager.warmup_label == "Lingbot WebRTC"
    assert manager.fatal_generation_errors is False


def test_runtime_defaults_use_canonical_v2_examples() -> None:
    """Use LingBot-World v2 assets for the shared WebRTC defaults."""
    config = LingbotRuntimeConfig()
    expected_base_url = (
        "https://raw.githubusercontent.com/Robbyant/lingbot-world-v2/main/examples/00"
    )

    assert config.default_image_url == f"{expected_base_url}/image.jpg"
    assert config.default_intrinsics_url == f"{expected_base_url}/intrinsics.npy"
    assert config.default_poses_url == f"{expected_base_url}/poses.npy"
    assert config.fps == 16
    assert config.encoder_backend == "auto"


def test_session_manager_uses_runtime_config_fps_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(session, "LingbotInferenceRuntime", _fake_runtime_factory)

    manager = create_lingbot_webrtc_session_manager(
        runtime_config=LingbotRuntimeConfig(device="cpu", warmup_chunks=0, fps=12)
    )

    assert manager.fps == 12


def test_initialize_video_encoder_sync_skips_on_non_master(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _select_encoder_should_not_be_called(**_kw: object) -> object:
        raise AssertionError("worker ranks must not initialize WebRTC encoders")

    monkeypatch.setattr(
        webrtc_runtime, "select_encoder", _select_encoder_should_not_be_called
    )
    runtime = session.LingbotInferenceRuntime(
        config=LingbotRuntimeConfig(device="cpu", warmup_chunks=0)
    )
    runtime.rank = 1
    runtime._device = torch.device("cpu")

    runtime._initialize_video_encoder_sync()

    assert runtime._video_encoder is None


def test_initialize_video_encoder_sync_selects_runtime_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _FakeVideoEncoder()
    calls: list[dict[str, object]] = []

    def _fake_select_encoder(**kwargs: object) -> _FakeVideoEncoder:
        calls.append(kwargs)
        return stub

    monkeypatch.setattr(webrtc_runtime, "select_encoder", _fake_select_encoder)
    runtime = session.LingbotInferenceRuntime(
        config=LingbotRuntimeConfig(
            device="cuda:2",
            warmup_chunks=0,
            video_height=360,
            video_width=640,
            fps=12,
            encoder_backend="auto",
            encoder_bitrate_bps=4_000_000,
            encoder_gop=24,
        )
    )
    runtime.rank = 0
    runtime._device = torch.device("cuda:2")

    runtime._initialize_video_encoder_sync()

    assert runtime._video_encoder is stub
    assert calls == [
        {
            "backend": "auto",
            "width": 640,
            "height": 360,
            "fps": 12,
            "bitrate": 4_000_000,
            "gpu_id": 2,
            "gop": 24,
        }
    ]


def test_generate_one_chunk_sync_hands_gpu_resident_output_to_output_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoCpuChunk:
        detach_calls = 0

        def detach(self) -> "_NoCpuChunk":
            self.detach_calls += 1
            return self

        def cpu(self) -> object:
            raise AssertionError("LingBot WebRTC must not eagerly call .cpu()")

    class _FakePipeline:
        def __init__(self) -> None:
            self.output = _NoCpuChunk()

        @staticmethod
        def get_num_output_frames(autoregressive_index: int) -> int:
            assert autoregressive_index == 0
            return 2

        def generate(self, **_kwargs: object) -> _NoCpuChunk:
            return self.output

        @staticmethod
        def finalize(autoregressive_index: int, cache: object) -> dict[str, float]:
            assert autoregressive_index == 0
            return {"total_ms": 3.0}

    captured: dict[str, object] = {}

    def _fake_process(
        _stream: VideoOutputStream, video_chunk: object, **kwargs: object
    ) -> StepResult:
        captured["video_chunk"] = video_chunk
        captured.update(kwargs)
        return StepResult.from_video_chunk(
            step_index=0,
            video_chunk=torch.zeros((2, 3, 4, 5)),
            layout="tchw",
            metrics={"total_ms": 3.0},
        )

    runtime = session.LingbotInferenceRuntime(
        config=LingbotRuntimeConfig(device="cpu", warmup_chunks=0)
    )
    pipeline = _FakePipeline()
    runtime._device = torch.device("cpu")
    runtime._pipeline = pipeline
    _attach_model_session(runtime, pipeline, cache=object())
    runtime._base_intrinsics = torch.ones(4)
    monkeypatch.setattr(
        VideoOutputStream,
        "process",
        _fake_process,
    )

    result = runtime._generate_one_chunk_sync(
        segments=[(0.0, 1.0, frozenset())],
        frame_times=[0.25, 0.5],
    )

    assert captured["video_chunk"] is pipeline.output
    metrics = cast(dict[str, float], captured["metrics"])
    assert metrics["total_ms"] == 3.0
    assert float(metrics["model_step_s"]) >= 0.0
    assert pipeline.output.detach_calls == 0
    assert result.metrics == {"total_ms": 3.0}
    assert runtime._model_session is not None
    assert runtime._model_session.step_index == 1


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
    manager = create_lingbot_webrtc_session_manager(
        runtime_config=LingbotRuntimeConfig(device="cpu", warmup_chunks=0)
    )
    controller = session.LingbotWebRTCSessionController(manager)

    scene = controller.get_initial_scene()

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
    manager = create_lingbot_webrtc_session_manager(
        runtime_config=LingbotRuntimeConfig(device="cpu", warmup_chunks=0)
    )
    controller = session.LingbotWebRTCSessionController(manager)
    custom_events = (
        session.TextEventSpec(
            event_id="rain",
            label="Rain",
            prompt="Rain begins falling across the street.",
            category="custom",
        ),
    )

    controller.set_pending_session_input(
        session.LingbotSessionInput(text_events=custom_events)
    )
    scene = controller.get_initial_scene()

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
    manager = create_lingbot_webrtc_session_manager(
        runtime_config=LingbotRuntimeConfig(device="cpu", warmup_chunks=0)
    )
    controller = session.LingbotWebRTCSessionController(manager)

    controller.set_pending_session_input(
        session.LingbotSessionInput(
            first_frame_image_url="https://example.test/scene.png"
        )
    )
    payload = controller.get_first_frame()

    assert fake_runtime is not None
    assert fake_runtime.decoded_images == [b"remote-image"]
    assert read_calls == ["https://example.test/scene.png"]
    assert payload == session.LingbotImagePayload(
        data=b"remote-image",
        content_type="image/png",
    )
    assert manager.pending_session_input is not None
    assert manager.pending_session_input.first_frame_remote_payload == payload


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


def test_apply_conditioning_update_swaps_precomputed_text_embeddings() -> None:
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
    _attach_model_session(runtime, runtime._pipeline, cache=cache)
    runtime._prompt = "base prompt"
    runtime._event_embeddings = {"portal": event_text}
    runtime._prompt_embeddings = {
        "base prompt": base_text,
        "a glowing portal opens": event_text,
    }

    runtime._apply_conditioning_update_sync(
        InferenceInput(global_conditioning={"prompt": "a glowing portal opens"})
    )

    transformer = runtime._pipeline.diffusion_model.transformer
    assert runtime._active_event_id == "portal"
    assert runtime._prompt == "a glowing portal opens"
    assert transformer.calls == [(transformer_cache, event_text)]

    runtime._apply_conditioning_update_sync(
        InferenceInput(global_conditioning={"prompt": "base prompt"})
    )

    assert runtime._active_event_id is None
    assert runtime._prompt == "base prompt"
    assert transformer.calls[-1] == (transformer_cache, base_text)


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
    _attach_model_session(runtime, runtime._pipeline, cache=cache)
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


def test_validate_user_event_rejects_invalid_text_events() -> None:
    runtime = session.LingbotInferenceRuntime(
        config=LingbotRuntimeConfig(
            device="cpu",
            warmup_chunks=0,
            text_events=(),
        )
    )
    runtime._event_embeddings = {"portal": torch.ones((1, 2, 3))}

    assert runtime.validate_user_event(
        event_type="text_event",
        payload={"event_id": "portal", "state": "trigger"},
    ) == {"event_id": "portal", "state": "trigger"}
    assert runtime.validate_user_event(
        event_type="text_event",
        payload={"event_id": None, "state": "clear"},
    ) == {"event_id": None, "state": "clear"}
    with pytest.raises(ValueError, match="Unknown event_id='unknown'"):
        runtime.validate_user_event(
            event_type="text_event",
            payload={"event_id": "unknown", "state": "trigger"},
        )
    with pytest.raises(ValueError, match="Event state must be one of"):
        runtime.validate_user_event(
            event_type="text_event",
            payload={"event_id": "portal", "state": "explode"},
        )


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
    _attach_model_session(runtime, pipeline)

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
    _attach_model_session(runtime, runtime._pipeline, cache=object())
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
    _attach_model_session(runtime, runtime._pipeline, cache=object())
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
        self: BaseWebRTCSessionManager, *, num_chunks: int
    ) -> None:
        del self
        warmup_calls.append(num_chunks)

    monkeypatch.setattr(session, "LingbotInferenceRuntime", _fake_runtime_factory)
    monkeypatch.setattr(
        BaseWebRTCSessionManager,
        "_run_loopback_warmup_session",
        _fake_loopback_warmup,
    )
    manager = create_lingbot_webrtc_session_manager(
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
    class _FakeInferenceSession:
        def __init__(self, runtime: "_FakeRuntime") -> None:
            self._runtime = runtime

        def next_step_request(self) -> StepRequest:
            step_index = self._runtime.step_index
            return StepRequest(
                step_index=step_index,
                metadata={
                    "input_frame_count": 1,
                    "num_frames": 1,
                    "frame_start": step_index,
                },
            )

        def step(self, inputs: InferenceInput) -> StepResult:
            chunk_index = self._runtime.step_index
            self._runtime.step_index += 1
            self._runtime.generated_inputs.append(inputs)
            return StepResult.from_video_chunk(
                step_index=chunk_index,
                video_chunk=torch.zeros((1, 1, 1, 3, 2, 2), dtype=torch.uint8),
                layout="bvtchw",
            )

    class _FakeRuntime:
        def __init__(self, config: LingbotRuntimeConfig) -> None:
            self.config = config
            self.initialize_calls = 0
            self.reset_calls = 0
            self.close_calls = 0
            self.step_index = 0
            self.generated_inputs: list[InferenceInput] = []
            self.input_canonicalizer = InputCanonicalizer(
                [KeyboardToCameraCommand(), TextEventSelection()]
            )
            self.input_mapping = LingbotInputMapping(
                fps=30,
                base_intrinsics=torch.tensor([416.0, 416.0, 416.0, 240.0]),
                world_scale=1.0,
                text_event_prompts={},
            )
            self.input_mapping.set_base_prompt("warmup prompt")
            self.input_source_schema = session.LINGBOT_WEBRTC_SOURCE_SCHEMA
            self._active_event_id = None

        async def initialize(self) -> None:
            self.initialize_calls += 1

        async def reset_for_new_session(
            self, session_input: session.LingbotSessionInput | None = None
        ) -> None:
            del session_input
            self.reset_calls += 1

        def peek_input_fps(self) -> float:
            return 30.0

        def peek_steady_output_num_frames(self) -> int:
            return 1

        def next_step_request(self) -> StepRequest:
            return StepRequest(
                step_index=self.step_index,
                metadata={"input_frame_count": 1},
            )

        async def start_inference_session(self) -> _FakeInferenceSession:
            return _FakeInferenceSession(self)

        async def close(self) -> None:
            self.close_calls += 1

    fake_runtime: _FakeRuntime | None = None

    def _fake_runtime_factory(config: LingbotRuntimeConfig) -> _FakeRuntime:
        nonlocal fake_runtime
        fake_runtime = _FakeRuntime(config)
        return fake_runtime

    monkeypatch.setattr(session, "LingbotInferenceRuntime", _fake_runtime_factory)
    manager = create_lingbot_webrtc_session_manager(
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
    assert len(fake_runtime.generated_inputs) == 2
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
    manager = create_lingbot_webrtc_session_manager(
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
    manager = create_lingbot_webrtc_session_manager(
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
    manager = create_lingbot_webrtc_session_manager(
        runtime_config=LingbotRuntimeConfig(device="cpu", warmup_chunks=0)
    )
    managed_session = ManagedWebRTCSession(
        runtime=object(),
        video_track=_FakeCloseable(),  # ty:ignore[invalid-argument-type]
        video_encoder=_FakeVideoEncoder(),  # ty:ignore[invalid-argument-type]
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
    manager = create_lingbot_webrtc_session_manager(
        runtime_config=LingbotRuntimeConfig(device="cpu", warmup_chunks=0),
        client_liveness_timeout_s=0.01,
    )
    video_track = _FakeCloseable()
    peer_connection = _FakeCloseable()
    managed_session = ManagedWebRTCSession(
        runtime=object(),
        video_track=video_track,  # ty:ignore[invalid-argument-type]
        video_encoder=_FakeVideoEncoder(),  # ty:ignore[invalid-argument-type]
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
    manager = create_lingbot_webrtc_session_manager(
        runtime_config=LingbotRuntimeConfig(device="cpu", warmup_chunks=0)
    )
    video_track = _FakeCloseable()
    peer_connection = _FakeCloseable()
    managed_session = ManagedWebRTCSession(
        runtime=object(),
        video_track=video_track,  # ty:ignore[invalid-argument-type]
        video_encoder=_FakeVideoEncoder(),  # ty:ignore[invalid-argument-type]
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
