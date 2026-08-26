# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for the thin native MiniMax H3 V2 applications."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
import tomli
from PIL import Image

import minimax_h3_v2.app as app_module
from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.audio_output import AudioOutput
from flashdreams.runtime_v2.application_runner import ApplicationRunner
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout
from minimax_h3.conditioning import audio_latent_num_frames
from minimax_h3.inference import (
    MiniMaxH3InferenceConfig,
    MiniMaxH3InferenceRequest,
    MiniMaxH3InferenceResult,
    MiniMaxH3Workflow,
)
from minimax_h3.reference_conditioning import MiniMaxH3VideoReference
from minimax_h3_v2.app import (
    MiniMaxH3Application,
    create_app,
    create_fl2va_app,
    create_ref2va_app,
    create_t2va_app,
)

pytestmark = pytest.mark.ci_cpu


class _FakeEngine:
    """Record native requests and return shape-exact synchronized media."""

    def __init__(self, config: MiniMaxH3InferenceConfig) -> None:
        """Retain the resource config and initialize lifecycle observations."""
        self.config = config
        self.requests: list[MiniMaxH3InferenceRequest] = []
        self.close_count = 0

    def generate(self, request: MiniMaxH3InferenceRequest) -> MiniMaxH3InferenceResult:
        """Return final-grid video and H3-hop audio for one request."""
        self.requests.append(request)
        audio_samples = audio_latent_num_frames(request.num_frames) * 800
        return MiniMaxH3InferenceResult(
            video=torch.zeros(
                request.num_frames,
                3,
                request.height,
                request.width,
                dtype=torch.float32,
            ),
            audio=AudioOutput(
                samples=torch.zeros(2, audio_samples),
                sample_rate=32000,
            ),
            metrics={"fake": 1},
        )

    def close(self) -> None:
        """Record application-owned release."""
        self.close_count += 1


class _EngineFactory:
    """Construct one fake engine and record when allocation occurs."""

    def __init__(self, events: list[str] | None = None) -> None:
        """Share an optional ordering log with media loader fakes."""
        self.events = events
        self.engines: list[_FakeEngine] = []

    def __call__(self, config: MiniMaxH3InferenceConfig) -> _FakeEngine:
        """Create and retain a fake engine."""
        if self.events is not None:
            self.events.append("engine")
        engine = _FakeEngine(config)
        self.engines.append(engine)
        return engine


class _OneFrameEngine(_FakeEngine):
    """Return one frame so a full runtime test does not wait five seconds."""

    def generate(self, request: MiniMaxH3InferenceRequest) -> MiniMaxH3InferenceResult:
        """Return one synchronized chunk while retaining the real request."""
        self.requests.append(request)
        return MiniMaxH3InferenceResult(
            video=torch.zeros(1, 3, request.height, request.width),
            audio=AudioOutput(samples=torch.zeros(2, 100), sample_rate=32000),
        )


class _FailingEngine(_FakeEngine):
    """Raise from the model thread to exercise transactional cleanup."""

    def generate(self, request: MiniMaxH3InferenceRequest) -> MiniMaxH3InferenceResult:
        """Fail after recording the validated request."""
        self.requests.append(request)
        raise RuntimeError("H3 inference failed")


class _RecordingWindow(IClientWindow):
    """Record synchronized UI results and close-versus-abort ownership."""

    def __init__(self) -> None:
        """Initialize lifecycle and media observations."""
        self.opened = False
        self.closed = False
        self.aborted = False
        self.results: list[StepResult] = []

    def get_user_input_events(self) -> UserInputEvents:
        """Return no cancellation events."""
        return UserInputEvents([])

    def open(self, session_desc: SessionDesc) -> None:
        """Record that the runtime opened the output contract."""
        del session_desc
        self.opened = True

    def write(self, result: StepResult) -> None:
        """Record one UI result."""
        self.results.append(result)

    def close(self) -> None:
        """Record successful transaction finalization."""
        self.closed = True

    def abort(self) -> None:
        """Record exceptional transaction cleanup."""
        self.aborted = True


def _initialized_app(
    workflow: str = "t2va",
    *,
    args: list[str] | None = None,
) -> tuple[MiniMaxH3Application, _FakeEngine]:
    """Return one initialized test app and its injected engine."""
    factory = _EngineFactory()
    app = MiniMaxH3Application(
        cast(MiniMaxH3Workflow, workflow),
        engine_factory=factory,
    )
    app.init([] if args is None else args)
    return app, factory.engines[0]


def _step(app: MiniMaxH3Application) -> tuple[StepResult, MiniMaxH3InferenceRequest]:
    """Create a tiny session, run its only step, and return result and request."""
    desc = replace(app.session_desc(), video_width=32, video_height=32)
    session = app.create_session(desc)
    session.init()
    results = session.model_loop.step(0, UserInputEvents([]))
    assert isinstance(results, list)
    request = session.model_loop.state.request
    return results[0], request


def test_factories_are_weight_free_and_describe_synchronized_media() -> None:
    """Expose all stable workflows without constructing default resources."""
    for workflow, factory in (
        ("t2va", create_t2va_app),
        ("fl2va", create_fl2va_app),
        ("ref2va", create_ref2va_app),
    ):
        app = factory()
        assert isinstance(app, IApplication)
        assert isinstance(app, MiniMaxH3Application)
        desc = app.session_desc()
        assert desc.metadata == {"workflow": workflow}
        assert desc.output_layout is VideoTensorLayout.tchw
        assert desc.backpressure_mode is BackpressureMode.BLOCK
        assert desc.presentation_mode is PresentationMode.ONLY_PRESENT_NEW
        assert desc.frames_per_second_for_step == 24
        assert (desc.video_width, desc.video_height) == (768, 768)
        assert (desc.audio_sample_rate, desc.audio_channels) == (32000, 2)
    default_app = create_app()
    assert isinstance(default_app, MiniMaxH3Application)
    assert default_app.session_desc().metadata == {"workflow": "t2va"}


def test_t2va_returns_maximum_aligned_video_audio_and_resets(tmp_path: Path) -> None:
    """Return 362 frames at 15 seconds and one synchronized audio payload."""
    app, engine = _initialized_app(
        args=[
            "--prompt",
            "A waterfall",
            "--duration",
            "15",
            "--steps",
            "2",
            "--seed",
            "7",
            "--work-dir",
            str(tmp_path),
            "--job-id",
            "rollout-7",
            "--device",
            "cpu",
            "--attention",
            "math",
        ]
    )
    desc = replace(app.session_desc(), video_width=32, video_height=32)
    session = app.create_session(desc)
    session.init()

    results = session.model_loop.step(0, UserInputEvents([]))
    assert isinstance(results, list)
    result = results[0]

    assert result.output.shape == (362, 3, 32, 32)
    assert result.frame_count == 362
    assert result.output_layout is VideoTensorLayout.tchw
    assert result.audio is not None
    assert result.audio.samples.shape == (2, 482_400)
    assert result.metrics == {"fake": 1}
    assert session.model_loop.is_finished()
    request = engine.requests[0]
    assert request.prompt == "A waterfall"
    assert request.seed == 7
    assert request.checkpoint_store is not None
    assert request.checkpoint_store.path == (
        tmp_path.resolve() / "rollout-7" / "minimax_h3" / "joint_latents.safetensors"
    )
    assert engine.config.device == "cpu"
    assert engine.config.attention_backend == "math"

    session.model_loop.reset()
    assert not session.model_loop.is_finished()
    session.model_loop.step(0, UserInputEvents([]))
    assert len(engine.requests) == 2
    app.close()
    app.close()
    assert engine.close_count == 1


def test_application_runner_presents_audio_once_and_closes_engine() -> None:
    """Carry synchronized audio through the default UI under runtime ownership."""
    engines: list[_OneFrameEngine] = []

    def factory(config: MiniMaxH3InferenceConfig) -> _OneFrameEngine:
        engine = _OneFrameEngine(config)
        engines.append(engine)
        return engine

    app = MiniMaxH3Application("t2va", engine_factory=factory)
    window = _RecordingWindow()
    desc = replace(app.session_desc(), video_width=32, video_height=32)

    ApplicationRunner(app, window).run(desc, ["--steps", "2"])

    assert window.opened and window.closed and not window.aborted
    assert len(window.results) == 1
    result = window.results[0]
    assert result.audio is not None
    assert result.audio.samples.shape == (2, 100)
    assert engines[0].close_count == 1


def test_application_runner_aborts_window_when_h3_inference_fails() -> None:
    """Propagate model failure while aborting output and releasing the engine."""
    engines: list[_FailingEngine] = []

    def factory(config: MiniMaxH3InferenceConfig) -> _FailingEngine:
        engine = _FailingEngine(config)
        engines.append(engine)
        return engine

    app = MiniMaxH3Application("t2va", engine_factory=factory)
    window = _RecordingWindow()
    desc = replace(app.session_desc(), video_width=32, video_height=32)

    with pytest.raises(RuntimeError, match="H3 inference failed"):
        ApplicationRunner(app, window).run(desc, ["--steps", "2"])

    assert window.opened and window.aborted and not window.closed
    assert window.results == []
    assert engines[0].close_count == 1


def test_fl2va_decodes_and_hashes_keyframes_before_engine(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Decode local keyframes before shared model metadata is allocated."""
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    first.write_bytes(b"first")
    last.write_bytes(b"last")
    events: list[str] = []

    def read_image(path: Path) -> np.ndarray:
        events.append(path.name)
        return np.zeros((8, 8, 3), dtype=np.uint8)

    monkeypatch.setattr(app_module, "read_image_rgb", read_image)
    factory = _EngineFactory(events)
    app = MiniMaxH3Application("fl2va", engine_factory=factory)
    app.init(
        [
            "--image-path",
            str(first),
            "--last-image-path",
            str(last),
            "--steps",
            "2",
        ]
    )
    _, request = _step(app)

    assert events == ["first.png", "last.png", "engine"]
    assert isinstance(request.first_image, Image.Image)
    assert isinstance(request.last_image, Image.Image)
    assert [value.source.split(":", 1)[0] for value in request.checkpoint_inputs] == [
        "first",
        "last",
    ]
    assert all(value.sha256 is not None for value in request.checkpoint_inputs)


def test_ref2va_preserves_order_and_optional_video_soundtrack(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Decode image, video, and audio references in their semantic order."""
    image_path = tmp_path / "subject.png"
    video_path = tmp_path / "motion.mp4"
    audio_path = tmp_path / "voice.wav"
    for path in (image_path, video_path, audio_path):
        path.write_bytes(path.name.encode())
    monkeypatch.setattr(
        app_module,
        "read_image_rgb",
        lambda _path: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        app_module,
        "read_video_rgb_with_fps",
        lambda _path: (np.zeros((2, 8, 8, 3), dtype=np.uint8), 24.0),
    )
    monkeypatch.setattr(
        app_module,
        "read_optional_audio_f32",
        lambda *_args, **_kwargs: np.zeros((2, 800), dtype=np.float32),
    )
    monkeypatch.setattr(
        app_module,
        "read_audio_f32",
        lambda *_args, **_kwargs: np.zeros((2, 400), dtype=np.float32),
    )
    app, _ = _initialized_app(
        "ref2va",
        args=[
            "--reference",
            f"image:{image_path}",
            "--reference",
            f"video:{video_path}",
            "--reference",
            f"audio:{audio_path}",
            "--steps",
            "2",
        ],
    )
    _, request = _step(app)

    assert [reference.kind for reference in request.references] == [
        "image",
        "video",
        "audio",
    ]
    video = request.references[1]
    assert isinstance(video, MiniMaxH3VideoReference)
    assert video.audio is not None
    assert video.sample_rate == 32000
    assert [value.source.split(":", 1)[0] for value in request.checkpoint_inputs] == [
        "image",
        "video",
        "audio",
    ]


@pytest.mark.parametrize(
    ("workflow", "args", "message"),
    [
        ("t2va", ["--duration", "16"], "duration"),
        ("t2va", ["--steps", "1"], "--steps"),
        ("t2va", ["--restart"], "--restart"),
        ("t2va", ["--lora-scale", "2"], "--lora-scale"),
        ("fl2va", [], "requires --image-path"),
    ],
)
def test_invalid_arguments_fail_before_engine_allocation(
    workflow: str,
    args: list[str],
    message: str,
) -> None:
    """Reject static work and workflow gaps before constructing resources."""
    factory = _EngineFactory()
    app = MiniMaxH3Application(
        cast(MiniMaxH3Workflow, workflow),
        engine_factory=factory,
    )

    with pytest.raises(ValueError, match=message):
        app.init(args)

    assert factory.engines == []


def test_audio_only_reference_fails_before_decode_or_engine(tmp_path: Path) -> None:
    """Enforce the released REF2VA modality constraint at the path boundary."""
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"voice")
    factory = _EngineFactory()
    app = MiniMaxH3Application("ref2va", engine_factory=factory)

    with pytest.raises(ValueError, match="paired with an image or video"):
        app.init(["--reference", f"audio:{audio}"])

    assert factory.engines == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("output_layout", VideoTensorLayout.bcthw, "tchw"),
        ("backpressure_mode", BackpressureMode.DROP_OLDEST, "block"),
        (
            "presentation_mode",
            PresentationMode.ONLY_PRESENT_NEWEST,
            "only_present_new",
        ),
        ("frames_per_second_for_step", 30, "24 fps"),
        ("audio_sample_rate", 16000, "32000 Hz"),
    ],
)
def test_session_rejects_runtime_contract_overrides(
    field: str,
    value: object,
    message: str,
) -> None:
    """Reject runtime descriptions the fixed H3 media cannot honor."""
    app, _ = _initialized_app()
    desc = replace(app.session_desc(), **{field: value})

    with pytest.raises(ValueError, match=message):
        app.create_session(desc)


def test_create_session_requires_init_and_valid_canvas() -> None:
    """Keep weight-free construction separate from request validation."""
    factory = _EngineFactory()
    app = MiniMaxH3Application("t2va", engine_factory=factory)
    with pytest.raises(RuntimeError, match="init"):
        app.create_session(app.session_desc())
    app.init(["--steps", "2"])
    with pytest.raises(ValueError, match="multiples of 32"):
        app.create_session(replace(app.session_desc(), video_width=33))


def test_manifest_registers_three_stable_v2_slugs() -> None:
    """Keep each workflow independently discoverable by the V2 registry."""
    manifest = tomli.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    entry_points = manifest["project"]["entry-points"]["flashdreams.applications_v2"]

    assert entry_points == {
        "minimax-h3-t2va": "minimax_h3_v2.app:create_t2va_app",
        "minimax-h3-fl2va": "minimax_h3_v2.app:create_fl2va_app",
        "minimax-h3-ref2va": "minimax_h3_v2.app:create_ref2va_app",
    }
