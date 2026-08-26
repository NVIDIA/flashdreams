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

"""CPU lifecycle tests for the LingBot-VA V2 application."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from lingbot_va.constants import ROBOTWIN_OBS_CAM_KEYS
from lingbot_va.engine import LingbotVAEngineConfig, LingbotVAEngineOutput
from lingbot_va_v2.app import (
    ACTIONS_SCHEMA,
    LingbotVAApplication,
    create_app,
)

from flashdreams.api_v2.application import IApplication
from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.runtime_v2.session_desc import (
    BackpressureMode,
    PresentationMode,
    SessionDesc,
)
from flashdreams.runtime_v2.session_runner import run_session
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.tensor_artifact_output_sink import (
    TensorArtifactOutputSink,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu


class _FakeEngine:
    """Emit deterministic natural-shape CPU outputs."""

    def __init__(
        self,
        config: LingbotVAEngineConfig,
        *,
        video_shape: tuple[int, ...] | None = None,
    ) -> None:
        self.config = config
        self.closed = False
        self.runs = 0
        frame_count = config.num_chunks * 8 - 3
        action_steps = config.num_chunks * 32
        self._output = LingbotVAEngineOutput(
            video=torch.zeros(
                video_shape or (frame_count, 3, 256, 320),
                dtype=torch.float32,
            ),
            actions=torch.arange(action_steps * 16, dtype=torch.float32).reshape(
                action_steps,
                16,
            ),
            metrics={"total_s": 1.25},
        )

    def run(self) -> LingbotVAEngineOutput:
        self.runs += 1
        return self._output

    def close(self) -> None:
        self.closed = True


class _FakeEngineFactory:
    """Record every session/reset-owned engine instance."""

    def __init__(self, *, video_shape: tuple[int, ...] | None = None) -> None:
        self.engines: list[_FakeEngine] = []
        self._video_shape = video_shape

    def __call__(self, config: LingbotVAEngineConfig) -> _FakeEngine:
        engine = _FakeEngine(config, video_shape=self._video_shape)
        self.engines.append(engine)
        return engine


class _RecordingWindow(IClientWindow):
    """Collect outputs while reporting no user input."""

    def __init__(self) -> None:
        self.session_desc: SessionDesc | None = None
        self.results: list[StepResult] = []
        self.closed = False

    def get_user_input_events(self) -> UserInputEvents:
        return UserInputEvents([])

    def open(self, session_desc: SessionDesc) -> None:
        self.session_desc = session_desc

    def write(self, result: StepResult) -> None:
        self.results.append(result)

    def close(self) -> None:
        self.closed = True


def _input_dir(tmp_path: Path) -> Path:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    for key in ROBOTWIN_OBS_CAM_KEYS:
        (input_dir / f"{key}.png").touch()
    return input_dir


def _init_app(
    tmp_path: Path,
    factory: _FakeEngineFactory,
    *extra_args: str,
) -> LingbotVAApplication:
    app = LingbotVAApplication(factory)
    app.init(
        [
            "--device",
            "cpu",
            "--input-image-dir",
            str(_input_dir(tmp_path)),
            "--num-chunks",
            "1",
            "--no-compile",
            *extra_args,
        ]
    )
    return app


def _step(session: Any, step_index: int = 0) -> StepResult:
    model_loop = session.model_loop
    results = model_loop.step(step_index, UserInputEvents([]))
    assert isinstance(results, list)
    return results[0]


def test_create_app_returns_v2_application() -> None:
    assert isinstance(create_app(), IApplication)


def test_session_desc_is_cheap_and_natural_before_init() -> None:
    factory = _FakeEngineFactory()
    app = LingbotVAApplication(factory)

    session_desc = app.session_desc()

    assert factory.engines == []
    assert session_desc.output_layout is VideoTensorLayout.tchw
    assert session_desc.backpressure_mode is BackpressureMode.BLOCK
    assert session_desc.presentation_mode is PresentationMode.ONLY_PRESENT_NEW
    assert session_desc.frames_per_second_for_ui == 10
    assert session_desc.frames_per_second_for_step == 10
    assert (session_desc.video_width, session_desc.video_height) == (320, 256)
    assert session_desc.tensor_artifact_schemas == (ACTIONS_SCHEMA,)


def test_application_init_validates_without_creating_engine(tmp_path: Path) -> None:
    factory = _FakeEngineFactory()

    app = _init_app(tmp_path, factory)

    assert factory.engines == []
    session = app.create_session(app.session_desc())
    assert factory.engines == []
    session.init()
    assert factory.engines == []


def test_one_step_returns_video_actions_and_metrics(tmp_path: Path) -> None:
    factory = _FakeEngineFactory()
    app = _init_app(tmp_path, factory)
    session = app.create_session(app.session_desc())
    session.init()

    result = _step(session)

    assert result.output.shape == (5, 3, 256, 320)
    assert result.output_layout is VideoTensorLayout.tchw
    assert result.frame_count == 5
    assert result.metrics == {"total_s": 1.25}
    assert len(result.tensor_artifacts) == 1
    actions = result.tensor_artifacts[0]
    assert actions.schema is ACTIONS_SCHEMA
    assert actions.tensor.shape == (32, 16)
    assert session.model_loop.is_finished()


def test_reset_closes_engine_and_lazily_creates_another(tmp_path: Path) -> None:
    factory = _FakeEngineFactory()
    app = _init_app(tmp_path, factory)
    session = app.create_session(app.session_desc())
    session.init()
    _step(session)

    session.model_loop.reset()

    assert factory.engines[0].closed
    assert not session.model_loop.is_finished()
    assert len(factory.engines) == 1
    _step(session)
    assert len(factory.engines) == 2
    session.close()
    assert factory.engines[1].closed


def test_create_session_rejects_misdescribed_robotwin_output(tmp_path: Path) -> None:
    app = _init_app(tmp_path, _FakeEngineFactory())

    with pytest.raises(ValueError, match="video_width"):
        app.create_session(replace(app.session_desc(), video_width=640))


def test_create_session_preserves_runtime_policies(tmp_path: Path) -> None:
    app = _init_app(tmp_path, _FakeEngineFactory())
    requested = replace(
        app.session_desc(),
        backpressure_mode=BackpressureMode.DROP_OLDEST,
        presentation_mode=PresentationMode.ONLY_PRESENT_NEWEST,
        metadata={"caller": "preserved"},
    )

    session = app.create_session(requested)

    assert session.session_desc.backpressure_mode is BackpressureMode.DROP_OLDEST
    assert (
        session.session_desc.presentation_mode is PresentationMode.ONLY_PRESENT_NEWEST
    )
    assert session.session_desc.metadata["caller"] == "preserved"
    assert session.session_desc.metadata["action_dim"] == 30


def test_create_session_before_init_fails() -> None:
    app = LingbotVAApplication(_FakeEngineFactory())

    with pytest.raises(RuntimeError, match="init"):
        app.create_session(app.session_desc())


def test_init_requires_an_explicit_input_image_directory() -> None:
    app = LingbotVAApplication(_FakeEngineFactory())

    with pytest.raises(SystemExit):
        app.init(["--device", "cpu"])


def test_init_rejects_missing_camera_inputs(tmp_path: Path) -> None:
    app = LingbotVAApplication(_FakeEngineFactory())

    with pytest.raises(FileNotFoundError, match="camera PNGs"):
        app.init(
            [
                "--device",
                "cpu",
                "--input-image-dir",
                str(tmp_path),
            ]
        )


def test_init_rejects_nonexistent_explicit_checkpoint(tmp_path: Path) -> None:
    app = LingbotVAApplication(_FakeEngineFactory())
    input_dir = _input_dir(tmp_path)

    with pytest.raises(FileNotFoundError, match="checkpoint root"):
        app.init(
            [
                "--device",
                "cpu",
                "--input-image-dir",
                str(input_dir),
                "--checkpoint-root",
                str(tmp_path / "missing-checkpoint"),
            ]
        )


def test_every_cli_override_reaches_engine_config(tmp_path: Path) -> None:
    factory = _FakeEngineFactory()
    app = _init_app(
        tmp_path,
        factory,
        "--checkpoint-root",
        "owner/repo",
        "--checkpoint-revision",
        "revision-1",
        "--prompt",
        "do the thing",
        "--seed",
        "9",
        "--enable-offload",
        "--guidance-scale",
        "3.5",
        "--action-guidance-scale",
        "2.5",
        "--video-inference-steps",
        "7",
        "--action-inference-steps",
        "8",
        "--video-snr-shift",
        "4.5",
        "--action-snr-shift",
        "1.5",
    )
    session = app.create_session(app.session_desc())
    session.init()

    _step(session)
    config = factory.engines[0].config

    assert config.checkpoint_root == "owner/repo"
    assert config.checkpoint_revision == "revision-1"
    assert config.prompt == "do the thing"
    assert config.seed == 9
    assert config.enable_offload is True
    assert config.compile_network is False
    assert config.guidance_scale == 3.5
    assert config.action_guidance_scale == 2.5
    assert config.video_inference_steps == 7
    assert config.action_inference_steps == 8
    assert config.video_snr_shift == 4.5
    assert config.action_snr_shift == 1.5


def test_model_loop_rejects_wrong_engine_video_shape(tmp_path: Path) -> None:
    factory = _FakeEngineFactory(video_shape=(1, 3, 256, 320))
    app = _init_app(tmp_path, factory)
    session = app.create_session(app.session_desc())
    session.init()

    with pytest.raises(ValueError, match="video shape"):
        _step(session)

    session.close()
    assert factory.engines[0].closed


def test_runtime_routes_actions_through_generic_sink(tmp_path: Path) -> None:
    factory = _FakeEngineFactory()
    app = _init_app(tmp_path, factory)
    session = app.create_session(app.session_desc())
    window = _RecordingWindow()
    artifact_dir = tmp_path / "artifacts"

    run_session(
        session,
        window,
        model_output_sinks=[TensorArtifactOutputSink(artifact_dir)],
    )

    assert window.closed
    assert factory.engines[0].closed
    actions = np.load(artifact_dir / "actions.npy")
    assert actions.shape == (32, 16)
    np.testing.assert_array_equal(actions, np.arange(32 * 16).reshape(32, 16))
