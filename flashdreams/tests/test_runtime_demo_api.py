# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from flashdreams.infra.video_output import VideoStepResult
from flashdreams.runtime import (
    DRIVER_COMMAND,
    RGB_VIDEO,
    CanonicalInputs,
    CanonicalInputSchema,
    IdentityInputMapping,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    InferenceOutputSchema,
    InferenceRuntime,
    InferenceSession,
    InputCanonicalizer,
    InputField,
    InputMapping,
    InputMappingSchema,
    NullMetricsRecorder,
    NullOutputTarget,
    OutputArtifact,
    OutputTarget,
    StepRequest,
    StepResult,
    TimeWindow,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.demo import (
    DemoRoute,
    DemoSpec,
    LocalWindowOutputSpec,
    Mp4OutputSpec,
    NullOutputSpec,
    PreparedSession,
    WebRTCOutputSpec,
    build_output_target,
    run_replay_demo,
)
from flashdreams.runtime.demo.local_window import (
    build_local_window_demo,
    build_local_window_io,
    run_local_window_session,
)
from flashdreams.runtime.demo.webrtc import build_webrtc_demo
from flashdreams.serving.presentation import DisplayFrame, NullOverlay
from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager

pytestmark = pytest.mark.ci_cpu


def test_shared_demo_import_does_not_require_local_window_dependencies() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "sys.modules['PIL'] = None; "
                "sys.modules['slangpy'] = None; "
                "import flashdreams.runtime.demo"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("factory", "mode"),
    (
        (lambda mode: NullOutputSpec(mode=mode), "mp4"),
        (lambda mode: Mp4OutputSpec(path="out.mp4", fps=30, mode=mode), "null"),
        (lambda mode: WebRTCOutputSpec(mode=mode), "local-window"),
        (lambda mode: LocalWindowOutputSpec(mode=mode), "webrtc"),
    ),
)
def test_output_spec_discriminator_cannot_disagree_with_its_type(
    factory: Any,
    mode: Any,
) -> None:
    with pytest.raises(ValueError, match="mode must be"):
        factory(mode)


def test_replay_demo_uses_shared_runner() -> None:
    adapter = _FakeDemoAdapter()
    output = _RecordingOutputTarget()
    calls: list[dict[str, Any]] = []

    def fake_runner(**kwargs: Any) -> Sequence[OutputArtifact]:
        calls.append(kwargs)
        return (OutputArtifact(kind="test/artifact", uri="memory://artifact"),)

    spec = DemoSpec(
        model_id="fake-demo",
        scenario="valid-scenario",
        input_mode="replay",
        output=NullOutputSpec(),
    )

    artifacts = run_replay_demo(
        spec=spec,
        adapter=adapter,
        output_target_factory=lambda output_spec: output,
        metrics=NullMetricsRecorder(),
        runner=fake_runner,
    )

    assert artifacts == (OutputArtifact(kind="test/artifact", uri="memory://artifact"),)
    assert len(calls) == 1
    assert calls[0]["adapter"] is adapter
    assert calls[0]["config"] == spec.config
    assert calls[0]["mapping"] is adapter.prepared_session.mapping
    assert calls[0]["canonicalizer"] is adapter.prepared_session.canonicalizer
    assert calls[0]["source_schema"] is adapter.prepared_session.source_schema
    assert calls[0]["user_inputs"] is adapter.prepared_session.user_inputs
    assert calls[0]["initial_inputs"] is adapter.prepared_session.initial_inputs
    assert calls[0]["output"] is output
    assert adapter.prepare_session_calls == [spec]
    assert not adapter.create_runtime_called


def test_replay_demo_builds_output_target_from_spec(tmp_path: Path) -> None:
    writer_calls: list[dict[str, Any]] = []

    def fake_writer(
        video: torch.Tensor,
        path: Path,
        *,
        fps: float,
        layout: str,
        install_hint: str,
    ) -> Path:
        del install_hint
        writer_calls.append(
            {
                "shape": tuple(video.shape),
                "path": path,
                "fps": fps,
                "layout": layout,
            }
        )
        return path

    spec = DemoSpec(
        model_id="fake-demo",
        scenario="valid-scenario",
        input_mode="replay",
        output=Mp4OutputSpec(path=tmp_path / "demo.mp4", fps=12),
    )

    artifacts = run_replay_demo(
        spec=spec,
        adapter=_FakeDemoAdapter(video_output=True),
        output_target_factory=lambda output_spec: build_output_target(
            output_spec,
            mp4_writer=fake_writer,
        ),
    )

    assert len(artifacts) == 1
    assert artifacts[0].kind == "video/mp4"
    assert artifacts[0].uri == str(tmp_path / "demo.mp4")
    assert writer_calls == [
        {
            "shape": (2, 2, 2, 3),
            "path": tmp_path / "demo.mp4",
            "fps": 12,
            "layout": "thwc",
        }
    ]


def test_replay_demo_fails_before_runtime_creation_when_scenario_invalid() -> None:
    adapter = _FakeDemoAdapter(scenario_valid=False)
    output_factory_calls = 0

    def output_factory(output_spec: object) -> OutputTarget:
        nonlocal output_factory_calls
        del output_spec
        output_factory_calls += 1
        return NullOutputTarget()

    spec = DemoSpec(
        model_id="fake-demo",
        scenario="missing-scenario",
        input_mode="replay",
        output=NullOutputSpec(),
    )

    with pytest.raises(ValueError, match="invalid scenario"):
        run_replay_demo(
            spec=spec,
            adapter=adapter,
            output_target_factory=output_factory,
        )

    assert adapter.prepare_session_calls == [spec]
    assert not adapter.create_runtime_called
    assert output_factory_calls == 0


def test_demo_adapter_declares_supported_routes() -> None:
    adapter = _FakeDemoAdapter(
        routes=(
            DemoRoute(input_mode="replay", output_mode="null"),
            DemoRoute(input_mode="replay", output_mode="mp4"),
            DemoRoute(input_mode="replay", output_mode="webrtc"),
        ),
    )

    assert adapter.supported_routes() == (
        DemoRoute(input_mode="replay", output_mode="null"),
        DemoRoute(input_mode="replay", output_mode="mp4"),
        DemoRoute(input_mode="replay", output_mode="webrtc"),
    )

    with pytest.raises(ValueError, match="Unsupported demo route"):
        run_replay_demo(
            spec=DemoSpec(
                model_id="fake-demo",
                scenario="valid-scenario",
                input_mode="keyboard-driving",
                output=NullOutputSpec(),
            ),
            adapter=adapter,
        )

    assert adapter.prepare_session_calls == []
    assert not adapter.create_runtime_called


def test_webrtc_demo_uses_existing_session_manager_with_adapter_runtime() -> None:
    adapter = _FakeDemoAdapter(video_output=True)
    spec = DemoSpec(
        model_id="fake-demo",
        scenario="valid-scenario",
        input_mode="keyboard-driving",
        output=WebRTCOutputSpec(
            host="0.0.0.0",
            port=8082,
            fps=24,
            video_width=16,
            video_height=8,
            warmup_chunks=0,
            warmup_timeout_s=1.0,
        ),
    )

    demo = build_webrtc_demo(spec=spec, adapter=adapter)

    assert isinstance(demo.session_manager, BaseWebRTCSessionManager)
    assert demo.runtime is adapter.webrtc_runtime
    assert demo.session_manager._runtime is adapter.webrtc_runtime
    assert demo.session_manager.runtime_config.video_width == 16
    assert demo.session_manager.runtime_config.video_height == 8
    assert demo.session_manager.fps == 24
    assert demo.session_manager._model_name() == "fake-demo"
    assert demo.app is None
    assert demo.host == "0.0.0.0"
    assert demo.port == 8082
    assert adapter.create_webrtc_runtime_calls == [spec]
    assert not adapter.create_runtime_called


def test_model_owned_local_window_app_may_own_custom_output() -> None:
    adapter = _FakeLocalWindowAdapter(video_output=False)
    spec = DemoSpec(
        model_id="fake-demo",
        scenario="valid-scenario",
        input_mode="keyboard-driving",
        output=LocalWindowOutputSpec(),
    )

    demo = build_local_window_demo(spec=spec, adapter=adapter)

    assert demo.app is adapter.local_window_app
    assert adapter.local_window_specs == [spec]


def test_local_window_demo_builds_a_typed_model_app_factory() -> None:
    adapter = _FakeLocalWindowAdapter(video_output=True)
    spec = DemoSpec(
        model_id="fake-demo",
        scenario="valid-scenario",
        input_mode="keyboard-driving",
        output=LocalWindowOutputSpec(),
    )

    demo = build_local_window_demo(spec=spec, adapter=adapter)

    assert demo.app is adapter.local_window_app
    assert adapter.local_window_specs == [spec]


def test_plug_compatible_local_window_uses_standard_runtime_and_video_target() -> None:
    adapter = _FakeLocalWindowAdapter(video_output=True)
    presenter = _FakePresenter()
    spec = DemoSpec(
        model_id="fake-demo",
        scenario="valid-scenario",
        input_mode="keyboard-driving",
        output=LocalWindowOutputSpec(),
    )
    io = build_local_window_io(
        spec=spec,
        overlay=NullOverlay(),
        presenter_factory=lambda **_: presenter,
    )

    artifacts = run_local_window_session(spec=spec, adapter=adapter, io=io)

    assert artifacts == ()
    assert len(presenter.presented) == 2
    assert adapter.create_runtime_called
    assert adapter.runtime is not None and adapter.runtime.closed
    assert presenter.closed


def test_driving_route_rejects_mapping_that_ignores_driver_command() -> None:
    adapter = _FakeLocalWindowAdapter(video_output=True)
    spec = DemoSpec(
        model_id="fake-demo",
        scenario="valid-scenario",
        input_mode="keyboard-driving",
        output=LocalWindowOutputSpec(),
    )

    with pytest.raises(ValueError, match="does not consume required"):
        run_local_window_session(
            spec=spec,
            adapter=adapter,
            required_modalities=(DRIVER_COMMAND,),
        )

    assert not adapter.create_runtime_called


def test_plug_route_reuses_runtime_and_window_across_scene_sessions() -> None:
    adapter = _FakeLocalWindowAdapter(video_output=True)
    presenter = _FakePresenter()
    spec = DemoSpec(
        model_id="fake-demo",
        scenario="scene-a",
        input_mode="keyboard-driving",
        output=LocalWindowOutputSpec(),
    )
    io = build_local_window_io(
        spec=spec,
        overlay=NullOverlay(),
        presenter_factory=lambda **_: presenter,
        close_presenter_on_close=False,
    )
    assert spec.config is not None
    runtime = adapter.create_runtime(spec.config)
    assert isinstance(runtime, _FakeRuntime)

    run_local_window_session(spec=spec, adapter=adapter, runtime=runtime, io=io)
    run_local_window_session(spec=spec, adapter=adapter, runtime=runtime, io=io)

    assert adapter.create_runtime_count == 1
    assert not runtime.closed
    assert len(presenter.presented) == 4
    assert not presenter.closed
    runtime.close()
    io.output.shutdown()
    assert presenter.closed


class _ChunkIndexMapping:
    mapping_schema = InputMappingSchema(
        name="chunk-index",
        produces_global_conditioning=(InputField(name="prompt"),),
        produces_step=(InputField(name="chunk_index"),),
    )

    def validate(
        self,
        *,
        canonical_schema: CanonicalInputSchema | None = None,
        inference_input_schema: InferenceInputSchema | None = None,
    ) -> None:
        del canonical_schema, inference_input_schema

    def map_global_conditioning_inputs(
        self,
        *,
        canonical_inputs: CanonicalInputs,
        inference_input: InferenceInput,
    ) -> InferenceInput:
        del canonical_inputs
        return inference_input

    def map_step_inputs(
        self,
        *,
        canonical_inputs: CanonicalInputs,
        inference_input: InferenceInput,
        request: StepRequest,
    ) -> InferenceInput:
        del canonical_inputs
        return InferenceInput(
            global_conditioning=inference_input.global_conditioning,
            step={"chunk_index": request.step_index},
            metadata=inference_input.metadata,
        )


class _FakeDemoAdapter:
    model_id = "fake-demo"
    inference_input_schema = InferenceInputSchema(
        global_conditioning_fields=(InputField(name="prompt"),),
        step_fields=(InputField(name="chunk_index"),),
    )
    canonical_input_schema = CanonicalInputSchema()

    def __init__(
        self,
        *,
        scenario_valid: bool = True,
        video_output: bool = False,
        routes: tuple[DemoRoute, ...] = (
            DemoRoute(input_mode="replay", output_mode="null"),
            DemoRoute(input_mode="replay", output_mode="mp4"),
            DemoRoute(input_mode="keyboard-driving", output_mode="webrtc"),
        ),
    ) -> None:
        self._scenario_valid = scenario_valid
        self._video_output = video_output
        self._routes = routes
        self.mapping = _ChunkIndexMapping()
        self.prepared_session = PreparedSession(
            initial_inputs=InferenceInput(
                global_conditioning={"prompt": "drive forward"},
            ),
            user_inputs=UserInputs(),
            source_schema=UserInputSchema(),
            canonicalizer=InputCanonicalizer(),
            mapping=self.mapping,
        )
        self.prepare_session_calls: list[DemoSpec] = []
        self.create_runtime_called = False
        self.create_runtime_count = 0
        self.runtime: _FakeRuntime | None = None
        self.webrtc_runtime: _FakeWebRTCRuntime | None = None
        self.create_webrtc_runtime_calls: list[DemoSpec] = []

    def supported_routes(self) -> tuple[DemoRoute, ...]:
        return self._routes

    @property
    def inference_output_schema(self) -> InferenceOutputSchema:
        if self._video_output:
            return InferenceOutputSchema(
                modality=RGB_VIDEO,
                python_type=VideoStepResult,
                layouts=frozenset({"bvtchw"}),
            )
        return InferenceOutputSchema(
            modality="text/plain",
            python_type=str,
        )

    def default_input_mapping(self) -> InputMapping:
        return IdentityInputMapping()

    def validate_config(self, config: InferenceConfig) -> None:
        if config.model_id != self.model_id:
            raise ValueError(f"Unsupported model_id={config.model_id!r}.")

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        self.create_runtime_called = True
        self.create_runtime_count += 1
        self.runtime = _FakeRuntime(
            config=config,
            inference_input_schema=self.inference_input_schema,
            video_output=self._video_output,
        )
        return self.runtime

    def create_demo_runtime(self, spec: DemoSpec) -> InferenceRuntime:
        assert spec.config is not None
        return self.create_runtime(spec.config)

    def list_sessions(self, spec: DemoSpec) -> tuple[DemoSpec, ...]:
        return (spec,)

    def prepare_session(self, spec: DemoSpec) -> PreparedSession:
        self.prepare_session_calls.append(spec)
        if not self._scenario_valid:
            raise ValueError("invalid scenario")
        return self.prepared_session

    def create_webrtc_runtime(self, spec: DemoSpec) -> _FakeWebRTCRuntime:
        self.create_webrtc_runtime_calls.append(spec)
        self.webrtc_runtime = _FakeWebRTCRuntime()
        return self.webrtc_runtime


class _FakeLocalWindowApp:
    def run(self) -> None:
        return


class _FakePresenter:
    def __init__(self) -> None:
        self.should_close = False
        self.presented: list[DisplayFrame] = []
        self.closed = False

    def process_events(self) -> None:
        return

    def prepare_frame(self, frame: DisplayFrame) -> None:
        del frame

    def present_frame(self, frame: DisplayFrame) -> None:
        self.presented.append(frame)

    def close(self) -> None:
        self.closed = True


class _FakeLocalWindowAdapter(_FakeDemoAdapter):
    def __init__(self, *, video_output: bool) -> None:
        super().__init__(
            video_output=video_output,
            routes=(
                DemoRoute(
                    input_mode="keyboard-driving",
                    output_mode="local-window",
                ),
            ),
        )
        self.local_window_app = _FakeLocalWindowApp()
        self.local_window_specs: list[DemoSpec] = []

    def create_local_window_app(self, *, spec: DemoSpec) -> _FakeLocalWindowApp:
        self.local_window_specs.append(spec)
        return self.local_window_app


class _FakeRuntime:
    def __init__(
        self,
        *,
        config: InferenceConfig,
        inference_input_schema: InferenceInputSchema,
        video_output: bool,
    ) -> None:
        self.config = config
        self._inference_input_schema = inference_input_schema
        self._video_output = video_output
        self.session: _FakeSession | None = None
        self.closed = False

    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        self._inference_input_schema.require_global_conditioning(inputs)
        self.session = _FakeSession(
            inference_input_schema=self._inference_input_schema,
            video_output=self._video_output,
        )
        return self.session

    def close(self) -> None:
        self.closed = True


class _FakeSession:
    def __init__(
        self,
        *,
        inference_input_schema: InferenceInputSchema,
        video_output: bool,
    ) -> None:
        self._inference_input_schema = inference_input_schema
        self._video_output = video_output
        self.step_index = 0
        self.closed = False

    def next_step_request(self) -> StepRequest | None:
        if self.step_index >= 2:
            return None
        return StepRequest(
            step_index=self.step_index,
            user_input_window=TimeWindow(
                start_s=0.5 * self.step_index,
                end_s=0.5 * (self.step_index + 1),
            ),
        )

    def step(self, inputs: InferenceInput) -> StepResult:
        self._inference_input_schema.require_step(inputs)
        output: object
        if self._video_output:
            output = VideoStepResult.from_video_chunk(
                chunk_index=self.step_index,
                video_chunk=torch.full(
                    (1, 1, 1, 3, 2, 2),
                    self.step_index,
                    dtype=torch.float32,
                ),
                layout="bvtchw",
            )
        else:
            output = f"chunk-{self.step_index}"
        result = StepResult(
            step_index=self.step_index,
            output=output,
            frame_count=1,
            output_window=TimeWindow(
                start_s=0.5 * self.step_index,
                end_s=0.5 * (self.step_index + 1),
            ),
        )
        self.step_index += 1
        return result

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        self.step_index = 0

    def close(self) -> None:
        self.closed = True


class _RecordingOutputTarget:
    def open(self) -> None:
        return None

    def write(self, result: StepResult) -> None:
        del result

    def close(self) -> Sequence[OutputArtifact]:
        return ()


class _FakeWebRTCRuntime:
    async def initialize(self) -> None:
        return None

    async def reset_for_new_session(self) -> None:
        return None

    def peek_steady_chunk_num_frames(self) -> int:
        return 1

    def peek_next_chunk_num_frames(self) -> int:
        return 1

    async def generate_chunk(
        self,
        *,
        segments: list[Any],
        frame_times: list[float],
    ) -> Any:
        del segments, frame_times
        return None

    async def close(self) -> None:
        return None

    def send_exit_signal(self) -> None:
        return None

    def wait_for_termination(self) -> None:
        return None
