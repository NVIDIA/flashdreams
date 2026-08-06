# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch

from flashdreams.infra.video_output import VideoStepResult
from flashdreams.runtime import (
    CanonicalInputs,
    CanonicalInputSchema,
    IdentityInputMapping,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
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
    DemoSpec,
    Mp4OutputSpec,
    NullOutputSpec,
    PreparedScenario,
    WebRTCOutputSpec,
    build_output_target,
    run_replay_demo,
)
from flashdreams.runtime.demo.webrtc import build_webrtc_demo
from flashdreams.serving.webrtc.manager import BaseWebRTCSessionManager

pytestmark = pytest.mark.ci_cpu


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
    assert calls[0]["mapping"] is adapter.prepared_scenario.mapping
    assert calls[0]["canonicalizer"] is adapter.prepared_scenario.canonicalizer
    assert calls[0]["source_schema"] is adapter.prepared_scenario.source_schema
    assert calls[0]["user_inputs"] is adapter.prepared_scenario.user_inputs
    assert calls[0]["initial_inputs"] is adapter.prepared_scenario.initial_inputs
    assert calls[0]["output"] is output
    assert adapter.prepare_scenario_calls == [spec]
    assert not adapter.create_runtime_called


def test_replay_demo_builds_output_target_from_spec(tmp_path: Path) -> None:
    writer_calls: list[dict[str, Any]] = []

    def fake_writer(
        video: torch.Tensor,
        path: Path,
        *,
        fps: int | float,
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

    assert adapter.prepare_scenario_calls == [spec]
    assert not adapter.create_runtime_called
    assert output_factory_calls == 0


def test_demo_adapter_declares_supported_modes() -> None:
    adapter = _FakeDemoAdapter(
        input_modes=("replay",),
        output_modes=("null", "mp4", "webrtc"),
    )

    assert adapter.supported_input_modes() == ("replay",)
    assert adapter.supported_output_modes() == ("null", "mp4", "webrtc")

    with pytest.raises(ValueError, match="input_mode='keyboard-driving'"):
        run_replay_demo(
            spec=DemoSpec(
                model_id="fake-demo",
                scenario="valid-scenario",
                input_mode="keyboard-driving",
                output=NullOutputSpec(),
            ),
            adapter=adapter,
        )

    assert adapter.prepare_scenario_calls == []
    assert not adapter.create_runtime_called


def test_webrtc_demo_uses_existing_session_manager_with_adapter_runtime() -> None:
    adapter = _FakeDemoAdapter()
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
        input_modes: tuple[str, ...] = ("replay", "keyboard-driving"),
        output_modes: tuple[str, ...] = ("null", "mp4", "webrtc"),
    ) -> None:
        self._scenario_valid = scenario_valid
        self._video_output = video_output
        self._input_modes = input_modes
        self._output_modes = output_modes
        self.mapping = _ChunkIndexMapping()
        self.prepared_scenario = PreparedScenario(
            initial_inputs=InferenceInput(
                global_conditioning={"prompt": "drive forward"},
            ),
            user_inputs=UserInputs(),
            source_schema=UserInputSchema(),
            canonicalizer=InputCanonicalizer(),
            mapping=self.mapping,
        )
        self.prepare_scenario_calls: list[DemoSpec] = []
        self.create_runtime_called = False
        self.runtime: _FakeRuntime | None = None
        self.webrtc_runtime: _FakeWebRTCRuntime | None = None
        self.create_webrtc_runtime_calls: list[DemoSpec] = []

    def supported_input_modes(self) -> tuple[str, ...]:
        return self._input_modes

    def supported_output_modes(self) -> tuple[str, ...]:
        return self._output_modes

    def default_input_mapping(self) -> InputMapping:
        return IdentityInputMapping()

    def validate_config(self, config: InferenceConfig) -> None:
        if config.model_id != self.model_id:
            raise ValueError(f"Unsupported model_id={config.model_id!r}.")

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        self.create_runtime_called = True
        self.runtime = _FakeRuntime(
            inference_input_schema=self.inference_input_schema,
            video_output=self._video_output,
        )
        return self.runtime

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        self.prepare_scenario_calls.append(spec)
        if not self._scenario_valid:
            raise ValueError("invalid scenario")
        return self.prepared_scenario

    def create_webrtc_runtime(self, spec: DemoSpec) -> "_FakeWebRTCRuntime":
        self.create_webrtc_runtime_calls.append(spec)
        self.webrtc_runtime = _FakeWebRTCRuntime()
        return self.webrtc_runtime


class _FakeRuntime:
    def __init__(
        self,
        *,
        inference_input_schema: InferenceInputSchema,
        video_output: bool,
    ) -> None:
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
