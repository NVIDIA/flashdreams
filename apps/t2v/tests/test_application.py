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

import inspect
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from t2v import (
    T2VApplication,
    T2VApplicationDefaults,
)

from flashdreams.demo import (
    CanonicalInputs,
    CanonicalInputWindow,
    Mp4OutputSink,
    NullInputHandler,
    NullOutputSink,
    OutputDecision,
    ProvidedIOFactory,
    SessionInfo,
)
from flashdreams.demo import application as application_module
from flashdreams.demo.bridge import (
    ApplicationDemoAdapter,
    application_demo_spec,
    application_scenario,
    current_application_inputs,
)
from flashdreams.infra.results import StepResult
from flashdreams.infra.time import TimeWindow
from flashdreams.runtime import (
    CanonicalInputSchema,
    CanonicalModality,
    StepRequirements,
)
from flashdreams.runtime.demo import (
    RuntimeHost,
    WebRTCOutputSpec,
    build_model_warmup_plan,
)
from flashdreams.runtime.demo import drivers as driver_module
from flashdreams.runtime.inputs import InferenceInput

pytestmark = pytest.mark.ci_cpu


class _FakeDecoder:
    spatial_compression_ratio = 8


class _FakePipeline:
    def __init__(self) -> None:
        self.decoder = _FakeDecoder()
        self.device: str | None = None
        self.cache_kwargs: dict[str, Any] | None = None
        self.cache_initializations: list[dict[str, Any]] = []
        self.setup_count = 0
        self.generated: list[int] = []
        self.finalized: list[int] = []
        self.closed = False

    def to(self, device: str) -> "_FakePipeline":
        self.device = device
        return self

    def eval(self) -> "_FakePipeline":
        return self

    def initialize_cache(self, **kwargs: Any) -> object:
        self.cache_kwargs = kwargs
        self.cache_initializations.append(kwargs)
        return object()

    def get_num_output_frames(self, autoregressive_index: int) -> int:
        del autoregressive_index
        return 2

    def generate(
        self,
        *,
        autoregressive_index: int,
        cache: object,
    ) -> torch.Tensor:
        del cache
        self.generated.append(autoregressive_index)
        return torch.full((2, 3, 4, 5), float(autoregressive_index))

    def finalize(
        self,
        *,
        autoregressive_index: int,
        cache: object,
    ) -> dict[str, float]:
        del cache
        self.finalized.append(autoregressive_index)
        return {"model_step_s": 0.25}

    def close(self) -> None:
        self.closed = True


class _FakePipelineConfig:
    def __init__(self, pipeline: _FakePipeline) -> None:
        self.pipeline = pipeline

    def setup(self) -> _FakePipeline:
        self.pipeline.setup_count += 1
        return self.pipeline


class _StoppingSink(NullOutputSink):
    def write(self, result: StepResult) -> OutputDecision:
        super().write(result)
        return OutputDecision(should_stop=True)


def _application(pipeline: _FakePipeline) -> T2VApplication:
    return T2VApplication(
        defaults=T2VApplicationDefaults(
            pipeline_config=_FakePipelineConfig(pipeline),
            total_blocks=4,
            pixel_height=480,
            pixel_width=832,
        )
    )


def test_prompt_is_required() -> None:
    application = _application(_FakePipeline())
    with pytest.raises(ValueError, match="--prompt is required"):
        application.init([])


def test_compile_override_updates_single_transformer_config() -> None:
    application = _application(_FakePipeline())
    pipeline_config = SimpleNamespace(
        diffusion_model=SimpleNamespace(
            transformer=SimpleNamespace(compile_network=True)
        )
    )

    resolved = application._apply_compile_override(pipeline_config, False)

    assert resolved.diffusion_model.transformer.compile_network is False
    assert pipeline_config.diffusion_model.transformer.compile_network is True


def test_total_block_validation_can_be_narrowed_by_an_integration() -> None:
    class _SingleBlockApplication(T2VApplication):
        def _validate_total_blocks(self, total_blocks: int) -> None:
            super()._validate_total_blocks(total_blocks)
            if total_blocks > 1:
                raise ValueError("test application supports exactly one block")

    application = _SingleBlockApplication(
        defaults=T2VApplicationDefaults(
            pipeline_config=_FakePipelineConfig(_FakePipeline()),
            total_blocks=1,
            pixel_height=480,
            pixel_width=832,
        )
    )
    application.init(["--prompt", "A waterfall", "--total-blocks", "1"])

    with pytest.raises(ValueError, match="greater than zero"):
        application.init(["--prompt", "A waterfall", "--total-blocks", "0"])
    with pytest.raises(ValueError, match="exactly one block"):
        application.init(["--prompt", "A waterfall", "--total-blocks", "2"])


def test_t2v_model_warmup_covers_observed_autoregressive_signatures() -> None:
    pipeline = _FakePipeline()
    application = _application(pipeline)
    application.init(
        [
            "--prompt",
            "A waterfall",
            "--total-blocks",
            "10",
            "--device",
            "cpu",
        ]
    )
    spec = application_demo_spec(
        app=application,
        application_slug="t2v-fake",
        output=WebRTCOutputSpec(),
    )
    scenario = application_scenario(application, realtime=True)
    adapter = ApplicationDemoAdapter(
        app=application,
        spec=spec,
        scenario=scenario,
    )
    config = spec.config
    assert config is not None
    host = RuntimeHost(adapter.create_runtime(config))
    try:
        plan = build_model_warmup_plan(
            host=host,
            adapter=adapter,
            spec=spec,
            scenario=scenario,
        )
        assert len(plan.sessions) == 1
        assert len(plan.sessions[0].step_inputs) == 7

        host.preload()
        host.warmup(plan)
    finally:
        host.close()

    assert pipeline.device == "cpu"
    assert pipeline.cache_kwargs == {
        "text": ["A waterfall"],
        "image": None,
        "height": 60,
        "width": 104,
    }
    assert pipeline.generated == list(range(7))
    assert pipeline.finalized == list(range(7))
    assert pipeline.closed


def test_t2v_pipeline_is_reused_across_warmup_reset_and_sessions() -> None:
    pipeline = _FakePipeline()
    application = _application(pipeline)
    application.init(
        [
            "--prompt",
            "A waterfall",
            "--total-blocks",
            "10",
            "--device",
            "cpu",
        ]
    )
    spec = application_demo_spec(
        app=application,
        application_slug="t2v-fake",
        output=WebRTCOutputSpec(),
    )
    scenario = application_scenario(application, realtime=True)
    adapter = ApplicationDemoAdapter(app=application, spec=spec, scenario=scenario)
    config = spec.config
    assert config is not None
    host = RuntimeHost(adapter.create_runtime(config))
    try:
        plan = build_model_warmup_plan(
            host=host,
            adapter=adapter,
            spec=spec,
            scenario=scenario,
        )
        step_input = plan.sessions[0].step_inputs[0]
        host.preload()
        host.warmup(plan)

        first = host.call(host.start_session, InferenceInput())
        host.call(first.step, step_input)
        host.call(first.reset)
        host.call(first.step, step_input)
        host.call(first.close)

        second = host.call(host.start_session, InferenceInput())
        host.call(second.step, step_input)
        host.call(second.close)

        assert pipeline.setup_count == 1
        assert len(pipeline.cache_initializations) == 4
        assert pipeline.generated == [*range(7), 0, 0, 0]
        assert not pipeline.closed
    finally:
        host.close()

    assert pipeline.closed


def test_application_session_emits_canonical_video_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _FakePipeline()
    output_sink = NullOutputSink(store_results=True, store_outputs=True)
    application = _application(pipeline)
    monkeypatch.setattr(
        application_module,
        "create_application",
        lambda _slug: (application, []),
    )
    artifacts = application_module.run_application(
        "t2v-fake",
        ["--prompt", "A waterfall", "--total-blocks", "2", "--device", "cpu"],
        io_factory=ProvidedIOFactory(NullInputHandler(), output_sink),
    )

    assert artifacts == ()
    assert pipeline.device == "cpu"
    assert pipeline.cache_kwargs == {
        "text": ["A waterfall"],
        "image": None,
        "height": 60,
        "width": 104,
    }
    assert pipeline.generated == [0, 1]
    assert pipeline.finalized == [0, 1]
    assert [tuple(output.shape) for output in output_sink.outputs] == [
        (2, 3, 4, 5),
        (2, 3, 4, 5),
    ]
    assert [record["layout"] for record in output_sink.results] == ["tchw", "tchw"]
    assert [record["metrics"] for record in output_sink.results] == [{}, {}]
    assert output_sink.session_info is not None
    assert output_sink.session_info.frames_per_second == 16
    assert output_sink.session_info.video_width == 832
    assert output_sink.session_info.video_height == 480
    assert pipeline.closed


def test_application_session_paces_steps_to_output_frame_count() -> None:
    application = _application(_FakePipeline())
    application.init(["--prompt", "A waterfall", "--device", "cpu"])
    session = application.create_session()

    assert session.next_step_requirements() == StepRequirements(
        step_index=0,
        input_frame_count=2,
        steady_output_frame_count=2,
    )


def test_application_session_honors_sink_stop_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _FakePipeline()
    output_sink = _StoppingSink(store_outputs=True)
    application = _application(pipeline)
    monkeypatch.setattr(
        application_module,
        "create_application",
        lambda _slug: (application, []),
    )
    artifacts = application_module.run_application(
        "t2v-fake",
        ["--prompt", "A waterfall", "--total-blocks", "4", "--device", "cpu"],
        io_factory=ProvidedIOFactory(NullInputHandler(), output_sink),
    )

    assert artifacts == ()
    assert pipeline.generated == [0]
    assert output_sink.output_count == 1


def test_application_session_does_not_own_driver_runtime_state() -> None:
    assert not hasattr(
        application_module.IFlashDreamsApplicationSession,
        "session_metrics",
    )
    assert not hasattr(application_module.IFlashDreamsApplicationSession, "_step")


def test_application_drivers_keep_shared_runtime_contract() -> None:
    expected = ("self", "host", "provider", "session_edges", "pipeline")

    assert (
        tuple(
            inspect.signature(
                driver_module.BatchSessionDriver.run_one_session
            ).parameters
        )
        == expected
    )
    assert (
        tuple(
            inspect.signature(
                driver_module.RealtimeSessionDriver.run_one_session
            ).parameters
        )
        == expected
    )


def test_batch_driver_preserves_public_application_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _FakePipeline()
    application = _application(pipeline)
    monkeypatch.setattr(
        application_module,
        "create_application",
        lambda _slug: (application, []),
    )

    artifacts = application_module.run_application(
        "t2v-fake",
        ["--prompt", "A waterfall", "--total-blocks", "2", "--device", "cpu"],
        io_factory=ProvidedIOFactory(
            NullInputHandler(),
            NullOutputSink(store_results=True),
        ),
    )

    assert type(driver_module.BatchSessionDriver()).__name__ == "BatchSessionDriver"
    assert artifacts == ()
    assert pipeline.generated == [0, 1]


class _NamedInputHandler:
    def __init__(self) -> None:
        self.inputs = CanonicalInputWindow(
            values={"camera": {"yaw": 0.25, "pitch": -0.5}},
            window=TimeWindow(start_s=1.0, end_s=2.0),
        )

    def open(self, session_info: object) -> None:
        del session_info

    def current_inputs(self) -> CanonicalInputWindow:
        return self.inputs

    def close(self) -> None:
        return


def test_input_handler_provides_schema_named_canonical_inputs() -> None:
    schema = CanonicalInputSchema(
        modalities=(
            CanonicalModality(
                name="camera",
                payload_fields=frozenset({"yaw", "pitch"}),
            ),
        )
    )
    inputs = current_application_inputs(_NamedInputHandler(), schema)

    assert inputs.values == {"camera": {"yaw": 0.25, "pitch": -0.5}}
    assert inputs.window == TimeWindow(start_s=1.0, end_s=2.0)


def test_application_host_rejects_unwindowed_canonical_inputs() -> None:
    class _LegacyInputHandler(_NamedInputHandler):
        def current_inputs(self) -> CanonicalInputWindow:
            return cast(CanonicalInputWindow, CanonicalInputs())

    with pytest.raises(TypeError, match="CanonicalInputWindow"):
        current_application_inputs(
            _LegacyInputHandler(),
            CanonicalInputSchema(),
        )


def test_null_input_handler_provides_contiguous_windows() -> None:
    now = [10.0]
    handler = NullInputHandler(clock=lambda: now[0])
    handler.open(SessionInfo())
    first = handler.current_inputs()
    now[0] = 10.25
    second = handler.current_inputs()

    assert first.window == TimeWindow(start_s=0.0, end_s=0.0)
    assert second.window == TimeWindow(start_s=0.0, end_s=0.25)


def test_application_host_writes_mp4_through_shared_io_factory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pipeline = _FakePipeline()
    application = _application(pipeline)
    monkeypatch.setattr(
        application_module,
        "create_application",
        lambda _slug: (application, []),
    )
    writer_calls: list[dict[str, object]] = []

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

    input_handler = NullInputHandler()
    output_sink = Mp4OutputSink(
        output_path=tmp_path / "out.mp4",
        output_layout="tchw",
        writer=fake_writer,
        move_to_cpu=False,
    )
    artifacts = application_module.run_application(
        "t2v-fake",
        ["--prompt", "A waterfall", "--total-blocks", "2", "--device", "cpu"],
        io_factory=ProvidedIOFactory(input_handler, output_sink),
    )

    assert writer_calls == [
        {
            "shape": (4, 3, 4, 5),
            "path": tmp_path / "out.mp4",
            "fps": 16,
            "layout": "tchw",
        }
    ]
    assert len(artifacts) == 1
    assert artifacts[0].kind == "video/mp4"
    assert artifacts[0].uri == str(tmp_path / "out.mp4")


def test_application_driver_enforces_one_model_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_threads: list[int] = []

    class _ThreadRecordingPipeline(_FakePipeline):
        def to(self, device: str) -> "_ThreadRecordingPipeline":
            call_threads.append(threading.get_ident())
            super().to(device)
            return self

        def initialize_cache(self, **kwargs: Any) -> object:
            call_threads.append(threading.get_ident())
            return super().initialize_cache(**kwargs)

        def generate(
            self,
            *,
            autoregressive_index: int,
            cache: object,
        ) -> torch.Tensor:
            call_threads.append(threading.get_ident())
            return super().generate(
                autoregressive_index=autoregressive_index,
                cache=cache,
            )

        def finalize(
            self,
            *,
            autoregressive_index: int,
            cache: object,
        ) -> dict[str, float]:
            call_threads.append(threading.get_ident())
            return super().finalize(
                autoregressive_index=autoregressive_index,
                cache=cache,
            )

        def close(self) -> None:
            call_threads.append(threading.get_ident())
            super().close()

    pipeline = _ThreadRecordingPipeline()
    application = _application(pipeline)
    monkeypatch.setattr(
        application_module,
        "create_application",
        lambda _slug: (application, []),
    )
    calling_thread = threading.get_ident()

    artifacts = application_module.run_application(
        "t2v-fake",
        ["--prompt", "A waterfall", "--total-blocks", "2", "--device", "cpu"],
        io_factory=ProvidedIOFactory(NullInputHandler(), NullOutputSink()),
    )

    assert artifacts == ()
    assert len(set(call_threads)) == 1
    assert call_threads[0] != calling_thread


def test_application_defaults_derive_from_runner_config() -> None:
    pipeline_config = object()
    runner_config = type(
        "RunnerConfig",
        (),
        {
            "pipeline": pipeline_config,
            "total_blocks": 7,
            "pixel_height": 360,
            "pixel_width": 640,
            "fps": 24,
            "postprocess_output_layout": "cthw",
        },
    )()

    defaults = T2VApplicationDefaults.from_runner_config(runner_config)

    assert defaults.pipeline_config is pipeline_config
    assert defaults.total_blocks == 7
    assert defaults.pixel_height == 360
    assert defaults.pixel_width == 640
    assert defaults.fps == 24
    assert defaults.output_layout == "cthw"


def test_application_defaults_accept_explicit_total_blocks() -> None:
    pipeline_config = object()
    runner_config = type(
        "RunnerConfig",
        (),
        {
            "pipeline": pipeline_config,
            "pixel_height": 480,
            "pixel_width": 832,
        },
    )()

    defaults = T2VApplicationDefaults.from_runner_config(
        runner_config,
        total_blocks=1,
    )

    assert defaults.pipeline_config is pipeline_config
    assert defaults.total_blocks == 1
    assert defaults.pixel_height == 480
    assert defaults.pixel_width == 832
