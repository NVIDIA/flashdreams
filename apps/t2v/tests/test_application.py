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
from flashdreams.infra.results import StepResult
from flashdreams.infra.time import TimeWindow
from flashdreams.runtime import CanonicalInputSchema, CanonicalModality
from flashdreams.runtime.demo import application_runtime
from flashdreams.runtime.demo import drivers as driver_module

pytestmark = pytest.mark.ci_cpu


class _FakeDecoder:
    spatial_compression_ratio = 8


class _FakePipeline:
    def __init__(self) -> None:
        self.decoder = _FakeDecoder()
        self.device: str | None = None
        self.cache_kwargs: dict[str, Any] | None = None
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


def test_application_session_emits_canonical_video_results() -> None:
    pipeline = _FakePipeline()
    output_sink = NullOutputSink(store_results=True, store_outputs=True)
    application = _application(pipeline)
    application.init(
        ["--prompt", "A waterfall", "--total-blocks", "2", "--device", "cpu"]
    )
    result = application_runtime.run_batch_application_session(
        application=application,
        input_handler=NullInputHandler(),
        input_schema=application.input_schema,
        output_sink=output_sink,
    )

    assert result.status == "completed"
    assert result.metrics is not None
    assert result.metrics.counters["steps"] == 2
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


def test_application_session_honors_sink_stop_decision() -> None:
    pipeline = _FakePipeline()
    output_sink = _StoppingSink(store_outputs=True)
    application = _application(pipeline)
    application.init(
        ["--prompt", "A waterfall", "--total-blocks", "4", "--device", "cpu"]
    )
    result = application_runtime.run_batch_application_session(
        application=application,
        input_handler=NullInputHandler(),
        input_schema=application.input_schema,
        output_sink=output_sink,
    )

    assert pipeline.generated == [0]
    assert output_sink.output_count == 1
    assert result.metrics is not None
    assert result.metrics.counters["steps"] == 1


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


def test_batch_driver_preserves_result_contract() -> None:
    pipeline = _FakePipeline()
    application = _application(pipeline)
    application.init(
        ["--prompt", "A waterfall", "--total-blocks", "2", "--device", "cpu"]
    )

    result = application_runtime.run_batch_application_session(
        application=application,
        input_handler=NullInputHandler(),
        input_schema=application.input_schema,
        output_sink=NullOutputSink(store_results=True),
    )

    assert type(driver_module.BatchSessionDriver()).__name__ == "BatchSessionDriver"
    assert result.status == "completed"
    assert result.artifacts == ()
    assert result.metrics is not None
    assert result.metrics.counters["steps"] == 2
    assert result.reason is None
    assert result.error is None


@pytest.mark.asyncio
async def test_realtime_driver_preserves_result_contract() -> None:
    pipeline = _FakePipeline()
    application = _application(pipeline)
    application.init(
        ["--prompt", "A waterfall", "--total-blocks", "2", "--device", "cpu"]
    )

    driver = driver_module.RealtimeSessionDriver()
    result = await application_runtime.run_realtime_application_session(
        application=application,
        input_handler=NullInputHandler(),
        input_schema=application.input_schema,
        output_sink=NullOutputSink(store_results=True),
    )

    assert type(driver).__name__ == "RealtimeSessionDriver"
    assert result.status == "completed"
    assert result.artifacts == ()
    assert result.metrics is not None
    assert result.metrics.counters["steps"] == 2
    assert result.reason is None
    assert result.error is None


@pytest.mark.asyncio
async def test_realtime_driver_awaits_output_backpressure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _PacingSink(NullOutputSink):
        def write(self, result: StepResult) -> OutputDecision:
            decision_index = self.output_count
            super().write(result)
            if decision_index == 0:
                return OutputDecision(backpressure_s=0.25)
            return OutputDecision(should_stop=True)

    delays: list[float] = []

    async def record_sleep(delay_s: float) -> None:
        delays.append(delay_s)

    monkeypatch.setattr(application_runtime.asyncio, "sleep", record_sleep)
    pipeline = _FakePipeline()
    application = _application(pipeline)
    application.init(
        ["--prompt", "A waterfall", "--total-blocks", "2", "--device", "cpu"]
    )
    sink = _PacingSink(store_results=True)

    result = await application_runtime.run_realtime_application_session(
        application=application,
        input_handler=NullInputHandler(),
        input_schema=application.input_schema,
        output_sink=sink,
    )

    assert result.status == "completed"
    assert sink.output_count == 2
    assert delays == pytest.approx([0.0, 0.25, 0.0])


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
    inputs = application_runtime._current_application_inputs(
        _NamedInputHandler(), schema
    )

    assert inputs.values == {"camera": {"yaw": 0.25, "pitch": -0.5}}
    assert inputs.window == TimeWindow(start_s=1.0, end_s=2.0)


def test_application_host_rejects_unwindowed_canonical_inputs() -> None:
    class _LegacyInputHandler(_NamedInputHandler):
        def current_inputs(self) -> CanonicalInputWindow:
            return cast(CanonicalInputWindow, CanonicalInputs())

    with pytest.raises(TypeError, match="CanonicalInputWindow"):
        application_runtime._current_application_inputs(
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


def test_application_driver_enforces_one_model_thread() -> None:
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
    application.init(
        ["--prompt", "A waterfall", "--total-blocks", "2", "--device", "cpu"]
    )
    calling_thread = threading.get_ident()

    result = application_runtime.run_batch_application_session(
        application=application,
        input_handler=NullInputHandler(),
        input_schema=application.input_schema,
        output_sink=NullOutputSink(),
    )

    assert result.status == "completed"
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
