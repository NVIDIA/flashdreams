# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from flashdreams.runtime import (
    IdentityInputMapping,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    InferenceRuntime,
    InferenceSession,
    InputMapping,
    StepRequest,
    StepRequirements,
    StepResult,
    UserInputSchema,
)
from flashdreams.runtime.demo import (
    BATCH_INPUT_FPS_METADATA_KEY,
    BATCH_INPUT_FRAME_START_METADATA_KEY,
    BenchmarkRunMode,
    BenchmarkStatsOutputSink,
    DemoSpec,
    ErrorAction,
    InMemorySessionMetricsRecorder,
    ModelWarmupPlan,
    NullOutputSpec,
    PreparedScenario,
    PreparedStep,
    ProviderCapabilities,
    RunResult,
    RuntimeHost,
    SessionInfo,
    StepPipeline,
    UserInputWindow,
    WarmupSessionInputs,
    run_benchmark_demo,
    run_demo_session,
    run_replay_demo,
)

pytestmark = pytest.mark.ci_cpu


def test_benchmark_run_mode_excludes_warmup_from_stats_artifact(
    tmp_path: Path,
) -> None:
    warmup_session = _BenchmarkSession(num_steps=0, name="warmup")
    measured_session = _BenchmarkSession(num_steps=2, name="measured")
    runtime = _BenchmarkRuntime(sessions=(warmup_session, measured_session))
    adapter = _BenchmarkAdapter(runtime=runtime, warmup_steps=1)
    spec = _spec("measured")

    result = run_benchmark_demo(
        spec=spec,
        adapter=adapter,
        stats_dir=tmp_path,
        capture_output=False,
    )

    assert result.status == "completed"
    assert result.metrics is not None
    assert result.metrics.counters["steps"] == 2
    assert [session.name for session in runtime.started_sessions] == [
        "warmup",
        "measured",
    ]
    assert len(warmup_session.step_inputs) == 1
    assert len(measured_session.step_inputs) == 2

    stats_path = tmp_path / "stats_measured.json"
    assert result.artifacts[0].uri == str(stats_path)
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    assert [sample["step_index"] for sample in payload["samples"]] == [0, 1]
    assert [sample["name"] for sample in payload["samples"]] == [
        "model_step_s",
        "model_step_s",
    ]
    assert payload["steps"] == [
        {
            "frame_count": 1,
            "metadata": {},
            "metrics": {"model_step_s": 0.01},
            "sample_count": 1,
            "step_index": 0,
        },
        {
            "frame_count": 1,
            "metadata": {},
            "metrics": {"model_step_s": 0.02},
            "sample_count": 1,
            "step_index": 1,
        },
    ]


def test_benchmark_loop_can_continue_after_failed_scenario(
    tmp_path: Path,
) -> None:
    runtime = _BenchmarkRuntime(
        sessions=(
            _BenchmarkSession(num_steps=1, name="failed", fail_step=0),
            _BenchmarkSession(num_steps=1, name="success"),
        )
    )
    adapter = _BenchmarkAdapter(runtime=runtime)
    mode = BenchmarkRunMode(stats_dir=tmp_path)
    context = mode.create_run_context(
        spec=_spec("failed"),
        adapter=adapter,
        host=RuntimeHost(runtime),
        model_warmup_plan=ModelWarmupPlan(),
    )
    specs = (_spec("failed"), _spec("success"))
    results: list[str] = []

    try:
        for spec in specs:
            mode.validate_run(spec=spec, adapter=adapter)
            scenario = adapter.prepare_scenario(spec)
            result = run_demo_session(
                context=context,
                spec=spec,
                scenario=scenario,
                adapter=adapter,
                run_mode=mode,
                pipeline=StepPipeline(),
            )
            results.append(result.status)
            if not mode.should_continue_after(result):
                break

        assert results == ["failed", "completed"]
        run_metrics = context.run_metrics
        assert isinstance(run_metrics, InMemorySessionMetricsRecorder)
        session_results = [
            result for result in run_metrics.sessions if isinstance(result, RunResult)
        ]
        assert session_results == run_metrics.sessions
        assert [result.status for result in session_results] == results
        assert (
            json.loads((tmp_path / "stats_failed.json").read_text(encoding="utf-8"))[
                "samples"
            ]
            == []
        )
        assert (
            json.loads((tmp_path / "stats_success.json").read_text(encoding="utf-8"))[
                "samples"
            ][0]["step_index"]
            == 0
        )
    finally:
        context.close()
        context.host.close()


def test_benchmark_run_mode_can_mark_setup_failure_skipped(
    tmp_path: Path,
) -> None:
    runtime = _BenchmarkRuntime(sessions=())
    adapter = _BenchmarkAdapter(
        runtime=runtime,
        providers=(
            _BenchmarkProvider(fail_initial=RuntimeError("assets unavailable")),
        ),
    )
    mode = BenchmarkRunMode(stats_dir=tmp_path, error_policy=_SkipSetupPolicy())

    result = _run_one_benchmark_session(
        mode=mode,
        spec=_spec("skipped"),
        adapter=adapter,
        runtime=runtime,
    )

    assert result.status == "skipped"
    assert result.reason == "assets unavailable"
    assert result.error is None
    assert mode.should_continue_after(result)
    payload = json.loads((tmp_path / "stats_skipped.json").read_text(encoding="utf-8"))
    assert payload["steps"] == []
    assert payload["samples"] == []


def test_benchmark_and_replay_use_same_timed_input_windows(
    tmp_path: Path,
) -> None:
    scenario = PreparedScenario(
        initial_inputs=InferenceInput(),
        metadata={
            BATCH_INPUT_FPS_METADATA_KEY: 2,
            "scenario_id": "windowed",
        },
    )
    spec = _spec("windowed")
    replay_stats = tmp_path / "replay_stats.json"
    benchmark_stats = tmp_path / "benchmark_stats.json"

    replay_result = run_replay_demo(
        spec=spec,
        adapter=_BenchmarkAdapter(
            runtime=_BenchmarkRuntime(
                sessions=(
                    _WindowedBenchmarkSession(
                        num_steps=2,
                        name="replay",
                        record_input_window=True,
                    ),
                )
            ),
            prepared_scenario=scenario,
        ),
        output_sink_factory=lambda output: BenchmarkStatsOutputSink(
            output_path=replay_stats
        ),
    )
    benchmark_result = run_benchmark_demo(
        spec=spec,
        adapter=_BenchmarkAdapter(
            runtime=_BenchmarkRuntime(
                sessions=(
                    _WindowedBenchmarkSession(
                        num_steps=2,
                        name="benchmark",
                        record_input_window=True,
                    ),
                )
            ),
            prepared_scenario=scenario,
        ),
        stats_path=benchmark_stats,
        capture_output=False,
    )

    assert replay_result.status == "completed"
    assert benchmark_result.status == "completed"
    replay_windows = _step_metadata_windows(replay_stats)
    benchmark_windows = _step_metadata_windows(benchmark_stats)
    assert replay_windows == benchmark_windows == [(0.0, 0.5), (0.5, 1.0)]


def _run_one_benchmark_session(
    *,
    mode: BenchmarkRunMode,
    spec: DemoSpec,
    adapter: "_BenchmarkAdapter",
    runtime: "_BenchmarkRuntime",
) -> RunResult:
    mode.validate_run(spec=spec, adapter=adapter)
    scenario = adapter.prepare_scenario(spec)
    context = mode.create_run_context(
        spec=spec,
        adapter=adapter,
        host=RuntimeHost(runtime),
        model_warmup_plan=ModelWarmupPlan(),
    )
    try:
        return run_demo_session(
            context=context,
            spec=spec,
            scenario=scenario,
            adapter=adapter,
            run_mode=mode,
            pipeline=StepPipeline(),
        )
    finally:
        context.close()
        context.host.close()


def _spec(scenario: str) -> DemoSpec:
    return DemoSpec(
        model_id="fake-benchmark",
        input_mode="replay",
        output=NullOutputSpec(),
        scenario=scenario,
        config=InferenceConfig(model_id="fake-benchmark", seed=123),
    )


class _BenchmarkAdapter:
    model_id = "fake-benchmark"
    inference_input_schema = InferenceInputSchema()
    canonical_input_schema = None

    def __init__(
        self,
        *,
        runtime: "_BenchmarkRuntime",
        warmup_steps: int = 0,
        providers: Sequence["_BenchmarkProvider"] = (),
        prepared_scenario: PreparedScenario | None = None,
    ) -> None:
        self.runtime = runtime
        self.warmup_steps = warmup_steps
        self._providers = list(providers)
        self.created_providers: list[_BenchmarkProvider] = []
        self.prepared_scenario = prepared_scenario

    def supported_input_modes(self) -> tuple[str, ...]:
        return ("replay",)

    def supported_output_modes(self) -> tuple[str, ...]:
        return ("null",)

    def default_input_mapping(self) -> InputMapping:
        return IdentityInputMapping()

    def validate_config(self, config: InferenceConfig) -> None:
        if config.model_id != self.model_id:
            raise ValueError(f"Unsupported model_id={config.model_id!r}.")

    def create_runtime(self, config: InferenceConfig) -> InferenceRuntime:
        self.validate_config(config)
        return self.runtime

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        if self.prepared_scenario is not None:
            return self.prepared_scenario
        return PreparedScenario(
            initial_inputs=InferenceInput(),
            metadata={"scenario_id": str(spec.scenario)},
        )

    def create_model_warmup_sessions(
        self,
        spec: DemoSpec,
        scenario: PreparedScenario,
    ) -> tuple[WarmupSessionInputs, ...]:
        del spec, scenario
        if self.warmup_steps <= 0:
            return ()
        return (
            WarmupSessionInputs(
                initial_input=InferenceInput(global_conditioning={"phase": "warmup"}),
                step_inputs=tuple(
                    InferenceInput(step={"phase": "warmup", "step": step_index})
                    for step_index in range(self.warmup_steps)
                ),
            ),
        )

    def create_model_input_provider(
        self,
        spec: DemoSpec,
        scenario: PreparedScenario,
    ) -> "_BenchmarkProvider":
        del spec, scenario
        provider = self._providers.pop(0) if self._providers else _BenchmarkProvider()
        self.created_providers.append(provider)
        return provider


class _BenchmarkProvider:
    capabilities = ProviderCapabilities(
        supports_recorded_input=True,
        deterministic_given_inputs=True,
        user_input_schema=UserInputSchema(),
        inference_input_schema=InferenceInputSchema(),
    )

    def __init__(self, *, fail_initial: Exception | None = None) -> None:
        self.fail_initial = fail_initial
        self.close_count = 0

    def prepare_initial_input(self) -> InferenceInput:
        if self.fail_initial is not None:
            raise self.fail_initial
        return InferenceInput(global_conditioning={"provider": "benchmark"})

    def prepare_step(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
    ) -> PreparedStep:
        return PreparedStep(
            inference_input=InferenceInput(
                step={
                    "request_step": request.step_index,
                    "window": (user_window.start_s, user_window.end_s),
                }
            )
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs

    def close(self) -> None:
        self.close_count += 1


class _BenchmarkRuntime:
    def __init__(self, *, sessions: Sequence["_BenchmarkSession"]) -> None:
        self._sessions = list(sessions)
        self.started_sessions: list[_BenchmarkSession] = []
        self.close_count = 0

    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        del inputs
        session = self._sessions.pop(0)
        self.started_sessions.append(session)
        return session

    def close(self) -> None:
        self.close_count += 1


class _BenchmarkSession:
    def __init__(
        self,
        *,
        num_steps: int,
        name: str,
        fail_step: int | None = None,
        record_input_window: bool = False,
    ) -> None:
        self.num_steps = num_steps
        self.name = name
        self.fail_step = fail_step
        self.record_input_window = record_input_window
        self.next_request_index = 0
        self.step_inputs: list[InferenceInput] = []
        self.close_count = 0

    def session_info(self) -> SessionInfo:
        return SessionInfo(output_layout="fake", steady_output_frame_count=1)

    def next_step_requirements(self) -> StepRequirements | None:
        if self.next_request_index >= self.num_steps:
            return None
        request = StepRequirements(step_index=self.next_request_index)
        self.next_request_index += 1
        return request

    def next_step_request(self) -> StepRequest | None:
        raise AssertionError("benchmark driver should request StepRequirements")

    def step(self, inputs: InferenceInput) -> StepResult:
        step_index = len(self.step_inputs)
        if self.fail_step == step_index:
            raise RuntimeError(f"{self.name} step failed")
        self.step_inputs.append(inputs)
        metadata = {}
        if self.record_input_window:
            metadata["input_window"] = inputs.step["window"]
        return StepResult(
            step_index=step_index,
            output=f"{self.name}-{step_index}",
            frame_count=1,
            metrics={"model_step_s": 0.01 * (step_index + 1)},
            metadata=metadata,
        )

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs

    def close(self) -> None:
        self.close_count += 1


class _SkipSetupPolicy:
    def handle_setup_error(self, exc: Exception) -> ErrorAction:
        del exc
        return ErrorAction(result_status="skipped", continue_next_scenario=True)

    def handle(self, exc: Exception) -> ErrorAction:
        del exc
        return ErrorAction(result_status="failed", continue_next_scenario=True)


class _WindowedBenchmarkSession(_BenchmarkSession):
    def next_step_requirements(self) -> StepRequirements | None:
        if self.next_request_index >= self.num_steps:
            return None
        step_index = self.next_request_index
        self.next_request_index += 1
        return StepRequirements(
            step_index=step_index,
            input_frame_count=1,
            metadata={BATCH_INPUT_FRAME_START_METADATA_KEY: step_index},
        )


def _step_metadata_windows(path: Path) -> list[tuple[float, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        tuple(step["metadata"]["input_window"])
        for step in payload["steps"]
        if "input_window" in step["metadata"]
    ]
