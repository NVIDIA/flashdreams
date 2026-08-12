# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark run mode for shared demo sessions."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from flashdreams.runtime.config import InferenceConfig
from flashdreams.runtime.inputs import TimeWindow, UserInputs
from flashdreams.runtime.metrics import MetricsRecorder
from flashdreams.runtime.video_output import VideoWriter

from .drivers import BatchSessionDriver, run_demo_session
from .host import ModelWarmupPlan, RuntimeHost
from .outputs import build_benchmark_output_sink
from .pipeline import StepPipeline
from .run_modes import (
    AdmissionPolicy,
    BenchmarkErrorPolicy,
    ErrorPolicy,
    InMemorySessionMetricsRecorder,
    RunContext,
    RunModeCapabilities,
    RunResult,
    SessionEdges,
    SessionMetricsRecorder,
    SingleSessionAdmissionPolicy,
    build_model_warmup_plan,
    warmup_run_context,
)
from .session_inputs import UserInputWindow
from .spec import (
    DemoAdapter,
    DemoSpec,
    PreparedScenario,
    WebRTCOutputSpec,
)

StatsPathFactory = Callable[[DemoSpec, PreparedScenario], str | Path]
SessionMetricsFactory = Callable[[], SessionMetricsRecorder]

_SAFE_STATS_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(slots=True)
class BenchmarkRunMode:
    """Finite benchmark run mode over the shared batch session driver."""

    stats_path: str | Path | None = None
    stats_dir: str | Path | None = None
    stats_path_factory: StatsPathFactory | None = None
    capture_output: bool = True
    mp4_writer: VideoWriter | None = None
    error_policy: ErrorPolicy = field(default_factory=BenchmarkErrorPolicy)
    run_metrics: SessionMetricsRecorder | None = None
    session_metrics_factory: SessionMetricsFactory = InMemorySessionMetricsRecorder
    admission: AdmissionPolicy | None = None
    name: str = "benchmark"
    capabilities: RunModeCapabilities = field(
        default_factory=lambda: RunModeCapabilities(
            requires_finite_input=True,
            supports_artifacts=True,
        )
    )

    def __post_init__(self) -> None:
        if self.stats_path is not None and (
            self.stats_dir is not None or self.stats_path_factory is not None
        ):
            raise ValueError(
                "BenchmarkRunMode.stats_path cannot be combined with stats_dir "
                "or stats_path_factory."
            )
        if self.stats_dir is not None and self.stats_path_factory is not None:
            raise ValueError(
                "BenchmarkRunMode.stats_dir cannot be combined with "
                "stats_path_factory."
            )
        if self.stats_path is not None:
            self.stats_path = Path(self.stats_path)
        if self.stats_dir is not None:
            self.stats_dir = Path(self.stats_dir)

    def validate_run(self, *, spec: DemoSpec, adapter: DemoAdapter) -> None:
        _require_supported_mode(
            mode=spec.input_mode,
            supported=adapter.supported_input_modes(),
            label="input_mode",
        )
        if isinstance(spec.output, WebRTCOutputSpec):
            raise ValueError("BenchmarkRunMode does not support WebRTC output.")
        if self.capture_output:
            _require_supported_mode(
                mode=spec.output.mode,
                supported=adapter.supported_output_modes(),
                label="output.mode",
            )
        adapter.validate_config(_require_config(spec))

    def validate_session(
        self,
        *,
        spec: DemoSpec,
        scenario: PreparedScenario,
        adapter: DemoAdapter,
        provider: object,
    ) -> None:
        del spec, scenario, adapter, provider

    def create_run_context(
        self,
        *,
        spec: DemoSpec,
        adapter: DemoAdapter,
        host: RuntimeHost,
        model_warmup_plan: ModelWarmupPlan,
    ) -> RunContext:
        del spec, adapter
        admission = self.admission or SingleSessionAdmissionPolicy(
            health_check=lambda: host.is_healthy
        )
        return RunContext(
            host=host,
            run_metrics=self.run_metrics or InMemorySessionMetricsRecorder(),
            admission=admission,
            model_warmup_plan=model_warmup_plan,
        )

    def create_session_edges(
        self,
        *,
        context: RunContext,
        spec: DemoSpec,
        scenario: PreparedScenario,
        provider: object,
        adapter: DemoAdapter,
    ) -> SessionEdges:
        del provider, adapter
        output = spec.output if self.capture_output else None
        return SessionEdges(
            input_source=BenchmarkBatchInputSource(scenario=scenario),
            output_sink=build_benchmark_output_sink(
                output,
                stats_path=self.stats_path_for(spec=spec, scenario=scenario),
                mp4_writer=self.mp4_writer,
            ),
            cleanup_tasks=context.cleanup_tasks,
            metrics=self.session_metrics_factory(),
            error_policy=self.error_policy,
        )

    def select_driver(self) -> BatchSessionDriver:
        return BatchSessionDriver()

    def stats_path_for(self, *, spec: DemoSpec, scenario: PreparedScenario) -> Path:
        if self.stats_path_factory is not None:
            return Path(self.stats_path_factory(spec, scenario))
        if self.stats_path is not None:
            return Path(self.stats_path)
        stats_dir = Path("." if self.stats_dir is None else self.stats_dir)
        return stats_dir / f"stats_{_stats_slug(spec=spec, scenario=scenario)}.json"

    def should_continue_after(self, result: RunResult) -> bool:
        """Return whether an outer benchmark loop should try the next scenario."""
        if result.status in {"completed", "skipped"}:
            return True
        if result.status != "failed" or result.error is None:
            return False
        return self.error_policy.handle(result.error).continue_next_scenario


class BenchmarkBatchInputSource:
    """Finite deterministic input source for one prepared benchmark scenario."""

    is_finite = True
    is_deterministic = True

    def __init__(self, *, scenario: PreparedScenario) -> None:
        self.user_input_schema = scenario.source_schema
        self._user_inputs = scenario.user_inputs

    def is_finished(self) -> bool:
        return False

    def next_window(self, request: object) -> UserInputWindow:
        del request
        window = _all_user_inputs_window(self._user_inputs)
        return UserInputWindow(
            start_s=window.start_s,
            end_s=window.end_s,
            inputs=self._user_inputs.window(window),
        )


def run_benchmark_demo(
    *,
    spec: DemoSpec,
    adapter: DemoAdapter,
    stats_path: str | Path | None = None,
    stats_dir: str | Path | None = None,
    capture_output: bool = True,
    metrics: MetricsRecorder | None = None,
    mp4_writer: VideoWriter | None = None,
    pipeline: StepPipeline | None = None,
) -> RunResult:
    """Run one benchmarked demo session through ``BenchmarkRunMode``."""
    mode = BenchmarkRunMode(
        stats_path=stats_path,
        stats_dir=stats_dir,
        capture_output=capture_output,
        run_metrics=metrics,
        mp4_writer=mp4_writer,
    )
    mode.validate_run(spec=spec, adapter=adapter)
    scenario = adapter.prepare_scenario(spec)
    runtime = adapter.create_runtime(_require_config(spec))
    host = RuntimeHost(runtime)
    try:
        model_warmup_plan = build_model_warmup_plan(
            host=host,
            adapter=adapter,
            spec=spec,
            scenario=scenario,
        )
        context = mode.create_run_context(
            spec=spec,
            adapter=adapter,
            host=host,
            model_warmup_plan=model_warmup_plan,
        )
        try:
            warmup_run_context(
                context=context,
                spec=spec,
                scenario=scenario,
                adapter=adapter,
                run_mode=mode,
            )
            return run_demo_session(
                context=context,
                spec=spec,
                scenario=scenario,
                adapter=adapter,
                run_mode=mode,
                pipeline=pipeline or StepPipeline(),
            )
        finally:
            context.close()
    finally:
        host.close()


def _all_user_inputs_window(user_inputs: UserInputs) -> TimeWindow:
    if not user_inputs.events:
        return TimeWindow(start_s=0.0, end_s=3600.0)
    return TimeWindow(
        start_s=0.0,
        end_s=max(
            3600.0,
            math.nextafter(user_inputs.events[-1].timestamp_s, math.inf),
        ),
    )


def _stats_slug(*, spec: DemoSpec, scenario: PreparedScenario) -> str:
    candidate = _metadata_string(scenario.metadata, "benchmark_scenario_id")
    if candidate is None:
        candidate = _metadata_string(scenario.metadata, "scenario_id")
    if candidate is None:
        candidate = _metadata_string(spec.metadata, "benchmark_scenario_id")
    if candidate is None:
        candidate = _metadata_string(spec.metadata, "scenario_id")
    if candidate is None and isinstance(spec.scenario, str):
        candidate = spec.scenario
    if candidate is None:
        candidate = spec.model_id
    value = _SAFE_STATS_ID_RE.sub("_", candidate.strip()).strip("._-")
    return value or "benchmark"


def _metadata_string(metadata: object, key: str) -> str | None:
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get(key)
    if isinstance(value, str) and value.strip():
        return value
    return None


def _require_config(spec: DemoSpec) -> InferenceConfig:
    if spec.config is None:
        raise RuntimeError("DemoSpec.config was not initialized.")
    return spec.config


def _require_supported_mode(
    *,
    mode: str,
    supported: tuple[str, ...],
    label: str,
) -> None:
    if mode in supported:
        return
    supported_text = ", ".join(repr(each) for each in supported) or "<none>"
    raise ValueError(
        f"Unsupported demo {label}={mode!r}; supported modes: {supported_text}."
    )


__all__ = [
    "BenchmarkBatchInputSource",
    "BenchmarkRunMode",
    "run_benchmark_demo",
]
