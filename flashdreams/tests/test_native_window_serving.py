# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU coverage for shared local-window presentation."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, cast

import pytest
import torch
from flashdreams.runtime import (
    CanonicalInputSchema,
    IdentityInputMapping,
    InferenceConfig,
    InferenceInput,
    InferenceInputSchema,
    StepRequest,
    StepRequirements,
    UserInputs,
    UserInputSchema,
)
from flashdreams.runtime.demo import (
    REALTIME_SKIPPED_INPUTS_METADATA_KEY,
    REALTIME_SKIPPED_WINDOW_METADATA_KEY,
    DemoSpec,
    NativeWindowOutputSpec,
    PreparedScenario,
    PreparedStep,
    ProviderCapabilities,
    RealtimeWindowResult,
    ResamplerRealtimeClock,
    SessionInfo,
    UserInputWindow,
)
from flashdreams.runtime.types import StepResult
from flashdreams.serving.native_window import run_native_window_presentation
from flashdreams.serving.native_window.presenter import (
    DEFAULT_NATIVE_KEY_BINDINGS,
    SlangPyNativePresenter,
)
from flashdreams.serving.native_window.services import (
    NativeFrameQueue,
    NativeWindowInputSource,
    NativeWindowOutputSink,
)

pytestmark = pytest.mark.ci_cpu


def _result(value: int, *, frames: int = 2) -> StepResult:
    return StepResult.from_video_chunk(
        step_index=value,
        video_chunk=torch.full((frames, 3, 2, 3), value, dtype=torch.uint8),
        layout="tchw",
    )


def test_native_output_spec_uses_public_local_window_mode() -> None:
    output = NativeWindowOutputSpec()
    assert output.mode == "local-window"
    assert output.max_queued_chunks == 2
    with pytest.raises(ValueError, match="dimensions"):
        NativeWindowOutputSpec(video_width=0)
    with pytest.raises(ValueError, match="close_timeout_s"):
        NativeWindowOutputSpec(close_timeout_s=0)
    with pytest.raises(ValueError, match="indices"):
        NativeWindowOutputSpec(view_index=-1)


def test_native_queue_drops_stale_chunks_and_drains_final_frames() -> None:
    queue = NativeFrameQueue(max_chunks=1)
    stopped, dropped, queued = queue.publish(_result(1))
    assert not stopped and not dropped and queued == 2

    stopped, dropped, queued = queue.publish(_result(2))
    assert not stopped and dropped and queued == 2
    queue.finish()
    assert not queue.drained

    frames = [queue.pop(), queue.pop()]
    assert all(frame is not None for frame in frames)
    assert cast(Any, frames[0]).to_numpy()[0, 0].tolist() == [2, 2, 2]
    assert queue.drained


def test_native_queue_uses_explicit_video_view() -> None:
    queue = NativeFrameQueue(max_chunks=1)
    video = torch.stack(
        (
            torch.full((1, 3, 2, 3), 1, dtype=torch.uint8),
            torch.full((1, 3, 2, 3), 2, dtype=torch.uint8),
        ),
        dim=0,
    ).unsqueeze(0)
    result = StepResult.from_video_chunk(
        step_index=0,
        video_chunk=video,
        layout="bvtchw",
    )

    queue.publish(result, view_index=1)

    frame = queue.pop()
    assert frame is not None
    assert cast(Any, frame).to_numpy()[0, 0].tolist() == [2, 2, 2]


def test_native_queue_starts_host_prefetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefetched: list[object] = []
    from flashdreams.serving.native_window import services

    monkeypatch.setattr(
        services,
        "prefetch_to_numpy",
        prefetched.append,
    )

    NativeFrameQueue(max_chunks=1).publish(_result(1))

    assert len(prefetched) == 2


def test_native_sink_marks_normal_completion_without_discarding_frames() -> None:
    queue = NativeFrameQueue(max_chunks=2)
    sink = NativeWindowOutputSink(queue=queue)
    sink.open(SessionInfo())
    decision = sink.write(_result(1))

    assert decision.metadata["queued_frames"] == 2
    assert sink.close() == ()
    assert not queue.drained
    assert queue.pop() is not None
    assert queue.pop() is not None
    assert queue.drained


def test_native_sink_tolerates_close_before_generation() -> None:
    queue = NativeFrameQueue(max_chunks=1)
    sink = NativeWindowOutputSink(queue=queue)
    sink.open(SessionInfo())
    queue.close()

    sink.begin_generation(0)
    decision = sink.write(_result(1))

    assert decision.should_stop


def test_native_source_preserves_skipped_key_transitions() -> None:
    async def sample() -> RealtimeWindowResult:
        source = NativeWindowInputSource(fps=10)
        source.reset(start_v=0.0)
        source.max_lag_s = 0.0
        source.record_key(event="keydown", key="w", timestamp_s=0.05)
        source.record_key(event="keyup", key="w", timestamp_s=0.25)
        source.record_key(event="keydown", key="q", timestamp_s=0.35)
        clock = ResamplerRealtimeClock(
            resampler=source.resampler,
            now_fn=lambda: 0.4,
        )
        return await source.next_realtime_window(
            request=StepRequirements(step_index=0, input_frame_count=1),
            clock=clock,
        )

    result = asyncio.run(sample())
    skipped = result.window.metadata[REALTIME_SKIPPED_INPUTS_METADATA_KEY]
    assert isinstance(skipped, UserInputs)
    assert [event.event_type for event in skipped.events] == ["key_down", "key_up"]
    assert [event.payload["key"] for event in result.window.inputs.events] == ["q"]
    skipped_start, skipped_end = cast(
        tuple[float, float],
        result.window.metadata[REALTIME_SKIPPED_WINDOW_METADATA_KEY],
    )
    assert skipped_start == 0.0
    assert skipped_end == pytest.approx(0.3)


def test_native_source_folds_late_event_into_next_window() -> None:
    async def sample() -> RealtimeWindowResult:
        source = NativeWindowInputSource(fps=10)
        source.reset(start_v=0.0)
        now = 0.1
        clock = ResamplerRealtimeClock(
            resampler=source.resampler,
            now_fn=lambda: now,
        )
        await source.next_realtime_window(
            request=StepRequirements(step_index=0, input_frame_count=1),
            clock=clock,
        )
        source.record_key(event="keydown", key="r", timestamp_s=0.05)
        now = 0.2
        return await source.next_realtime_window(
            request=StepRequirements(step_index=1, input_frame_count=1),
            clock=clock,
        )

    result = asyncio.run(sample())
    skipped = result.window.metadata[REALTIME_SKIPPED_INPUTS_METADATA_KEY]
    assert isinstance(skipped, UserInputs)
    assert [event.payload["key"] for event in skipped.events] == ["r"]


def test_native_source_does_not_repeat_consumed_events() -> None:
    async def sample() -> tuple[RealtimeWindowResult, RealtimeWindowResult]:
        source = NativeWindowInputSource(fps=10)
        source.reset(start_v=0.0)
        now = 0.1
        clock = ResamplerRealtimeClock(
            resampler=source.resampler,
            now_fn=lambda: now,
        )
        source.record_key(event="keydown", key="space", timestamp_s=0.05)
        first = await source.next_realtime_window(
            request=StepRequirements(step_index=0, input_frame_count=1),
            clock=clock,
        )
        now = 0.2
        second = await source.next_realtime_window(
            request=StepRequirements(step_index=1, input_frame_count=1),
            clock=clock,
        )
        return first, second

    first, second = asyncio.run(sample())
    assert [event.payload["key"] for event in first.window.inputs.events] == ["space"]
    assert second.window.inputs.events == ()
    assert REALTIME_SKIPPED_INPUTS_METADATA_KEY not in second.window.metadata


def test_default_presenter_bindings_cover_camera_controls() -> None:
    assert set(DEFAULT_NATIVE_KEY_BINDINGS) == {
        "w",
        "a",
        "s",
        "d",
        "q",
        "e",
        "i",
        "j",
        "k",
        "l",
        "r",
        "g",
        "b",
        "space",
        "shift",
        "control",
    }


def test_empty_presenter_bindings_disable_controls() -> None:
    from types import SimpleNamespace

    from flashdreams.serving.native_window import presenter

    spy = SimpleNamespace(KeyCode=SimpleNamespace(r=object()))
    assert presenter._build_key_map(spy, {}) == {}


class _FakeProvider:
    capabilities = ProviderCapabilities(
        supports_realtime_clock=True,
        user_input_schema=UserInputSchema(),
        inference_input_schema=InferenceInputSchema(),
    )

    def prepare_initial_input(self) -> InferenceInput:
        return InferenceInput()

    def prepare_step(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
    ) -> PreparedStep:
        del request, user_window
        return PreparedStep(inference_input=InferenceInput())

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs

    def close(self) -> None:
        return None


class _FakeSession:
    def __init__(self, *, steps: int) -> None:
        self.steps = steps
        self.index = 0

    def next_step_request(self) -> StepRequest | None:
        if self.index >= self.steps:
            return None
        return StepRequest(
            step_index=self.index,
            metadata={"input_frame_count": 1},
        )

    def step(self, inputs: InferenceInput) -> StepResult:
        del inputs
        result = _result(self.index, frames=1)
        self.index += 1
        return result

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        self.index = 0

    def close(self) -> None:
        return None


class _FakeRuntime:
    def __init__(self, *, steps: int) -> None:
        self.steps = steps

    def start_session(self, inputs: InferenceInput) -> _FakeSession:
        del inputs
        return _FakeSession(steps=self.steps)

    def close(self) -> None:
        return None


class _FakeAdapter:
    model_id = "fake-native"
    inference_input_schema = InferenceInputSchema()
    canonical_input_schema = CanonicalInputSchema()

    def __init__(self, *, steps: int = 2, runtime_error: Exception | None = None):
        self.steps = steps
        self.runtime_error = runtime_error

    def supported_input_modes(self) -> tuple[str, ...]:
        return ("realtime",)

    def supported_output_modes(self) -> tuple[str, ...]:
        return ("local-window",)

    def default_input_mapping(self) -> IdentityInputMapping:
        return IdentityInputMapping()

    def validate_config(self, config: InferenceConfig) -> None:
        del config

    def create_runtime(self, config: InferenceConfig) -> _FakeRuntime:
        del config
        if self.runtime_error is not None:
            raise self.runtime_error
        return _FakeRuntime(steps=self.steps)

    def prepare_scenario(self, spec: DemoSpec) -> PreparedScenario:
        del spec
        return PreparedScenario(initial_inputs=InferenceInput())

    def create_model_input_provider(
        self,
        spec: DemoSpec,
        scenario: PreparedScenario,
    ) -> _FakeProvider:
        del spec, scenario
        return _FakeProvider()


class _FakePresenter:
    def __init__(self, **kwargs: object) -> None:
        del kwargs
        self.frames: list[object] = []
        self.closed = False

    @property
    def should_close(self) -> bool:
        return False

    def process_events(self) -> None:
        return None

    def present_frame(self, frame: object) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        self.closed = True


def _native_spec(*, close_timeout_s: float = 1.0) -> DemoSpec:
    return DemoSpec(
        model_id="fake-native",
        input_mode="realtime",
        output=NativeWindowOutputSpec(
            fps=1000,
            video_width=3,
            video_height=2,
            close_timeout_s=close_timeout_s,
        ),
        config=InferenceConfig(model_id="fake-native", device="cpu"),
    )


def test_native_runner_drains_finite_tail_with_fake_presenter() -> None:
    presenter = _FakePresenter()

    result = run_native_window_presentation(
        spec=_native_spec(),
        adapter=_FakeAdapter(steps=2),
        presenter_factory=lambda **_kwargs: presenter,
    )

    assert result.status == "completed"
    assert len(presenter.frames) == 2
    assert presenter.closed


def test_native_runner_surfaces_worker_failure() -> None:
    with pytest.raises(RuntimeError, match="session failed"):
        run_native_window_presentation(
            spec=_native_spec(),
            adapter=_FakeAdapter(runtime_error=RuntimeError("runtime failed")),
            presenter_factory=_FakePresenter,
        )


def test_native_runner_surfaces_presenter_failure() -> None:
    class _FailingPresenter(_FakePresenter):
        def present_frame(self, frame: object) -> None:
            del frame
            raise RuntimeError("present failed")

    with pytest.raises(RuntimeError, match="presenter failed"):
        run_native_window_presentation(
            spec=_native_spec(),
            adapter=_FakeAdapter(steps=1),
            presenter_factory=_FailingPresenter,
        )


def test_native_runner_rejects_multi_process_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    adapter = _FakeAdapter()

    with pytest.raises(RuntimeError, match="one process"):
        run_native_window_presentation(
            spec=_native_spec(),
            adapter=adapter,
            presenter_factory=_FakePresenter,
        )


def test_presenter_reports_missing_native_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from flashdreams.serving.native_window import presenter as presenter_module

    def missing(_name: str) -> object:
        raise ImportError("missing")

    monkeypatch.setattr(presenter_module.importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="native-window"):
        SlangPyNativePresenter(
            width=2,
            height=2,
            title="test",
            on_key=lambda _event, _key: None,
        )


def test_native_runner_bounds_close_before_worker_ready() -> None:
    release = threading.Event()
    closed = threading.Event()
    preloaded = threading.Event()

    class _TrackedRuntime(_FakeRuntime):
        def preload(self) -> None:
            preloaded.set()

        def close(self) -> None:
            closed.set()

    class _BlockingAdapter(_FakeAdapter):
        def create_runtime(self, config: InferenceConfig) -> _FakeRuntime:
            del config
            release.wait(0.2)
            return _TrackedRuntime(steps=1)

    class _ClosingPresenter(_FakePresenter):
        @property
        def should_close(self) -> bool:
            return True

    try:
        with pytest.raises(RuntimeError, match="did not stop"):
            run_native_window_presentation(
                spec=_native_spec(close_timeout_s=0.02),
                adapter=_BlockingAdapter(),
                presenter_factory=_ClosingPresenter,
            )
    finally:
        release.set()
    assert closed.wait(1.0)
    assert not preloaded.is_set()
