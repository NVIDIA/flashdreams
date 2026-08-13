# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native-window session edges for the shared realtime driver."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from typing import Any

from flashdreams.runtime import (
    StepResult,
    UserInputCapability,
    UserInputEvent,
    UserInputSchema,
)
from flashdreams.runtime.demo import (
    AlwaysActiveActivationPolicy,
    InMemorySessionMetricsRecorder,
    ModelInputProvider,
    ModelWarmupPlan,
    NativeWindowErrorPolicy,
    NativeWindowOutputSpec,
    NoopTransportService,
    OutputDecision,
    PreparedScenario,
    RealtimeEventInputSource,
    RealtimeEventResampler,
    RealtimeSessionDriver,
    ResamplerRealtimeClock,
    RunContext,
    RunModeCapabilities,
    RuntimeHost,
    SessionEdges,
    SessionInfo,
    SingleSessionAdmissionPolicy,
)
from flashdreams.runtime.demo.spec import DemoAdapter, DemoSpec


class NativeFrameQueue:
    """Bounded chunk queue with distinct graceful-finish and abort states."""

    def __init__(self, *, max_chunks: int) -> None:
        if max_chunks <= 0:
            raise ValueError("max_chunks must be > 0.")
        self._max_chunks = max_chunks
        self._chunks: deque[deque[object]] = deque()
        self._lock = threading.Lock()
        self._closed = False
        self._producer_finished = False

    @property
    def empty(self) -> bool:
        with self._lock:
            return not self._chunks

    @property
    def drained(self) -> bool:
        with self._lock:
            return self._producer_finished and not self._chunks

    def begin_generation(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot begin generation on a closed queue.")
            self._chunks.clear()
            self._producer_finished = False

    def publish(
        self,
        result: StepResult,
        *,
        batch_index: int = 0,
        view_index: int = 0,
    ) -> tuple[bool, bool, int]:
        frames = deque[object](
            result.lazy_rgb_frames(
                batch_index=batch_index,
                view_index=view_index,
                record_cuda_event=True,
            )
        )
        with self._lock:
            if self._closed:
                return True, False, 0
            if not frames:
                return False, False, sum(len(chunk) for chunk in self._chunks)
            dropped = len(self._chunks) >= self._max_chunks
            if dropped:
                self._chunks.popleft()
            self._chunks.append(frames)
            return False, dropped, sum(len(chunk) for chunk in self._chunks)

    def pop(self) -> object | None:
        with self._lock:
            if not self._chunks:
                return None
            frame = self._chunks[0].popleft()
            if not self._chunks[0]:
                self._chunks.popleft()
            return frame

    def finish(self) -> None:
        """Mark normal producer completion while retaining queued frames."""
        with self._lock:
            self._producer_finished = True

    def close(self) -> None:
        """Abort presentation and release every queued frame."""
        with self._lock:
            self._closed = True
            self._producer_finished = True
            self._chunks.clear()


class NativeWindowOutputSink:
    """Publish generated frames to the local presentation queue."""

    produces_artifacts = False

    def __init__(
        self,
        *,
        queue: NativeFrameQueue,
        batch_index: int = 0,
        view_index: int = 0,
    ) -> None:
        self._queue = queue
        self._batch_index = batch_index
        self._view_index = view_index
        self._open = False

    def open(self, session_info: SessionInfo) -> None:
        del session_info
        self._open = True

    def begin_generation(self, generation: int) -> None:
        del generation
        self._queue.begin_generation()

    def write(self, result: StepResult) -> OutputDecision:
        if not self._open:
            raise RuntimeError("Cannot write to a closed native-window sink.")
        stopped, dropped, queued = self._queue.publish(
            result,
            batch_index=self._batch_index,
            view_index=self._view_index,
        )
        return OutputDecision(
            should_stop=stopped,
            dropped=dropped,
            drop_policy="drop_oldest" if dropped else "none",
            metadata={"queued_frames": queued},
        )

    def close(self) -> Sequence[Any]:
        if self._open:
            self._open = False
            self._queue.finish()
        return ()


_NATIVE_INPUT_SCHEMA = UserInputSchema(
    capabilities=(
        UserInputCapability(event_type="key_down", payload_fields=frozenset({"key"})),
        UserInputCapability(event_type="key_up", payload_fields=frozenset({"key"})),
        UserInputCapability(
            event_type="text_event",
            payload_fields=frozenset({"event_id", "state"}),
        ),
    ),
    description="Native-window keyboard and text events.",
)


class NativeWindowInputSource(RealtimeEventInputSource):
    """Thread-safe realtime input source fed by the native UI thread."""

    def __init__(self, *, fps: int) -> None:
        self._lock = threading.RLock()
        super().__init__(
            resampler=RealtimeEventResampler(fps=fps),
            user_input_schema=_NATIVE_INPUT_SCHEMA,
        )

    def record_key(self, *, event: str, key: str, timestamp_s: float) -> None:
        event_type = {"keydown": "key_down", "keyup": "key_up"}.get(event.lower())
        if event_type is None or not key.strip():
            raise ValueError("Native key events require keydown/keyup and a key.")
        self.record_user_event(
            UserInputEvent(
                timestamp_s=timestamp_s,
                event_type=event_type,
                payload={"key": key.strip()},
                source="native-window",
            )
        )

    def record_user_event(self, event: UserInputEvent) -> None:
        with self._lock:
            super().record_user_event(event)

    def _consume_events(
        self,
        *,
        start_s: float,
        end_s: float,
    ) -> tuple[tuple[UserInputEvent, ...], tuple[UserInputEvent, ...]]:
        with self._lock:
            return super()._consume_events(
                start_s=start_s,
                end_s=end_s,
            )

    def reset(self, *, start_v: float) -> None:
        with self._lock:
            super().reset(start_v=start_v)


class NativeWindowRunMode:
    """Realtime run mode for a local window."""

    name = "local-window"
    capabilities = RunModeCapabilities(
        realtime=True,
        supports_interactive_events=True,
    )

    def __init__(
        self,
        *,
        input_source: NativeWindowInputSource,
        output_sink: NativeWindowOutputSink,
        transport: NoopTransportService,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.input = input_source
        self.output = output_sink
        self.transport = transport
        self.clock_fn = clock_fn

    def validate_run(self, *, spec: DemoSpec, adapter: DemoAdapter) -> None:
        del adapter
        if not isinstance(spec.output, NativeWindowOutputSpec):
            raise TypeError("NativeWindowRunMode requires NativeWindowOutputSpec.")

    def validate_session(
        self,
        *,
        spec: DemoSpec,
        scenario: PreparedScenario,
        adapter: DemoAdapter,
        provider: ModelInputProvider,
    ) -> None:
        del spec, scenario, adapter
        if not provider.capabilities.supports_realtime_clock:
            raise ValueError("Local-window providers must support realtime clocks.")

    def create_run_context(
        self,
        *,
        spec: DemoSpec,
        adapter: DemoAdapter,
        host: RuntimeHost,
        model_warmup_plan: ModelWarmupPlan,
    ) -> RunContext:
        del spec, adapter
        return RunContext(
            host=host,
            run_metrics=InMemorySessionMetricsRecorder(),
            admission=SingleSessionAdmissionPolicy(
                health_check=lambda: host.is_healthy
            ),
            model_warmup_plan=model_warmup_plan,
        )

    def create_session_edges(
        self,
        *,
        context: RunContext,
        spec: DemoSpec,
        scenario: PreparedScenario,
        provider: ModelInputProvider,
        adapter: DemoAdapter,
    ) -> SessionEdges:
        del spec, scenario, provider, adapter
        return SessionEdges(
            input_source=self.input,
            output_sink=self.output,
            cleanup_tasks=context.cleanup_tasks,
            error_policy=NativeWindowErrorPolicy(),
            transport=self.transport,
            clock=ResamplerRealtimeClock(
                resampler=self.input.resampler,
                now_fn=self.clock_fn,
            ),
            activation=AlwaysActiveActivationPolicy(anchor_clock=True),
        )

    def select_driver(self) -> RealtimeSessionDriver:
        return RealtimeSessionDriver()


__all__ = [
    "NativeFrameQueue",
    "NativeWindowInputSource",
    "NativeWindowOutputSink",
    "NativeWindowRunMode",
]
