# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread-affine interactive inference worker with streamed step results."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from flashdreams.runtime.canonical import InputCanonicalizer
from flashdreams.runtime.config import InferenceConfig
from flashdreams.runtime.inputs import (
    InferenceInput,
    InferenceInputSchema,
    UserInputSchema,
)
from flashdreams.runtime.interfaces import InferenceRuntime, ModelAdapter
from flashdreams.runtime.mapping import InputMapping
from flashdreams.runtime.metrics import MetricsRecorder, NullMetricsRecorder
from flashdreams.runtime.output import OutputArtifact
from flashdreams.runtime.output_schema import OutputTargetRequirement
from flashdreams.runtime.runner import run_inference_session
from flashdreams.runtime.sources import UserInputSource
from flashdreams.runtime.types import StepResult


@dataclass(frozen=True, kw_only=True, slots=True)
class InteractiveSessionJob:
    """Inputs required to run one session on a reusable model worker."""

    session_id: str
    """Application identity for correlating streamed events."""

    mapping: InputMapping
    """Canonical-to-model input mapping."""

    canonicalizer: InputCanonicalizer
    """Application device-input canonicalizer."""

    source_schema: UserInputSchema
    """Raw capabilities provided by ``user_inputs``."""

    user_inputs: UserInputSource
    """Replay or live input source."""

    initial_inputs: InferenceInput
    """Global conditioning and initial step values."""

    inference_input_schema: InferenceInputSchema | None = None
    """Route-specific schema; ``None`` uses the adapter default."""

    metrics: MetricsRecorder = field(default_factory=NullMetricsRecorder)
    """Per-session metrics recorder."""


@dataclass(frozen=True, kw_only=True, slots=True)
class InteractiveStep:
    """One generated result published by an interactive worker."""

    session_id: str
    result: StepResult


@dataclass(frozen=True, kw_only=True, slots=True)
class InteractiveSessionEnded:
    """Terminal event for one interactive session."""

    session_id: str
    stopped: bool
    error: BaseException | None = None


InteractiveEvent = InteractiveStep | InteractiveSessionEnded


class InteractiveInferenceWorker:
    """Own one model runtime on a worker thread across sequential sessions."""

    def __init__(
        self,
        *,
        adapter: ModelAdapter,
        config: InferenceConfig,
        result_queue_size: int = 8,
        runtime_factory: Callable[[], InferenceRuntime] | None = None,
    ) -> None:
        if result_queue_size <= 0:
            raise ValueError("result_queue_size must be > 0.")
        self._adapter = adapter
        self._config = config
        self._runtime_factory = runtime_factory
        self._commands: queue.Queue[InteractiveSessionJob | None] = queue.Queue(
            maxsize=1
        )
        self._events: queue.Queue[InteractiveEvent] = queue.Queue(
            maxsize=result_queue_size
        )
        self._ready = threading.Event()
        self._closing = threading.Event()
        self._session_stop = threading.Event()
        self._state_lock = threading.Lock()
        self._active_session_id: str | None = None
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=f"{adapter.model_id}-interactive-runtime",
            daemon=True,
        )

    @property
    def active_session_id(self) -> str | None:
        """Return the session currently owned by the worker."""
        with self._state_lock:
            return self._active_session_id

    def start(self, *, wait: bool = True, timeout_s: float = 60.0) -> None:
        """Start runtime creation on the worker thread."""
        if self._thread.is_alive() or self._ready.is_set():
            raise RuntimeError("InteractiveInferenceWorker has already started.")
        self._thread.start()
        if wait:
            self.wait_until_ready(timeout_s=timeout_s)

    def wait_until_ready(self, *, timeout_s: float = 60.0) -> None:
        """Wait for worker-owned runtime creation to finish."""
        if not self._ready.wait(timeout_s):
            raise TimeoutError("Timed out waiting for interactive runtime startup.")
        if self._startup_error is not None:
            raise RuntimeError(
                "Interactive runtime startup failed."
            ) from self._startup_error

    def submit(self, job: InteractiveSessionJob) -> None:
        """Start one session when no other session is active."""
        if not self._ready.is_set() or self._startup_error is not None:
            raise RuntimeError("InteractiveInferenceWorker is not ready.")
        with self._state_lock:
            if self._active_session_id is not None:
                raise RuntimeError(
                    f"Session {self._active_session_id!r} is still active."
                )
            self._active_session_id = job.session_id
        self._session_stop.clear()
        self._commands.put_nowait(job)

    def stop_session(self) -> None:
        """Request that the active session stop after its current model step."""
        self._session_stop.set()

    def get_event(self, *, timeout_s: float | None = None) -> InteractiveEvent | None:
        """Return the next generated or terminal event, or ``None`` on timeout."""
        try:
            return self._events.get(timeout=timeout_s)
        except queue.Empty:
            return None

    def close(self, *, timeout_s: float = 60.0) -> None:
        """Stop the active session and close the worker-owned runtime."""
        self._closing.set()
        self._session_stop.set()
        if self._thread.is_alive():
            while True:
                try:
                    self._commands.put(None, timeout=0.05)
                    break
                except queue.Full:
                    if not self._thread.is_alive():
                        break
            self._thread.join(timeout_s)
            if self._thread.is_alive():
                raise TimeoutError("Timed out closing interactive inference worker.")

    def _run(self) -> None:
        runtime: InferenceRuntime | None = None
        try:
            self._adapter.validate_config(self._config)
            runtime = (
                self._runtime_factory()
                if self._runtime_factory is not None
                else self._adapter.create_runtime(self._config)
            )
            if runtime.config != self._config:
                raise ValueError(
                    "Interactive runtime config does not match the worker config."
                )
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            return
        self._ready.set()

        try:
            while not self._closing.is_set():
                job = self._commands.get()
                if job is None:
                    break
                self._run_job(runtime=runtime, job=job)
        finally:
            runtime.close()

    def _run_job(
        self,
        *,
        runtime: InferenceRuntime,
        job: InteractiveSessionJob,
    ) -> None:
        error: BaseException | None = None
        try:
            run_inference_session(
                adapter=self._adapter,
                config=self._config,
                mapping=job.mapping,
                canonicalizer=job.canonicalizer,
                source_schema=job.source_schema,
                user_inputs=job.user_inputs,
                initial_inputs=job.initial_inputs,
                output=_WorkerOutputTarget(
                    session_id=job.session_id,
                    requirement=_requirement_for(self._adapter),
                    events=self._events,
                    stop_requested=self._session_stop,
                    closing=self._closing,
                ),
                metrics=job.metrics,
                runtime=runtime,
                inference_input_schema=job.inference_input_schema,
            )
        except BaseException as exc:
            error = exc
        finally:
            self._publish(
                InteractiveSessionEnded(
                    session_id=job.session_id,
                    stopped=self._session_stop.is_set(),
                    error=error,
                )
            )
            with self._state_lock:
                self._active_session_id = None

    def _publish(self, event: InteractiveEvent) -> None:
        while not self._closing.is_set():
            try:
                self._events.put(event, timeout=0.05)
                return
            except queue.Full:
                continue


class _WorkerOutputTarget:
    def __init__(
        self,
        *,
        session_id: str,
        requirement: OutputTargetRequirement,
        events: queue.Queue[InteractiveEvent],
        stop_requested: threading.Event,
        closing: threading.Event,
    ) -> None:
        self._session_id = session_id
        self._requirement = requirement
        self._events = events
        self._stop_requested = stop_requested
        self._closing = closing
        self._opened = False

    @property
    def output_requirement(self) -> OutputTargetRequirement:
        return self._requirement

    @property
    def should_stop(self) -> bool:
        return self._stop_requested.is_set() or self._closing.is_set()

    def open(self) -> None:
        self._opened = True

    def poll(self) -> None:
        return

    def write(self, result: StepResult) -> None:
        if not self._opened:
            raise RuntimeError("Cannot write to a closed worker output target.")
        event = InteractiveStep(session_id=self._session_id, result=result)
        while not self.should_stop:
            try:
                self._events.put(event, timeout=0.05)
                return
            except queue.Full:
                continue

    def close(self) -> Sequence[OutputArtifact]:
        self._opened = False
        return ()


def _requirement_for(adapter: ModelAdapter) -> OutputTargetRequirement:
    schema = adapter.inference_output_schema
    return OutputTargetRequirement(
        modalities=frozenset({schema.modality}),
        python_type=schema.python_type,
        layouts=schema.layouts,
    )


__all__ = [
    "InteractiveEvent",
    "InteractiveInferenceWorker",
    "InteractiveSessionEnded",
    "InteractiveSessionJob",
    "InteractiveStep",
]
