# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime host and model-execution boundary for shared demos."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TypeVar

from flashdreams.runtime._utils import freeze_mapping
from flashdreams.runtime.inputs import InferenceInput
from flashdreams.runtime.interfaces import InferenceRuntime, InferenceSession
from flashdreams.runtime.worker import ModelExecutionWorker

_T = TypeVar("_T")


@dataclass(frozen=True, kw_only=True, slots=True)
class WarmupSessionInputs:
    """Inputs used to warm one temporary runtime session."""

    initial_input: InferenceInput
    step_inputs: Sequence[InferenceInput] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_inputs", tuple(self.step_inputs))


@dataclass(frozen=True, kw_only=True, slots=True)
class ModelWarmupPlan:
    """Host-owned model warmup plan built by a demo adapter or run mode."""

    sessions: Sequence[WarmupSessionInputs] = ()
    measured: bool = False
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sessions", tuple(self.sessions))
        object.__setattr__(self, "metadata", freeze_mapping(self.metadata))


class RuntimeHost:
    """Own one runtime and the worker used for model-affine calls."""

    def __init__(
        self,
        runtime: InferenceRuntime,
        *,
        worker: ModelExecutionWorker | None = None,
        is_control_rank: bool = True,
        worker_loop: Callable[[], None] | None = None,
    ) -> None:
        self._runtime = runtime
        self._worker = worker or ModelExecutionWorker()
        self._is_control_rank = is_control_rank
        self._worker_loop = worker_loop
        self._healthy = True
        self._closed = False
        self._unhealthy_reason: str | None = None
        self._unhealthy_error: Exception | None = None

    @property
    def runtime(self) -> InferenceRuntime:
        """Return the hosted runtime."""
        return self._runtime

    @property
    def worker(self) -> ModelExecutionWorker:
        """Return the host's model-execution worker."""
        return self._worker

    @property
    def is_control_rank(self) -> bool:
        """Whether this process owns run modes, providers, sinks, and metrics."""
        return self._is_control_rank

    @property
    def is_healthy(self) -> bool:
        """Return whether admission should continue accepting sessions."""
        return self._healthy and not self._closed

    @property
    def unhealthy_reason(self) -> str | None:
        """Return the first latched unhealthy reason, if any."""
        return self._unhealthy_reason

    @property
    def unhealthy_error(self) -> Exception | None:
        """Return the first latched unhealthy error, if any."""
        return self._unhealthy_error

    def mark_unhealthy(
        self,
        reason: str = "marked unhealthy",
        error: Exception | None = None,
    ) -> None:
        """Latch the host as unhealthy without overwriting the first reason."""
        if not self._healthy:
            return
        self._healthy = False
        self._unhealthy_reason = reason
        self._unhealthy_error = error

    def preload(self) -> None:
        """Initialize optional distributed state and preload runtime resources."""
        self._call_optional_runtime_hook("initialize_distributed")
        self._call_optional_runtime_hook("preload")

    def warmup(self, plan: ModelWarmupPlan | None = None) -> None:
        """Run warmup sessions through the same worker boundary as real sessions."""
        plan = plan or ModelWarmupPlan()
        for warmup_session in plan.sessions:
            session = self.call(self.start_session, warmup_session.initial_input)
            try:
                for step_input in warmup_session.step_inputs:
                    self.call(session.step, step_input)
            finally:
                self.call(session.close)

    def call(self, func: Callable[..., _T], /, *args: object, **kwargs: object) -> _T:
        """Run one model-affine callable synchronously on the worker."""
        self._require_open()
        return self._worker.call_blocking(func, *args, **kwargs)

    async def call_async(
        self,
        func: Callable[..., _T],
        /,
        *args: object,
        **kwargs: object,
    ) -> _T:
        """Run model-affine work without blocking realtime event loops."""
        self._require_open()
        return await self._worker.call(func, *args, **kwargs)

    def start_session(self, inputs: InferenceInput) -> InferenceSession:
        """Start one inference session through the hosted runtime."""
        self._require_open()
        return self._runtime.start_session(inputs)

    def run_worker_loop(self) -> None:
        """Serve control-rank work on non-control ranks until runtime shutdown."""
        worker_loop = self._worker_loop
        if worker_loop is None:
            worker_loop = getattr(self._runtime, "run_worker_loop", None)
        if worker_loop is None:
            worker_loop = getattr(self._runtime, "wait_for_termination", None)
        if callable(worker_loop):
            worker_loop()

    def close(self) -> None:
        """Close runtime-owned state and stop the model-execution worker."""
        if self._closed:
            return
        try:
            self._worker.call_blocking(self._runtime.close)
            self._call_optional_runtime_hook("close_distributed")
        finally:
            self._closed = True
            self._worker.close_blocking()

    def _call_optional_runtime_hook(self, name: str) -> None:
        hook = getattr(self._runtime, name, None)
        if callable(hook):
            self.call(hook)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("runtime host is closed")


__all__ = ["ModelWarmupPlan", "RuntimeHost", "WarmupSessionInputs"]
