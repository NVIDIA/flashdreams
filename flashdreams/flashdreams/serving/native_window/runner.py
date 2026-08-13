# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Coordinate a native-window presentation and model session."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch.distributed as dist

from flashdreams.runtime.demo import (
    DemoAdapter,
    DemoSpec,
    ModelWarmupPlan,
    NativeWindowOutputSpec,
    NoopTransportService,
    PreparedScenario,
    RunResult,
    RuntimeHost,
    StepPipeline,
    run_demo_session_async,
)
from flashdreams.runtime.worker import ModelExecutionWorker

from .presenter import SlangPyNativePresenter
from .services import (
    NativeFrameQueue,
    NativeWindowInputSource,
    NativeWindowOutputSink,
    NativeWindowRunMode,
)


class NativePresenter(Protocol):
    @property
    def should_close(self) -> bool: ...

    def process_events(self) -> None: ...

    def present_frame(self, frame: object) -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class _State:
    finished: threading.Event = field(default_factory=threading.Event)
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    deferred_cleanup: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    pending_keys: deque[tuple[str, str, float]] = field(default_factory=deque)
    input_source: NativeWindowInputSource | None = None
    transport: NoopTransportService | None = None
    host: RuntimeHost | None = None
    result: RunResult | None = None
    error: Exception | None = None


def run_native_window_presentation(
    *,
    spec: DemoSpec,
    adapter: DemoAdapter,
    presenter_factory: Callable[..., NativePresenter] = SlangPyNativePresenter,
    key_bindings: Mapping[str, Sequence[str]] | None = None,
    clock_fn: Callable[[], float] = time.monotonic,
) -> RunResult:
    """Run one realtime model session in a native window."""
    output = spec.output
    if not isinstance(output, NativeWindowOutputSpec):
        raise TypeError("Local-window output requires NativeWindowOutputSpec.")
    config = spec.config
    if config is None:
        raise RuntimeError("DemoSpec.config was not initialized.")
    _require_single_process()

    scenario = adapter.prepare_scenario(spec)
    queue = NativeFrameQueue(max_chunks=output.max_queued_chunks)
    state = _State()

    def on_key(event: str, key: str) -> None:
        timestamp_s = clock_fn()
        with state.lock:
            source = state.input_source
            if source is None:
                state.pending_keys.append((event, key, timestamp_s))
                return
        source.record_key(event=event, key=key, timestamp_s=timestamp_s)

    presenter = presenter_factory(
        width=output.video_width,
        height=output.video_height,
        title=output.title,
        on_key=on_key,
        key_bindings=key_bindings,
    )

    def worker() -> None:
        model_worker = ModelExecutionWorker(device=config.device)
        host: RuntimeHost | None = None
        try:
            runtime = model_worker.call_blocking(adapter.create_runtime, config)
            host = RuntimeHost(runtime, worker=model_worker)
            state.host = host
            state.result = asyncio.run(
                _run_session(
                    state=state,
                    host=host,
                    spec=spec,
                    scenario=scenario,
                    adapter=adapter,
                    queue=queue,
                    output=output,
                    clock_fn=clock_fn,
                )
            )
        except Exception as exc:  # noqa: BLE001 - worker boundary captures failures
            state.error = exc
            if host is None:
                model_worker.close_blocking()
        finally:
            if state.deferred_cleanup.is_set() and host is not None:
                try:
                    host.close()
                except Exception as exc:  # noqa: BLE001 - deferred cleanup is best effort
                    state.error = state.error or exc
            state.finished.set()

    thread = threading.Thread(
        target=worker,
        name="flashdreams-native-window",
        daemon=True,
    )
    thread.start()

    presenter_error: Exception | None = None
    user_closed = False
    interval_s = 1.0 / output.fps
    next_frame_s = clock_fn()
    try:
        while True:
            presenter.process_events()
            if presenter.should_close:
                user_closed = True
                break
            frame = queue.pop()
            if frame is None:
                if queue.drained or (state.finished.is_set() and queue.empty):
                    break
                time.sleep(0.001)
                continue
            time.sleep(max(0.0, next_frame_s - clock_fn()))
            presenter.present_frame(frame)
            next_frame_s = max(next_frame_s + interval_s, clock_fn())
    except Exception as exc:  # noqa: BLE001 - presenter boundary captures failures
        presenter_error = exc
    finally:
        if user_closed or presenter_error is not None:
            state.cancel_requested.set()
            queue.close()
        _request_transport_close(state)
        close_error = _finish_native_session(
            state=state,
            presenter=presenter,
            queue=queue,
            thread=thread,
            timeout_s=output.close_timeout_s,
        )

    if presenter_error is not None:
        raise RuntimeError("Local-window presenter failed.") from presenter_error
    if close_error is not None:
        raise close_error
    if state.error is not None:
        raise RuntimeError("Local-window session failed.") from state.error
    if state.result is None:
        raise RuntimeError("Local-window session returned no result.")
    expected_user_stop = user_closed and state.result.status in {
        "cancelled",
        "not_activated",
    }
    if state.result.status != "completed" and not expected_user_stop:
        reason = state.result.reason or str(state.result.error) or state.result.status
        raise RuntimeError(f"Local-window session failed: {reason}")
    return state.result


async def _run_session(
    *,
    state: _State,
    host: RuntimeHost,
    spec: DemoSpec,
    scenario: PreparedScenario,
    adapter: DemoAdapter,
    queue: NativeFrameQueue,
    output: NativeWindowOutputSpec,
    clock_fn: Callable[[], float],
) -> RunResult:
    start_s = clock_fn()
    source = NativeWindowInputSource(fps=output.fps)
    source.reset(start_v=start_s)
    transport = NoopTransportService()
    mode = NativeWindowRunMode(
        input_source=source,
        output_sink=NativeWindowOutputSink(
            queue=queue,
            batch_index=output.batch_index,
            view_index=output.view_index,
        ),
        transport=transport,
        clock_fn=clock_fn,
    )
    context = mode.create_run_context(
        spec=spec,
        adapter=adapter,
        host=host,
        model_warmup_plan=ModelWarmupPlan(),
    )
    with state.lock:
        state.input_source = source
        state.transport = transport
        pending_keys = tuple(state.pending_keys)
        state.pending_keys.clear()
    for event, key, timestamp_s in pending_keys:
        source.record_key(
            event=event,
            key=key,
            timestamp_s=max(start_s, timestamp_s),
        )
    if state.cancel_requested.is_set():
        transport.close()
    try:
        if state.cancel_requested.is_set():
            return RunResult(
                status="cancelled",
                reason="window closed before preload",
            )
        await asyncio.to_thread(host.preload)
        return await run_demo_session_async(
            context=context,
            spec=spec,
            scenario=scenario,
            adapter=adapter,
            run_mode=mode,
            pipeline=StepPipeline(),
        )
    finally:
        await context.close_async()


def _request_transport_close(state: _State) -> None:
    with state.lock:
        transport = state.transport
    if transport is not None:
        transport.close()


def _require_single_process() -> None:
    world_size = (
        dist.get_world_size()
        if dist.is_available() and dist.is_initialized()
        else int(os.environ.get("WORLD_SIZE", "1"))
    )
    if world_size != 1:
        raise RuntimeError("Local-window output requires one process.")


def _finish_native_session(
    *,
    state: _State,
    presenter: NativePresenter,
    queue: NativeFrameQueue,
    thread: threading.Thread,
    timeout_s: float,
) -> RuntimeError | None:
    errors: list[Exception] = []
    try:
        presenter.close()
    except Exception as exc:  # noqa: BLE001 - cleanup must continue after failure
        errors.append(exc)

    if not state.finished.is_set():
        state.deferred_cleanup.set()
    if not state.finished.wait(timeout_s):
        queue.close()
        return RuntimeError(
            f"Local-window worker did not stop within {timeout_s:.1f} seconds."
        )

    host = state.host
    if host is not None:
        close_error = _close_host_bounded(host, timeout_s=timeout_s)
        if close_error is not None:
            errors.append(close_error)
    queue.close()
    thread.join(min(timeout_s, 1.0))
    if errors:
        return RuntimeError(
            "Local-window cleanup failed: " + "; ".join(str(error) for error in errors)
        )
    return None


def _close_host_bounded(
    host: RuntimeHost,
    *,
    timeout_s: float,
) -> RuntimeError | None:
    finished = threading.Event()
    errors: list[Exception] = []

    def close() -> None:
        try:
            host.close()
        except Exception as exc:  # noqa: BLE001 - close thread reports failures
            errors.append(exc)
        finally:
            finished.set()

    thread = threading.Thread(target=close, name="native-window-close", daemon=True)
    thread.start()
    if not finished.wait(timeout_s):
        return RuntimeError(
            f"Local-window runtime did not close within {timeout_s:.1f} seconds."
        )
    if errors:
        return RuntimeError(str(errors[0]))
    return None


__all__ = [
    "NativePresenter",
    "run_native_window_presentation",
]
