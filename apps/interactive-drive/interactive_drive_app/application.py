# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""App-owned native loop over a reusable interactive model worker."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal

from flashdreams.runtime import (
    DRIVER_COMMAND,
    DeclaresMappingSchema,
    InferenceConfig,
    InputCanonicalizer,
    InteractiveInferenceWorker,
    InteractiveSessionEnded,
    InteractiveSessionJob,
    InteractiveStep,
    KeyboardToDriverCommand,
    require_output_compatibility,
)
from flashdreams.runtime.demo import (
    DemoAdapter,
    DemoRoute,
    DemoSpec,
    LocalWindowOutputSpec,
)
from flashdreams.runtime.demo.local_window import LocalWindowIO, build_local_window_io
from flashdreams.runtime.mapping import InputMapping
from flashdreams.runtime.metrics import MetricsRecorder, NullMetricsRecorder
from flashdreams.serving.presentation import (
    CompositeOverlay,
    HudOverlay,
    KeyEvent,
    NullOverlay,
)

from interactive_drive_app.input.wheel import WheelToDriverCommand
from interactive_drive_app.overlays.composition import build_driving_overlay
from interactive_drive_app.state import DrivingViewState

SessionAction = Literal[
    "completed",
    "stopped",
    "reset",
    "next",
    "previous",
    "exit",
    "closed",
]


@dataclass(frozen=True, kw_only=True, slots=True)
class DrivingSessionOutcome:
    """Terminal state returned to the app-owned scene loop."""

    session_id: str
    action: SessionAction


class InteractiveDriveApplication:
    """Run compatible driving sessions over one model runtime and native window."""

    def __init__(
        self,
        *,
        adapter: DemoAdapter,
        initial_spec: DemoSpec,
        overlay: HudOverlay | None = None,
        state: DrivingViewState | None = None,
        result_queue_size: int = 8,
    ) -> None:
        _require_driving_route(adapter=adapter, spec=initial_spec)
        self._adapter = adapter
        self._config = _require_config(initial_spec)
        self._state = state or DrivingViewState()
        self._controls = _SessionControls()
        chrome = overlay or build_driving_overlay(self._state)
        self._io = build_local_window_io(
            spec=initial_spec,
            overlay=CompositeOverlay(
                layers=(chrome, _SessionControlOverlay(self._controls))
            ),
            close_presenter_on_close=False,
            frame_projector=self._state.project_frame,
        )
        require_output_compatibility(
            produced=adapter.inference_output_schema,
            required=self._io.output.output_requirement,
        )
        self._worker = InteractiveInferenceWorker(
            adapter=adapter,
            config=self._config,
            result_queue_size=result_queue_size,
            runtime_factory=lambda: adapter.create_demo_runtime(initial_spec),
        )
        self._worker_started = False
        self._closed = False

    @property
    def io(self) -> LocalWindowIO:
        """Return the app-owned native input and output boundaries."""
        return self._io

    @property
    def state(self) -> DrivingViewState:
        """Return mutable application state read by the native chrome."""
        return self._state

    def run_session(
        self,
        *,
        spec: DemoSpec,
        session_id: str,
        metrics: MetricsRecorder | None = None,
    ) -> DrivingSessionOutcome:
        """Run one scene session while the application thread pumps the window."""
        if self._closed:
            raise RuntimeError("InteractiveDriveApplication is closed.")
        _require_driving_route(adapter=self._adapter, spec=spec)
        if _require_config(spec) != self._config:
            raise ValueError("Scene session config does not match the loaded runtime.")
        self._state.scene_label = str(
            spec.metadata.get("scene_label", self._state.scene_label)
        )
        self._state.variant_label = str(
            spec.metadata.get("variant_label", self._state.variant_label)
        )

        if not self._worker_started:
            self._worker.start(wait=False)
            self._worker_started = True
        prepared = self._adapter.prepare_session(spec)
        mapping = prepared.mapping or self._adapter.default_input_mapping()
        if mapping is None:
            raise ValueError(
                f"Adapter {type(self._adapter).__name__} provides no input mapping."
            )
        _require_driver_command_mapping(mapping)
        self._worker.wait_until_ready()

        self._io.output.reset_camera()
        self._controls.clear()
        self._io.output.open()
        self._worker.submit(
            InteractiveSessionJob(
                session_id=session_id,
                mapping=mapping,
                canonicalizer=InputCanonicalizer(
                    [
                        WheelToDriverCommand(),
                        KeyboardToDriverCommand(),
                    ]
                ),
                source_schema=self._io.user_inputs.source_schema,
                user_inputs=self._io.user_inputs,
                initial_inputs=prepared.initial_inputs,
                inference_input_schema=prepared.inference_input_schema,
                metrics=metrics or NullMetricsRecorder(),
            )
        )

        ended: InteractiveSessionEnded | None = None
        action: SessionAction = "completed"
        try:
            while ended is None:
                self._io.output.poll()
                if self._io.output.should_stop:
                    action = "closed"
                    self._worker.stop_session()
                requested = self._controls.consume()
                if requested is not None:
                    action = requested
                    self._worker.stop_session()
                event = self._worker.get_event(timeout_s=0.005)
                if isinstance(event, InteractiveStep):
                    self._io.output.write(event.result)
                elif isinstance(event, InteractiveSessionEnded):
                    ended = event
        finally:
            self._io.output.close()

        if ended.error is not None:
            raise RuntimeError(
                f"Interactive session {session_id!r} failed."
            ) from ended.error
        if action == "completed" and ended.stopped:
            action = "stopped"
        return DrivingSessionOutcome(session_id=session_id, action=action)

    def stop_session(self) -> None:
        """Request that the current scene stop after its active model step."""
        self._controls.request("stopped")
        self._worker.stop_session()

    def update_wheel(
        self,
        *,
        steer: float,
        throttle: float,
        brake: float,
        reverse: bool,
        stop: bool = False,
    ) -> None:
        """Publish one app-owned wheel snapshot to the active session."""
        self._io.user_inputs.append_wheel_state(
            steer=steer,
            throttle=throttle,
            brake=brake,
            reverse=reverse,
            stop=stop,
        )

    def close(self) -> None:
        """Close the model worker and app-owned presenter."""
        if self._closed:
            return
        self._closed = True
        try:
            self._worker.close()
        finally:
            self._io.output.shutdown()


class _SessionControls:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._action: SessionAction | None = None

    def request(self, action: SessionAction) -> None:
        with self._lock:
            self._action = action

    def consume(self) -> SessionAction | None:
        with self._lock:
            action, self._action = self._action, None
            return action

    def clear(self) -> None:
        with self._lock:
            self._action = None


class _SessionControlOverlay(NullOverlay):
    def __init__(self, controls: _SessionControls) -> None:
        self._controls = controls

    def on_key(self, event: KeyEvent) -> bool:
        controlled = {"r", "x", "tab", "backspace"}
        if event.action != "press":
            return event.key in controlled
        if event.key == "r":
            self._controls.request("reset")
            return True
        if event.key == "x":
            self._controls.request("exit")
            return True
        if event.key == "tab":
            self._controls.request("next")
            return True
        if event.key == "backspace":
            self._controls.request("previous")
            return True
        return False


def _require_driving_route(*, adapter: DemoAdapter, spec: DemoSpec) -> None:
    if not isinstance(spec.output, LocalWindowOutputSpec):
        raise TypeError("Interactive driving requires LocalWindowOutputSpec.")
    route = DemoRoute(
        input_mode=spec.input_mode,
        output_mode=spec.output.mode,
    )
    if route not in adapter.supported_routes():
        raise ValueError(
            f"Adapter {adapter.model_id!r} does not support driving route {route!r}."
        )


def _require_config(spec: DemoSpec) -> InferenceConfig:
    config = spec.config
    if config is None:
        raise RuntimeError("DemoSpec.config was not initialized.")
    return config


def _require_driver_command_mapping(mapping: InputMapping) -> None:
    if not isinstance(mapping, DeclaresMappingSchema):
        raise TypeError(
            "Interactive driving requires an input mapping with a declared schema."
        )
    if not any(
        DRIVER_COMMAND.is_satisfied_by(consumed)
        for consumed in mapping.mapping_schema.consumes
    ):
        raise ValueError(
            "Interactive driving input mapping must consume driver_command."
        )


__all__ = [
    "DrivingSessionOutcome",
    "InteractiveDriveApplication",
    "SessionAction",
]
