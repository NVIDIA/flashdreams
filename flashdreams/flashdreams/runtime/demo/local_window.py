# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared local-window demo construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from flashdreams.infra.video_output import VideoStepResult
from flashdreams.runtime.canonical import InputCanonicalizer
from flashdreams.runtime.demo.local_input import LocalWindowInputSource
from flashdreams.runtime.inputs import CanonicalModality
from flashdreams.runtime.interfaces import InferenceRuntime
from flashdreams.runtime.mapping import DeclaresMappingSchema, InputMapping
from flashdreams.runtime.metrics import MetricsRecorder, NullMetricsRecorder
from flashdreams.runtime.output import OutputArtifact
from flashdreams.runtime.output_schema import (
    RGB_VIDEO,
    OutputTargetRequirement,
    require_output_compatibility,
)
from flashdreams.runtime.runner import run_inference_session
from flashdreams.serving.presentation.base import HudOverlay, NullOverlay
from flashdreams.serving.presentation.local_window import (
    LocalWindowPresenter,
    WindowConfig,
)
from flashdreams.serving.presentation.output import (
    DisplayFrameProjector,
    LocalWindowVideoOutputTarget,
    PresenterFactory,
)

from .replay import _require_supported_route
from .spec import DemoAdapter, DemoSpec, LocalWindowOutputSpec


@runtime_checkable
class LocalWindowApp(Protocol):
    """A windowed demo that owns its own render loop.

    The same shape as the WebRTC server this sits beside: ``run`` blocks
    until the user closes the window, so the launcher hands off rather than
    driving frames itself.
    """

    def run(self) -> Any:
        """Open the window and drive frames until the demo exits."""
        ...


@runtime_checkable
class CreatesLocalWindowApp(Protocol):
    """Adapter extension that constructs an interactive local application."""

    def create_local_window_app(self, *, spec: DemoSpec) -> LocalWindowApp:
        """Build the application described by ``spec``."""
        ...


@dataclass(frozen=True, kw_only=True, slots=True)
class LocalWindowDemo:
    """Constructed local-window demo pieces, before or after running."""

    app: LocalWindowApp
    output: LocalWindowOutputSpec


@dataclass(slots=True)
class LocalWindowIO:
    """Shared native input and output pieces for the standard runtime loop."""

    user_inputs: LocalWindowInputSource
    """Session-relative native event source."""

    output: LocalWindowVideoOutputTarget
    """Decoded RGB video output target."""


def build_local_window_io(
    *,
    spec: DemoSpec,
    overlay: HudOverlay | None = None,
    presenter_factory: PresenterFactory = LocalWindowPresenter,
    close_presenter_on_close: bool = True,
    frame_projector: DisplayFrameProjector | None = None,
) -> LocalWindowIO:
    """Build native input and output boundaries described by ``spec``.

    Raises:
        TypeError: ``spec`` does not select local-window output.
    """
    output_spec = spec.output
    if not isinstance(output_spec, LocalWindowOutputSpec):
        raise TypeError("build_local_window_io requires LocalWindowOutputSpec output.")
    user_inputs = LocalWindowInputSource()
    chrome = overlay if output_spec.show_hud and overlay is not None else NullOverlay()
    return LocalWindowIO(
        user_inputs=user_inputs,
        output=LocalWindowVideoOutputTarget(
            overlay=user_inputs.compose_overlay(chrome),
            config=WindowConfig(
                width=output_spec.width,
                height=output_spec.height,
                title=output_spec.title,
            ),
            presenter_factory=presenter_factory,
            close_presenter_on_close=close_presenter_on_close,
            frame_projector=frame_projector,
        ),
    )


def build_local_window_demo(
    *,
    spec: DemoSpec,
    adapter: DemoAdapter,
) -> LocalWindowDemo:
    """Build the model-owned windowed app described by ``spec``.

    Raises:
        TypeError: ``spec`` does not select local-window output, or the
            adapter exposes no ``create_local_window_app`` factory.
    """
    if not isinstance(spec.output, LocalWindowOutputSpec):
        raise TypeError(
            "build_local_window_demo requires LocalWindowOutputSpec output."
        )
    _require_supported_route(spec=spec, supported=adapter.supported_routes())

    if not isinstance(adapter, CreatesLocalWindowApp):
        raise TypeError(
            f"Adapter {type(adapter).__name__} supports local-window output but "
            "does not implement create_local_window_app."
        )
    return LocalWindowDemo(
        app=adapter.create_local_window_app(spec=spec),
        output=spec.output,
    )


def run_local_window_demo(
    *,
    spec: DemoSpec,
    adapter: DemoAdapter,
) -> LocalWindowDemo:
    """Build and run a local-window demo, blocking until the window closes."""
    demo = build_local_window_demo(spec=spec, adapter=adapter)
    demo.app.run()
    return demo


def run_local_window_session(
    *,
    spec: DemoSpec,
    adapter: DemoAdapter,
    overlay: HudOverlay | None = None,
    runtime: InferenceRuntime | None = None,
    io: LocalWindowIO | None = None,
    metrics: MetricsRecorder | None = None,
    mapping: InputMapping | None = None,
    canonicalizer: InputCanonicalizer | None = None,
    required_modalities: tuple[CanonicalModality, ...] = (),
) -> tuple[OutputArtifact, ...]:
    """Run one plug-compatible native session through the standard loop.

    Passing ``runtime`` keeps the loaded model alive across calls. Passing an
    ``io`` built with ``close_presenter_on_close=False`` additionally lets the
    application own one native window across scene/session changes.

    Raises:
        ValueError: The adapter provides no input mapping.
    """
    if not isinstance(spec.output, LocalWindowOutputSpec):
        raise TypeError(
            "run_local_window_session requires LocalWindowOutputSpec output."
        )
    _require_supported_route(spec=spec, supported=adapter.supported_routes())
    require_output_compatibility(
        produced=adapter.inference_output_schema,
        required=OutputTargetRequirement(
            modalities=frozenset({RGB_VIDEO}),
            python_type=VideoStepResult,
        ),
    )
    prepared = adapter.prepare_session(spec)
    selected_mapping = mapping or prepared.mapping or adapter.default_input_mapping()
    if selected_mapping is None:
        raise ValueError(f"Adapter {type(adapter).__name__} provides no input mapping.")
    _require_mapping_modalities(
        mapping=selected_mapping,
        required=required_modalities,
    )
    selected_canonicalizer = canonicalizer or prepared.canonicalizer
    local_io = io or build_local_window_io(spec=spec, overlay=overlay)
    local_io.output.reset_camera()
    config = spec.config
    assert config is not None
    return run_inference_session(
        adapter=adapter,
        config=config,
        mapping=selected_mapping,
        canonicalizer=selected_canonicalizer,
        source_schema=local_io.user_inputs.source_schema,
        user_inputs=local_io.user_inputs,
        initial_inputs=prepared.initial_inputs,
        output=local_io.output,
        metrics=metrics or NullMetricsRecorder(),
        runtime=runtime,
        inference_input_schema=prepared.inference_input_schema,
    )


def _require_mapping_modalities(
    *,
    mapping: InputMapping,
    required: tuple[CanonicalModality, ...],
) -> None:
    if not required:
        return
    if not isinstance(mapping, DeclaresMappingSchema):
        raise TypeError(
            "This demo requires a mapping that declares its consumed canonical "
            "modalities."
        )
    missing = tuple(
        modality.name
        for modality in required
        if not any(
            modality.is_satisfied_by(consumed)
            for consumed in mapping.mapping_schema.consumes
        )
    )
    if missing:
        raise ValueError(
            "Selected input mapping does not consume required canonical "
            f"modalities: {missing}."
        )


__all__ = [
    "CreatesLocalWindowApp",
    "LocalWindowApp",
    "LocalWindowDemo",
    "LocalWindowIO",
    "build_local_window_demo",
    "build_local_window_io",
    "run_local_window_demo",
    "run_local_window_session",
]
