# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capability resolution and validation for shared demo runs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from flashdreams.runtime.inputs import UserInputCapability, UserInputSchema

from .run_modes import RunMode, RunModeCapabilities, SessionEdges
from .session_inputs import (
    BatchInputSource,
    ModelInputProvider,
    ProviderCapabilities,
    RealtimeInputSource,
)
from .spec import DemoAdapter, DemoSpec


@dataclass(frozen=True, kw_only=True, slots=True)
class ResolvedRunCapabilities:
    """Capabilities of one concrete provider/run-mode/session-edges pairing."""

    finite: bool
    deterministic: bool
    realtime: bool
    resettable: bool
    produces_artifacts: bool


def resolve_run_capabilities(
    *,
    spec: DemoSpec,
    provider: ModelInputProvider,
    session_edges: SessionEdges,
) -> ResolvedRunCapabilities:
    """Resolve concrete run capabilities from provider, edges, and config."""

    provider_capabilities = _provider_capabilities(provider)
    clock = session_edges.clock
    realtime = bool(getattr(clock, "is_realtime", False))
    deterministic_clock = (
        bool(getattr(clock, "is_deterministic", False)) if clock is not None else True
    )
    config = spec.config
    seeded = config is not None and config.seed is not None
    return ResolvedRunCapabilities(
        finite=bool(session_edges.input_source.is_finite),
        deterministic=(
            provider_capabilities.deterministic_given_inputs
            and bool(session_edges.input_source.is_deterministic)
            and deterministic_clock
            and seeded
        ),
        realtime=realtime,
        resettable=provider_capabilities.supports_reset,
        produces_artifacts=bool(session_edges.output_sink.produces_artifacts),
    )


def validate_resolved_run(
    *,
    spec: DemoSpec,
    adapter: DemoAdapter,
    provider: ModelInputProvider,
    run_mode: RunMode,
    session_edges: SessionEdges,
    resolved: ResolvedRunCapabilities,
) -> None:
    """Reject structurally incompatible provider/input/run-mode combinations."""

    del spec, adapter
    provider_capabilities = _provider_capabilities(provider)
    run_mode_capabilities = _run_mode_capabilities(run_mode)
    _validate_input_source_shape(
        run_mode_capabilities=run_mode_capabilities,
        session_edges=session_edges,
        resolved=resolved,
    )
    _validate_provider_modes(
        provider_capabilities=provider_capabilities,
        run_mode_capabilities=run_mode_capabilities,
        resolved=resolved,
    )
    _validate_user_input_schema(
        provider_schema=provider_capabilities.user_input_schema,
        source_schema=_input_source_user_input_schema(session_edges.input_source),
    )


def _validate_input_source_shape(
    *,
    run_mode_capabilities: RunModeCapabilities,
    session_edges: SessionEdges,
    resolved: ResolvedRunCapabilities,
) -> None:
    input_source = session_edges.input_source
    if run_mode_capabilities.realtime:
        if not isinstance(input_source, RealtimeInputSource):
            raise ValueError("Realtime run modes require a RealtimeInputSource.")
        if session_edges.clock is None or not resolved.realtime:
            raise ValueError("Realtime run modes require a realtime clock.")
        return
    if not isinstance(input_source, BatchInputSource):
        raise ValueError("Batch run modes require a BatchInputSource.")
    if resolved.realtime:
        raise ValueError("Batch run modes cannot use a realtime clock.")


def _validate_provider_modes(
    *,
    provider_capabilities: ProviderCapabilities,
    run_mode_capabilities: RunModeCapabilities,
    resolved: ResolvedRunCapabilities,
) -> None:
    if run_mode_capabilities.realtime and not (
        provider_capabilities.supports_realtime_clock
    ):
        raise ValueError("Provider does not support realtime input.")
    if run_mode_capabilities.requires_finite_input:
        if not resolved.finite:
            raise ValueError("Run mode requires finite input.")
        if not provider_capabilities.supports_recorded_input:
            raise ValueError("Provider does not support recorded input.")
    if resolved.produces_artifacts and not run_mode_capabilities.supports_artifacts:
        raise ValueError("Run mode does not support artifact output.")
    if (
        run_mode_capabilities.supports_interactive_events
        and not provider_capabilities.supports_realtime_clock
    ):
        raise ValueError("Interactive run mode requires realtime provider support.")


def _validate_user_input_schema(
    *,
    provider_schema: UserInputSchema,
    source_schema: UserInputSchema,
) -> None:
    missing = _missing_capabilities(
        required=provider_schema.declared_capabilities(),
        provided=source_schema,
    )
    if missing:
        names = ", ".join(
            f"{capability.event_type}[{','.join(sorted(capability.payload_fields))}]"
            for capability in missing
        )
        raise ValueError(
            f"Input source does not satisfy provider raw user input schema: {names}."
        )


def _missing_capabilities(
    *,
    required: Sequence[UserInputCapability],
    provided: UserInputSchema,
) -> tuple[UserInputCapability, ...]:
    return tuple(
        capability for capability in required if not provided.supports(capability)
    )


def _provider_capabilities(provider: ModelInputProvider) -> ProviderCapabilities:
    capabilities = getattr(provider, "capabilities", None)
    if not isinstance(capabilities, ProviderCapabilities):
        raise TypeError(
            "ModelInputProvider.capabilities must be a ProviderCapabilities "
            f"instance, got {type(capabilities).__name__}."
        )
    return capabilities


def _run_mode_capabilities(run_mode: RunMode) -> RunModeCapabilities:
    capabilities = getattr(run_mode, "capabilities", None)
    if not isinstance(capabilities, RunModeCapabilities):
        raise TypeError(
            "RunMode.capabilities must be a RunModeCapabilities instance, "
            f"got {type(capabilities).__name__}."
        )
    return capabilities


def _input_source_user_input_schema(input_source: object) -> UserInputSchema:
    schema = getattr(input_source, "user_input_schema", None)
    if not isinstance(schema, UserInputSchema):
        raise TypeError(
            "InputSource.user_input_schema must be a UserInputSchema instance, "
            f"got {type(schema).__name__}."
        )
    return schema


__all__ = [
    "ResolvedRunCapabilities",
    "resolve_run_capabilities",
    "validate_resolved_run",
]
