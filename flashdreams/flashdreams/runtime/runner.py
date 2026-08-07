# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal synchronous standard runner for the runtime API."""

from __future__ import annotations

import math

from flashdreams.runtime.canonical import InputCanonicalizer
from flashdreams.runtime.config import InferenceConfig
from flashdreams.runtime.inputs import (
    CanonicalInputs,
    CanonicalInputSchema,
    InferenceInput,
    InferenceInputSchema,
    TimeWindow,
    UserInputSchema,
)
from flashdreams.runtime.interfaces import (
    InferenceRuntime,
    InferenceSession,
    ModelAdapter,
)
from flashdreams.runtime.mapping import (
    DeclaresMappingSchema,
    InputMapping,
    check_mapping_compatibility,
)
from flashdreams.runtime.metrics import MetricsRecorder
from flashdreams.runtime.output import (
    OutputArtifact,
    OutputTarget,
    PollableOutputTarget,
)
from flashdreams.runtime.output_schema import (
    DeclaresInferenceOutput,
    DeclaresOutputRequirement,
    require_output_compatibility,
)
from flashdreams.runtime.sources import SessionAwareUserInputSource, UserInputSource
from flashdreams.runtime.types import StepResult

_DEFAULT_SESSION_HORIZON_S = 3600.0


def run_inference_session(
    *,
    adapter: ModelAdapter,
    config: InferenceConfig,
    mapping: InputMapping,
    canonicalizer: InputCanonicalizer,
    source_schema: UserInputSchema,
    user_inputs: UserInputSource,
    initial_inputs: InferenceInput,
    output: OutputTarget,
    metrics: MetricsRecorder,
    runtime: InferenceRuntime | None = None,
    inference_input_schema: InferenceInputSchema | None = None,
) -> tuple[OutputArtifact, ...]:
    """Run one sequential inference session through the standard loop.

    This v0 loop intentionally handles one adapter/runtime/session, one selected
    input mapping, one input source, one output target, and one metrics
    recorder. It is synchronous and owns only orchestration.

    ``user_inputs`` is windowed once per step rather than read whole, so a
    fully-known replay batch and a live queue drive the same loop. See
    :class:`~flashdreams.runtime.sources.UserInputSource`.

    Passing a preloaded ``runtime`` lets an application reuse heavyweight model
    state across sequential sessions. ``None`` creates, owns, and closes one
    runtime for this call.
    """

    owns_runtime = runtime is None
    session: InferenceSession | None = None
    output_opened = False
    output_artifacts: tuple[OutputArtifact, ...] = ()
    primary_error: BaseException | None = None

    try:
        adapter.validate_config(config)
        if runtime is not None:
            _validate_reused_runtime(runtime=runtime, config=config)
        selected_input_schema = inference_input_schema or adapter.inference_input_schema
        canonical_schema = canonicalizer.canonical_schema(source_schema)
        _check_declared_mapping_compatibility(
            mapping=mapping,
            canonical_schema=canonical_schema,
            inference_input_schema=selected_input_schema,
        )
        mapping.validate(
            canonical_schema=canonical_schema,
            inference_input_schema=selected_input_schema,
        )
        _check_declared_output_compatibility(adapter=adapter, output=output)
        canonicalizer.reset()
        mapped_initial_inputs = mapping.map_global_conditioning_inputs(
            canonical_inputs=CanonicalInputs(),
            inference_input=initial_inputs,
        )
        if runtime is None:
            runtime = adapter.create_runtime(config)
        session = runtime.start_session(mapped_initial_inputs)
        if isinstance(user_inputs, SessionAwareUserInputSource):
            user_inputs.start_session()
        output.open()
        output_opened = True
        step_base_inputs = InferenceInput(
            step=initial_inputs.step,
            metadata=initial_inputs.metadata,
        )

        default_window = _default_user_input_window(user_inputs)

        while True:
            if isinstance(output, PollableOutputTarget):
                output.poll()
                if output.should_stop:
                    break
            request = session.next_step_request()
            if request is None:
                break
            window = request.user_input_window or default_window
            step_inputs = mapping.map_step_inputs(
                canonical_inputs=canonicalizer.canonicalize(
                    user_inputs.window(window),
                    window=window,
                    source_schema=source_schema,
                ),
                inference_input=step_base_inputs,
                request=request,
            )
            result = session.step(step_inputs)
            output.write(result)
            _record_timing_metrics(metrics, result)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_error, output_artifacts = _close_run_resources(
            output=output if output_opened else None,
            session=session,
            runtime=runtime if owns_runtime else None,
            metrics=metrics,
        )
        if cleanup_error is not None and primary_error is None:
            raise cleanup_error

    return output_artifacts


def _validate_reused_runtime(
    *,
    runtime: InferenceRuntime,
    config: InferenceConfig,
) -> None:
    runtime_config = getattr(runtime, "config", None)
    if not isinstance(runtime_config, InferenceConfig):
        raise TypeError(
            "A reused runtime must expose the InferenceConfig it was created from "
            "as runtime.config."
        )
    if runtime_config != config:
        raise ValueError(
            "Reused runtime config does not match this session config: "
            f"runtime={runtime_config!r}, session={config!r}."
        )


def _check_declared_output_compatibility(
    *,
    adapter: ModelAdapter,
    output: OutputTarget,
) -> None:
    if not isinstance(adapter, DeclaresInferenceOutput):
        return
    if not isinstance(output, DeclaresOutputRequirement):
        return
    require_output_compatibility(
        produced=adapter.inference_output_schema,
        required=output.output_requirement,
    )


def _check_declared_mapping_compatibility(
    *,
    mapping: InputMapping,
    canonical_schema: CanonicalInputSchema,
    inference_input_schema: InferenceInputSchema,
) -> None:
    if not isinstance(mapping, DeclaresMappingSchema):
        return
    compatibility = check_mapping_compatibility(
        canonical_schema=canonical_schema,
        inference_input_schema=inference_input_schema,
        mapping_schema=mapping.mapping_schema,
    )
    compatibility.raise_if_incompatible()


def _default_user_input_window(user_inputs: UserInputSource) -> TimeWindow:
    """Window used for sessions that do not request one per step.

    A source holding a fully-known batch exposes its events, so the horizon is
    stretched past the last one to keep long replay traces intact. A live queue
    has no final event, so the flat horizon applies.
    """
    events = getattr(user_inputs, "events", ())
    if not events:
        return TimeWindow(start_s=0.0, end_s=_DEFAULT_SESSION_HORIZON_S)
    return TimeWindow(
        start_s=0.0,
        end_s=max(
            _DEFAULT_SESSION_HORIZON_S,
            math.nextafter(events[-1].timestamp_s, math.inf),
        ),
    )


def _record_timing_metrics(metrics: MetricsRecorder, result: StepResult) -> None:
    for name, value in result.metrics.items():
        if not name.endswith("_s") or isinstance(value, bool):
            continue
        sample_name = name[:-2] or name
        metrics.record_timing(
            sample_name,
            float(value),
            step_index=result.step_index,
        )


def _close_run_resources(
    *,
    output: OutputTarget | None,
    session: InferenceSession | None,
    runtime: InferenceRuntime | None,
    metrics: MetricsRecorder,
) -> tuple[BaseException | None, tuple[OutputArtifact, ...]]:
    cleanup_error: BaseException | None = None
    artifacts: tuple[OutputArtifact, ...] = ()

    def remember_error(exc: BaseException) -> None:
        nonlocal cleanup_error
        if cleanup_error is None:
            cleanup_error = exc

    if output is not None:
        try:
            artifacts = tuple(output.close())
        except BaseException as exc:
            remember_error(exc)

    if session is not None:
        try:
            session.close()
        except BaseException as exc:
            remember_error(exc)

    if runtime is not None:
        try:
            runtime.close()
        except BaseException as exc:
            remember_error(exc)

    try:
        metrics.close()
    except BaseException as exc:
        remember_error(exc)

    return cleanup_error, artifacts


__all__ = ["run_inference_session"]
