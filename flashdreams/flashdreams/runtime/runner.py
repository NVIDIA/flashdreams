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
    TimeWindow,
    UserInputs,
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
from flashdreams.runtime.output import OutputArtifact, OutputTarget
from flashdreams.runtime.types import StepResult

_DEFAULT_SESSION_HORIZON_S = 3600.0


def run_inference_session(
    *,
    adapter: ModelAdapter,
    config: InferenceConfig,
    mapping: InputMapping,
    canonicalizer: InputCanonicalizer,
    source_schema: UserInputSchema,
    user_inputs: UserInputs,
    initial_inputs: InferenceInput,
    output: OutputTarget,
    metrics: MetricsRecorder,
) -> tuple[OutputArtifact, ...]:
    """Run one sequential inference session through the standard loop.

    This v0 loop intentionally handles one adapter/runtime/session, one selected
    input mapping, one replay/live input batch, one output target, and one
    metrics recorder. It is synchronous and owns only orchestration.
    """

    runtime: InferenceRuntime | None = None
    session: InferenceSession | None = None
    output_opened = False
    output_artifacts: tuple[OutputArtifact, ...] = ()
    primary_error: BaseException | None = None

    try:
        adapter.validate_config(config)
        canonical_schema = canonicalizer.canonical_schema(source_schema)
        _check_declared_mapping_compatibility(
            mapping=mapping,
            canonical_schema=canonical_schema,
            adapter=adapter,
        )
        mapping.validate(
            canonical_schema=canonical_schema,
            inference_input_schema=adapter.inference_input_schema,
        )
        canonicalizer.reset()
        mapped_initial_inputs = mapping.map_global_conditioning_inputs(
            canonical_inputs=CanonicalInputs(),
            inference_input=initial_inputs,
        )
        runtime = adapter.create_runtime(config)
        session = runtime.start_session(mapped_initial_inputs)
        output.open()
        output_opened = True
        step_base_inputs = InferenceInput(
            step=initial_inputs.step,
            metadata=initial_inputs.metadata,
        )

        while (request := session.next_step_request()) is not None:
            step_inputs = mapping.map_step_inputs(
                canonical_inputs=canonicalizer.canonicalize(
                    user_inputs,
                    window=request.user_input_window
                    or _all_user_inputs_window(user_inputs),
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
            runtime=runtime,
            metrics=metrics,
        )
        if cleanup_error is not None and primary_error is None:
            raise cleanup_error

    return output_artifacts


def _check_declared_mapping_compatibility(
    *,
    mapping: InputMapping,
    canonical_schema: CanonicalInputSchema,
    adapter: ModelAdapter,
) -> None:
    if not isinstance(mapping, DeclaresMappingSchema):
        return
    compatibility = check_mapping_compatibility(
        canonical_schema=canonical_schema,
        inference_input_schema=adapter.inference_input_schema,
        mapping_schema=mapping.mapping_schema,
    )
    compatibility.raise_if_incompatible()


def _all_user_inputs_window(user_inputs: UserInputs) -> TimeWindow:
    if not user_inputs.events:
        return TimeWindow(start_s=0.0, end_s=_DEFAULT_SESSION_HORIZON_S)
    return TimeWindow(
        start_s=0.0,
        end_s=max(
            _DEFAULT_SESSION_HORIZON_S,
            math.nextafter(user_inputs.events[-1].timestamp_s, math.inf),
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
