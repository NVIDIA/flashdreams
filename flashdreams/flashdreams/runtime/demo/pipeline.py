# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared per-step pipeline for demo session drivers."""

from __future__ import annotations

from dataclasses import dataclass, field

from flashdreams.runtime.interfaces import InferenceSession
from flashdreams.runtime.types import StepRequirements, StepResult

from .outputs import OutputDecision, OutputSink
from .run_modes import SessionMetricsRecorder
from .session_inputs import ControlDecision, ModelInputProvider, UserInputWindow


@dataclass(frozen=True, kw_only=True, slots=True)
class StepOutcome:
    """Combined output and control result from one shared model step."""

    output: OutputDecision = field(default_factory=OutputDecision)
    control: ControlDecision = field(default_factory=ControlDecision)


class StepPipeline:
    """Shared invariant for provider conversion, model step, output, and metrics."""

    def execute_step(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
        provider: ModelInputProvider,
        session: InferenceSession,
        output: OutputSink,
        metrics: SessionMetricsRecorder,
    ) -> StepOutcome:
        prepared = provider.prepare_step(
            request=request,
            user_window=user_window,
        )
        if prepared.control.reset or prepared.control.close_session:
            metrics.record_control(
                request=request,
                user_window=user_window,
                control=prepared.control,
            )
            return StepOutcome(control=prepared.control)
        if prepared.inference_input is None:
            raise RuntimeError("ModelInputProvider returned no inference input.")

        result = session.step(prepared.inference_input)
        if not isinstance(result, StepResult):
            raise TypeError(
                "InferenceSession.step must return StepResult, "
                f"got {type(result).__name__}."
            )
        decision = output.write(result)
        if not isinstance(decision, OutputDecision):
            raise TypeError(
                "OutputSink.write must return OutputDecision, "
                f"got {type(decision).__name__}."
            )
        metrics.record_step(
            request=request,
            user_window=user_window,
            inference_input=prepared.inference_input,
            result=result,
            decision=decision,
        )
        return StepOutcome(output=decision)


__all__ = ["StepOutcome", "StepPipeline"]
