# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared replay demo runner."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from flashdreams.runtime.metrics import MetricsRecorder, NullMetricsRecorder
from flashdreams.runtime.output import OutputArtifact, OutputTarget
from flashdreams.runtime.runner import run_inference_session

from .outputs import build_output_target
from .spec import DemoAdapter, DemoSpec, OutputSpec, WebRTCOutputSpec

OutputTargetFactory = Callable[[OutputSpec], OutputTarget]
InferenceSessionRunner = Callable[..., Sequence[OutputArtifact]]


def run_replay_demo(
    *,
    spec: DemoSpec,
    adapter: DemoAdapter,
    output_target_factory: OutputTargetFactory = build_output_target,
    metrics: MetricsRecorder | None = None,
    runner: InferenceSessionRunner = run_inference_session,
) -> tuple[OutputArtifact, ...]:
    """Run one prepared demo scenario through the shared runtime runner."""
    _require_supported_mode(
        mode=spec.input_mode,
        supported=adapter.supported_input_modes(),
        label="input_mode",
    )
    if spec.input_mode != "replay":
        raise ValueError(
            "run_replay_demo requires input_mode='replay', "
            f"got input_mode={spec.input_mode!r}."
        )
    _require_supported_mode(
        mode=spec.output.mode,
        supported=adapter.supported_output_modes(),
        label="output.mode",
    )
    if isinstance(spec.output, WebRTCOutputSpec):
        raise ValueError("run_replay_demo does not support WebRTC output.")

    prepared = adapter.prepare_scenario(spec)
    mapping = prepared.mapping or adapter.default_input_mapping()
    if mapping is None:
        raise ValueError(
            "Demo scenario did not provide an input mapping, and the adapter "
            "has no default input mapping."
        )
    if spec.config is None:
        raise RuntimeError("DemoSpec.config was not initialized.")

    output = output_target_factory(spec.output)
    metrics_recorder = metrics or NullMetricsRecorder()
    return tuple(
        runner(
            adapter=adapter,
            config=spec.config,
            mapping=mapping,
            canonicalizer=prepared.canonicalizer,
            source_schema=prepared.source_schema,
            user_inputs=prepared.user_inputs,
            initial_inputs=prepared.initial_inputs,
            output=output,
            metrics=metrics_recorder,
        )
    )


def _require_supported_mode(
    *,
    mode: str,
    supported: tuple[str, ...],
    label: str,
) -> None:
    if mode in supported:
        return
    supported_text = ", ".join(repr(each) for each in supported) or "<none>"
    raise ValueError(
        f"Unsupported demo {label}={mode!r}; supported modes: {supported_text}."
    )


__all__ = [
    "InferenceSessionRunner",
    "OutputTargetFactory",
    "run_replay_demo",
]
