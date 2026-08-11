# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot model-input providers for shared demo run modes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from flashdreams.runtime import (
    CanonicalInputs,
    InferenceInput,
    InferenceInputSchema,
    StepRequest,
    StepRequirements,
    TimeWindow,
    UserInputs,
)
from flashdreams.runtime.demo import (
    PreparedScenario,
    PreparedStep,
    ProviderCapabilities,
    UserInputWindow,
)
from flashdreams.serving.webrtc.services import (
    WEBRTC_SKIPPED_INPUTS_METADATA_KEY,
    WEBRTC_SKIPPED_WINDOW_METADATA_KEY,
)
from lingbot.input_mapping import CAMERA_COMMAND, LingbotInputMapping
from lingbot.runtime import LingbotModelAdapter


class LingbotInputProvider:
    """Convert shared user-input windows into Lingbot model inputs.

    Lingbot's existing mapping API still accepts the legacy ``StepRequest``
    shape, because the runtime owns frame-start metadata today. This provider is
    the model-owned bridge from shared demo drivers to that mapping boundary;
    the bridge stays here so WebRTC transport code never has to know Lingbot's
    camera or prompt semantics.
    """

    def __init__(
        self,
        *,
        scenario: PreparedScenario,
        inference_input_schema: InferenceInputSchema | None = None,
    ) -> None:
        mapping = scenario.mapping
        if not isinstance(mapping, LingbotInputMapping):
            raise TypeError(
                "LingbotInputProvider requires PreparedScenario.mapping to be "
                f"LingbotInputMapping, got {type(mapping).__name__}."
            )
        if inference_input_schema is None:
            inference_input_schema = LingbotModelAdapter().inference_input_schema

        self.capabilities = ProviderCapabilities(
            supports_realtime_clock=_supports_realtime_clock(mapping),
            supports_recorded_input=True,
            supports_reset=True,
            deterministic_given_inputs=True,
            user_input_schema=scenario.source_schema,
            inference_input_schema=inference_input_schema,
        )
        self._scenario = scenario
        self._mapping = mapping
        self._step_base_inputs = InferenceInput(
            step=scenario.initial_inputs.step,
            metadata=scenario.initial_inputs.metadata,
        )
        self._next_frame_start = 0
        self._closed = False

    def prepare_initial_input(self) -> InferenceInput:
        self._require_open()
        self._reset_state()
        return self._mapping.map_global_conditioning_inputs(
            canonical_inputs=CanonicalInputs(),
            inference_input=self._scenario.initial_inputs,
        )

    def prepare_step(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
    ) -> PreparedStep:
        self._require_open()
        if user_window.control is not None:
            return PreparedStep(control=user_window.control)

        self._advance_skipped_input_state(user_window)
        legacy_request = self._legacy_step_request(
            request=request,
            user_window=user_window,
        )
        assert legacy_request.user_input_window is not None
        canonical_inputs = self._scenario.canonicalizer.canonicalize(
            user_window.inputs,
            window=legacy_request.user_input_window,
            source_schema=self._scenario.source_schema,
        )
        inference_input = self._mapping.map_step_inputs(
            canonical_inputs=canonical_inputs,
            inference_input=self._step_base_inputs,
            request=legacy_request,
        )
        self._next_frame_start = _required_int(
            legacy_request.metadata,
            "frame_start",
        ) + _required_positive_int(legacy_request.metadata, "num_frames")
        return PreparedStep(inference_input=inference_input)

    def reset(self, inputs: InferenceInput | None = None) -> None:
        del inputs
        self._require_open()
        self._reset_state()

    def close(self) -> None:
        self._closed = True

    def _legacy_step_request(
        self,
        *,
        request: StepRequirements,
        user_window: UserInputWindow,
    ) -> StepRequest:
        metadata: dict[str, Any] = dict(request.metadata)
        metadata["num_frames"] = _metadata_positive_int(
            metadata,
            "num_frames",
            default=request.input_frame_count,
        )
        metadata["frame_start"] = _metadata_int(
            metadata,
            "frame_start",
            default=self._next_frame_start,
        )
        return StepRequest(
            step_index=request.step_index,
            inference_input_schema=request.inference_input_schema,
            user_input_window=TimeWindow(
                start_s=user_window.start_s,
                end_s=user_window.end_s,
            ),
            metadata=metadata,
        )

    def _reset_state(self) -> None:
        self._scenario.canonicalizer.reset()
        self._mapping.reset()
        self._next_frame_start = 0

    def _advance_skipped_input_state(self, user_window: UserInputWindow) -> None:
        skipped_inputs = user_window.metadata.get(WEBRTC_SKIPPED_INPUTS_METADATA_KEY)
        skipped_window = user_window.metadata.get(WEBRTC_SKIPPED_WINDOW_METADATA_KEY)
        if not isinstance(skipped_inputs, UserInputs):
            return
        if not isinstance(skipped_window, tuple) or len(skipped_window) != 2:
            return
        start_value, end_value = skipped_window
        if not isinstance(start_value, int | float) or not isinstance(
            end_value,
            int | float,
        ):
            return
        start_s = float(start_value)
        end_s = float(end_value)
        if end_s <= start_s:
            return
        self._scenario.canonicalizer.canonicalize(
            skipped_inputs,
            window=TimeWindow(start_s=start_s, end_s=end_s),
            source_schema=self._scenario.source_schema,
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("LingbotInputProvider is closed.")


def _supports_realtime_clock(mapping: LingbotInputMapping) -> bool:
    return any(
        modality.name == CAMERA_COMMAND.name
        for modality in mapping.mapping_schema.consumes
    )


def _metadata_int(
    metadata: Mapping[str, Any],
    name: str,
    *,
    default: int,
) -> int:
    if name not in metadata:
        return default
    return _required_int(metadata, name)


def _metadata_positive_int(
    metadata: Mapping[str, Any],
    name: str,
    *,
    default: int,
) -> int:
    if name not in metadata:
        if default <= 0:
            raise ValueError(f"StepRequirements.{name} fallback must be > 0.")
        return default
    return _required_positive_int(metadata, name)


def _required_int(metadata: Mapping[str, Any], name: str) -> int:
    value = metadata[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Step metadata {name!r} must be an integer.")
    return value


def _required_positive_int(metadata: Mapping[str, Any], name: str) -> int:
    value = _required_int(metadata, name)
    if value <= 0:
        raise ValueError(f"Step metadata {name!r} must be > 0.")
    return value


__all__ = ["LingbotInputProvider"]
