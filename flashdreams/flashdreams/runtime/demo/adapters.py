# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Default adapter and input-provider implementations for simple demos."""

from __future__ import annotations

from typing import Any

from flashdreams.runtime import IdentityInputMapping, InferenceInput

from .session_inputs import (
    PreparedStep,
    ProviderCapabilities,
    UserInputWindow,
)
from .spec import DemoSpec, PreparedScenario


class StaticInputProvider:
    """Provide fixed session conditioning and no per-step controls."""

    capabilities = ProviderCapabilities(
        supports_realtime_clock=True,
        supports_recorded_input=True,
        deterministic_given_inputs=True,
    )
    """Capabilities of a fixed, deterministic conditioning provider."""

    def __init__(self, *, initial_inputs: InferenceInput) -> None:
        self._initial_inputs = initial_inputs

    def prepare_initial_input(self) -> InferenceInput:
        """Return the fixed session-global conditioning."""
        return self._initial_inputs

    def prepare_step(
        self, *, request: Any, user_window: UserInputWindow
    ) -> PreparedStep:
        """Return an empty input for each model step."""
        del request, user_window
        return PreparedStep(inference_input=InferenceInput())

    def reset(self, inputs: InferenceInput | None = None) -> None:
        """Optionally replace the session-global conditioning."""
        if inputs is not None:
            self._initial_inputs = inputs

    def close(self) -> None:
        """Release no resources."""


class ReplayWebRTCDemoAdapter:
    """Default demo behavior for adapters supporting replay and WebRTC."""

    def supported_input_modes(self) -> tuple[str, ...]:
        """Support the shared finite replay and WebRTC input modes."""
        return ("replay", "webrtc")

    def supported_output_modes(self) -> tuple[str, ...]:
        """Support the standard replay and WebRTC output modes."""
        return ("mp4", "null", "webrtc")

    def default_input_mapping(self) -> IdentityInputMapping:
        """Use canonical inputs without model-specific remapping."""
        return IdentityInputMapping()

    def create_model_input_provider(
        self, spec: DemoSpec, scenario: PreparedScenario
    ) -> StaticInputProvider:
        """Create the default fixed-conditioning input provider."""
        del spec
        return StaticInputProvider(initial_inputs=scenario.initial_inputs)


__all__ = ["ReplayWebRTCDemoAdapter", "StaticInputProvider"]
