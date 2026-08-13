# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from flashdreams.runtime import (
    InferenceInput,
    StepRequirements,
    UserInputEvent,
    UserInputs,
)
from flashdreams.runtime.demo import (
    REALTIME_SKIPPED_INPUTS_METADATA_KEY,
    UserInputWindow,
)
from triangle_app import TriangleInputProvider

pytestmark = pytest.mark.ci_cpu


def test_triangle_app_maps_keyboard_input_to_model_color() -> None:
    provider = TriangleInputProvider(
        initial_inputs=InferenceInput(),
        interactive=True,
    )

    prepared = provider.prepare_step(
        request=StepRequirements(step_index=0),
        user_window=UserInputWindow(
            start_s=0.0,
            end_s=1.0,
            inputs=UserInputs(
                events=(
                    UserInputEvent(
                        timestamp_s=0.5,
                        event_type="key_down",
                        payload={"key": "g"},
                    ),
                )
            ),
        ),
    )

    assert prepared.inference_input is not None
    assert prepared.inference_input.step["color"] == (64, 255, 128)


def test_triangle_app_applies_skipped_keyboard_input() -> None:
    provider = TriangleInputProvider(
        initial_inputs=InferenceInput(),
        interactive=True,
    )
    skipped = UserInputs(
        events=(
            UserInputEvent(
                timestamp_s=0.5,
                event_type="key_down",
                payload={"key": "b"},
            ),
        )
    )

    prepared = provider.prepare_step(
        request=StepRequirements(step_index=0),
        user_window=UserInputWindow(
            start_s=1.0,
            end_s=2.0,
            metadata={REALTIME_SKIPPED_INPUTS_METADATA_KEY: skipped},
        ),
    )

    assert prepared.inference_input is not None
    assert prepared.inference_input.step["color"] == (64, 128, 255)
