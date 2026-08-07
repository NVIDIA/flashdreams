# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from flashdreams.infra.video_output import VideoStepResult
from flashdreams.runtime import (
    RGB_VIDEO,
    InferenceOutputSchema,
    OutputTargetRequirement,
    check_output_compatibility,
    require_output_compatibility,
)
from flashdreams.serving.webrtc.runtime import WebRTCStepResult

pytestmark = pytest.mark.ci_cpu


def test_matching_modality_and_type_are_compatible() -> None:
    compatibility = check_output_compatibility(
        produced=InferenceOutputSchema(
            modality=RGB_VIDEO,
            python_type=VideoStepResult,
        ),
        required=OutputTargetRequirement(
            modalities=frozenset({RGB_VIDEO}),
            python_type=VideoStepResult,
        ),
    )

    assert compatibility.compatible
    assert compatibility.reasons == ()


def test_subclass_result_satisfies_a_base_type_requirement() -> None:
    compatibility = check_output_compatibility(
        produced=InferenceOutputSchema(
            modality=RGB_VIDEO,
            python_type=WebRTCStepResult,
        ),
        required=OutputTargetRequirement(
            modalities=frozenset({RGB_VIDEO}),
            python_type=VideoStepResult,
        ),
    )

    assert compatibility.compatible


def test_wrong_modality_is_rejected_before_runtime_values_exist() -> None:
    with pytest.raises(ValueError, match="modality 'depth' is not accepted"):
        require_output_compatibility(
            produced=InferenceOutputSchema(
                modality="depth",
                python_type=VideoStepResult,
            ),
            required=OutputTargetRequirement(
                modalities=frozenset({RGB_VIDEO}),
                python_type=VideoStepResult,
            ),
        )


def test_wrong_runtime_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="str is not a subclass"):
        require_output_compatibility(
            produced=InferenceOutputSchema(
                modality=RGB_VIDEO,
                python_type=str,
            ),
            required=OutputTargetRequirement(
                modalities=frozenset({RGB_VIDEO}),
                python_type=VideoStepResult,
            ),
        )


def test_disjoint_tensor_layouts_are_rejected() -> None:
    with pytest.raises(ValueError, match="do not intersect"):
        require_output_compatibility(
            produced=InferenceOutputSchema(
                modality=RGB_VIDEO,
                python_type=VideoStepResult,
                layouts=frozenset({"bvtchw"}),
            ),
            required=OutputTargetRequirement(
                modalities=frozenset({RGB_VIDEO}),
                python_type=VideoStepResult,
                layouts=frozenset({"btchw"}),
            ),
        )


def test_null_requirement_accepts_every_output() -> None:
    compatibility = check_output_compatibility(
        produced=InferenceOutputSchema(
            modality="custom/model-output",
            python_type=dict,
        ),
        required=OutputTargetRequirement(modalities=None),
    )

    assert compatibility.compatible
