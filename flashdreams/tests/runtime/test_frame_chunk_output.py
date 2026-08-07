# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pydantic validation tests for builtin frame-chunk outputs."""

from typing import Any

import pytest
import torch
from flashdreams.runtime.builtin.inference_output.frame_chunk import (
    FrameChunkOutput,
)
from pydantic import TypeAdapter, ValidationError

pytestmark = pytest.mark.ci_cpu

_FRAME_CHUNK_OUTPUT_ADAPTER = TypeAdapter(FrameChunkOutput)


## Valid frame chunks


def test_frame_chunk_output_accepts_valid_timing_metadata() -> None:
    """Verify Pydantic retains valid frame data and presentation timing."""
    frame_chunk = torch.zeros(1, 1, 4, 3, 8, 8)

    output = _FRAME_CHUNK_OUTPUT_ADAPTER.validate_python(
        {"value": frame_chunk, "start_timestamp": 1.25, "fps": 30.0}
    )

    assert output.value is frame_chunk
    assert output.start_timestamp == pytest.approx(1.25)
    assert output.fps == pytest.approx(30.0)
    assert output.frame_present_time == pytest.approx(1.0 / 30.0)


## Invalid frame chunks


# Each payload isolates a missing field, invalid value, or unsupported extra field.
@pytest.mark.parametrize(
    "inference_output",
    [
        {},
        {"value": torch.zeros(1), "start_timestamp": 0.0},
        {"value": torch.zeros(1), "fps": 30.0},
        {"value": "not-a-tensor", "start_timestamp": 0.0, "fps": 30.0},
        {"value": torch.zeros(1), "start_timestamp": -1.0, "fps": 30.0},
        {"value": torch.zeros(1), "start_timestamp": float("nan"), "fps": 30.0},
        {"value": torch.zeros(1), "start_timestamp": 0.0, "fps": 0.0},
        {"value": torch.zeros(1), "start_timestamp": 0.0, "fps": float("inf")},
        {
            "value": torch.zeros(1),
            "start_timestamp": 0.0,
            "fps": 30.0,
            "extra": True,
        },
    ],
)
def test_frame_chunk_output_rejects_invalid_payloads(
    inference_output: Any,
) -> None:
    """Verify Pydantic rejects invalid frame data and timing metadata."""
    with pytest.raises(ValidationError):
        _FRAME_CHUNK_OUTPUT_ADAPTER.validate_python(inference_output)
