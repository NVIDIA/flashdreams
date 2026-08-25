# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for typed tensor artifacts and their NumPy output sink."""

from pathlib import Path

import numpy as np
import pytest
import torch

from flashdreams.runtime_v2.session_desc import SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.tensor_artifact import (
    TensorArtifactOutput,
    TensorArtifactSchema,
)
from flashdreams.runtime_v2.tensor_artifact_output_sink import (
    TensorArtifactOutputSink,
)
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

pytestmark = pytest.mark.ci_cpu

_ACTIONS = TensorArtifactSchema(
    name="actions",
    dimension_names=("step", "channel"),
    concatenate_axis=0,
)
"""Robot-action-like schema used to exercise generic tensor routing."""


def _session_desc() -> SessionDesc:
    return SessionDesc(tensor_artifact_schemas=(_ACTIONS,))


def _result(step_index: int, actions: torch.Tensor) -> StepResult:
    return StepResult(
        step_index=step_index,
        output=torch.zeros(1, 3, 1, 2, 2),
        frame_count=1,
        output_layout=VideoTensorLayout.bcthw,
        tensor_artifacts=(TensorArtifactOutput(schema=_ACTIONS, tensor=actions),),
    )


def test_tensor_artifact_sink_concatenates_steps_and_writes_once(
    tmp_path: Path,
) -> None:
    sink = TensorArtifactOutputSink(tmp_path)
    sink.open(_session_desc())
    sink.write(_result(0, torch.tensor([[1.0, 2.0]])))
    sink.write(_result(1, torch.tensor([[3.0, 4.0], [5.0, 6.0]])))

    sink.close()
    first_bytes = (tmp_path / "actions.npy").read_bytes()
    sink.close()

    np.testing.assert_array_equal(
        np.load(tmp_path / "actions.npy"),
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
    )
    assert (tmp_path / "actions.npy").read_bytes() == first_bytes
    assert not list(tmp_path.glob("*.tmp"))


def test_tensor_artifact_sink_rejects_undeclared_outputs(tmp_path: Path) -> None:
    other = TensorArtifactSchema(
        name="other",
        dimension_names=("step", "channel"),
    )
    result = _result(0, torch.zeros(1, 2))
    result = StepResult(
        step_index=result.step_index,
        output=result.output,
        frame_count=result.frame_count,
        output_layout=result.output_layout,
        tensor_artifacts=(
            TensorArtifactOutput(schema=other, tensor=torch.zeros(1, 2)),
        ),
    )
    sink = TensorArtifactOutputSink(tmp_path)
    sink.open(_session_desc())

    with pytest.raises(ValueError, match="not declared"):
        sink.write(result)


def test_tensor_artifact_output_validates_its_rank() -> None:
    with pytest.raises(ValueError, match="declares 2 dimensions"):
        TensorArtifactOutput(schema=_ACTIONS, tensor=torch.zeros(1, 2, 3))


def test_tensor_artifact_names_cannot_escape_the_output_directory() -> None:
    with pytest.raises(ValueError, match="Tensor artifact names"):
        TensorArtifactSchema(
            name="../actions",
            dimension_names=("step", "channel"),
        )
