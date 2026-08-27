# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for typed tensor artifacts and their NumPy output sink."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

import flashdreams.runtime_v2.tensor_artifact_output_sink as artifact_sink_module
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

_TRAJECTORY = TensorArtifactSchema(
    name="trajectory",
    dimension_names=("sample", "coordinate"),
    concatenate_axis=0,
)
"""Generic sequence schema used to exercise tensor routing."""


def _manifest(output_dir: Path) -> dict[str, Any]:
    return json.loads((output_dir / "tensor_artifacts.json").read_text())


def _session_desc(
    *schemas: TensorArtifactSchema,
) -> SessionDesc:
    return SessionDesc(tensor_artifact_schemas=schemas or (_TRAJECTORY,))


def _artifact(
    tensor: torch.Tensor,
    schema: TensorArtifactSchema = _TRAJECTORY,
) -> TensorArtifactOutput:
    return TensorArtifactOutput(schema=schema, tensor=tensor)


def _result(
    step_index: int,
    *artifacts: TensorArtifactOutput,
) -> StepResult:
    return StepResult(
        step_index=step_index,
        output=torch.zeros(1, 3, 1, 2, 2),
        frame_count=1,
        output_layout=VideoTensorLayout.bcthw,
        tensor_artifacts=tuple(artifacts),
    )


def test_sink_concatenates_steps_and_close_is_idempotent(tmp_path: Path) -> None:
    sink = TensorArtifactOutputSink(tmp_path)
    sink.open(_session_desc())
    sink.write(0, [_result(0, _artifact(torch.tensor([[1.0, 2.0]])))])
    sink.write(
        0,
        [_result(1, _artifact(torch.tensor([[3.0, 4.0], [5.0, 6.0]])))],
    )

    sink.close()
    first_bytes = (tmp_path / "trajectory.npy").read_bytes()
    sink.close()

    np.testing.assert_array_equal(
        np.load(tmp_path / "trajectory.npy"),
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
    )
    assert (tmp_path / "trajectory.npy").read_bytes() == first_bytes
    assert not list(tmp_path.glob("*.tmp"))
    assert _manifest(tmp_path) == {
        "artifact_type": "flashdreams.runtime_v2.tensor_artifacts",
        "artifacts": [
            {
                "concatenate_axis": 0,
                "dimension_names": ["sample", "coordinate"],
                "dtype": "float32",
                "emitted": True,
                "name": "trajectory",
                "path": "trajectory.npy",
                "shape": [3, 2],
            }
        ],
        "complete": True,
        "generation": 0,
        "schema_version": 1,
    }


def test_each_declared_artifact_is_optional(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts"
    sink = TensorArtifactOutputSink(output_dir)
    sink.open(_session_desc())
    sink.write(0, [_result(0)])

    sink.close()

    assert not (output_dir / "trajectory.npy").exists()
    manifest = _manifest(output_dir)
    assert manifest["complete"] is True
    assert manifest["artifacts"][0]["emitted"] is False
    assert manifest["artifacts"][0]["dimension_names"] == ["sample", "coordinate"]


def test_successful_run_removes_a_stale_optional_artifact(tmp_path: Path) -> None:
    destination = tmp_path / "trajectory.npy"
    destination.write_bytes(b"stale")
    sink = TensorArtifactOutputSink(tmp_path)

    sink.open(_session_desc())
    assert destination.exists()
    sink.close()

    assert not destination.exists()
    assert _manifest(tmp_path)["complete"] is True


def test_failed_run_discards_buffered_artifacts(tmp_path: Path) -> None:
    sink = TensorArtifactOutputSink(tmp_path)
    sink.open(_session_desc())
    sink.write(0, [_result(0, _artifact(torch.ones(1, 2)))])

    sink.close(commit=False)

    assert not (tmp_path / "trajectory.npy").exists()
    manifest = _manifest(tmp_path)
    assert manifest["complete"] is False
    assert manifest["artifacts"][0]["emitted"] is False


def test_open_rejects_an_invalid_output_path_before_writes(tmp_path: Path) -> None:
    output_path = tmp_path / "not-a-directory"
    output_path.write_text("occupied")
    sink = TensorArtifactOutputSink(output_path)

    with pytest.raises(FileExistsError):
        sink.open(_session_desc())


def test_sink_rejects_undeclared_outputs(tmp_path: Path) -> None:
    other = TensorArtifactSchema(
        name="other",
        dimension_names=("sample", "coordinate"),
    )
    sink = TensorArtifactOutputSink(tmp_path)
    sink.open(_session_desc())

    with pytest.raises(ValueError, match="'other'.*not declared"):
        sink.write(0, [_result(0, _artifact(torch.zeros(1, 2), other))])


def test_sink_rejects_changed_schema(tmp_path: Path) -> None:
    changed = TensorArtifactSchema(
        name="trajectory",
        dimension_names=("sample", "feature"),
    )
    sink = TensorArtifactOutputSink(tmp_path)
    sink.open(_session_desc())

    with pytest.raises(ValueError, match="'trajectory'.*does not match"):
        sink.write(0, [_result(0, _artifact(torch.zeros(1, 2), changed))])


def test_sink_rejects_duplicates_within_one_result(tmp_path: Path) -> None:
    sink = TensorArtifactOutputSink(tmp_path)
    sink.open(_session_desc())
    artifact = _artifact(torch.zeros(1, 2))

    with pytest.raises(ValueError, match="'trajectory'.*more than once"):
        sink.write(0, [_result(0, artifact, artifact)])

    sink.close(commit=False)
    assert not (tmp_path / "trajectory.npy").exists()
    assert _manifest(tmp_path)["complete"] is False


def test_sink_rejects_duplicates_across_result_channels(tmp_path: Path) -> None:
    sink = TensorArtifactOutputSink(tmp_path)
    sink.open(_session_desc())

    with pytest.raises(ValueError, match="'trajectory'.*output channels"):
        sink.write(
            0,
            [
                _result(0, _artifact(torch.zeros(1, 2))),
                _result(0, _artifact(torch.ones(1, 2))),
            ],
        )

    sink.close(commit=False)
    assert not (tmp_path / "trajectory.npy").exists()
    assert _manifest(tmp_path)["complete"] is False


def test_sink_rejects_dtype_changes(tmp_path: Path) -> None:
    sink = TensorArtifactOutputSink(tmp_path)
    sink.open(_session_desc())
    sink.write(0, [_result(0, _artifact(torch.zeros(1, 2)))])

    with pytest.raises(ValueError, match="'trajectory'.*changed dtype"):
        sink.write(
            0,
            [_result(1, _artifact(torch.zeros(1, 2, dtype=torch.int64)))],
        )


def test_sink_rejects_dtypes_numpy_cannot_store(tmp_path: Path) -> None:
    sink = TensorArtifactOutputSink(tmp_path)
    sink.open(_session_desc())

    with pytest.raises(ValueError, match="'trajectory'.*cannot be stored"):
        sink.write(
            0,
            [_result(0, _artifact(torch.zeros(1, 2, dtype=torch.bfloat16)))],
        )


def test_sink_rejects_non_concatenated_dimension_changes(tmp_path: Path) -> None:
    sink = TensorArtifactOutputSink(tmp_path)
    sink.open(_session_desc())
    sink.write(0, [_result(0, _artifact(torch.zeros(1, 2)))])

    with pytest.raises(ValueError, match="'trajectory'.*non-concatenated"):
        sink.write(0, [_result(1, _artifact(torch.zeros(3, 4)))])


def test_sink_permits_concatenated_dimension_changes(tmp_path: Path) -> None:
    sink = TensorArtifactOutputSink(tmp_path)
    sink.open(_session_desc())
    sink.write(0, [_result(0, _artifact(torch.zeros(1, 2)))])
    sink.write(0, [_result(1, _artifact(torch.ones(3, 2)))])

    sink.close()

    assert np.load(tmp_path / "trajectory.npy").shape == (4, 2)


def test_non_concatenated_artifact_permits_only_one_output(
    tmp_path: Path,
) -> None:
    summary = TensorArtifactSchema(
        name="summary",
        dimension_names=("coordinate",),
        concatenate_axis=None,
    )
    sink = TensorArtifactOutputSink(tmp_path)
    sink.open(_session_desc(summary))
    sink.write(0, [_result(0, _artifact(torch.zeros(2), summary))])

    with pytest.raises(ValueError, match="'summary'.*at most one"):
        sink.write(0, [_result(1, _artifact(torch.ones(2), summary))])


def test_new_generation_replaces_old_and_late_old_results_are_ignored(
    tmp_path: Path,
) -> None:
    sink = TensorArtifactOutputSink(tmp_path)
    sink.open(_session_desc())
    sink.write(0, [_result(0, _artifact(torch.tensor([[0.0, 0.0]])))])
    sink.write(1, [_result(0, _artifact(torch.tensor([[1.0, 1.0]])))])
    sink.write(0, [_result(1, _artifact(torch.tensor([[9.0, 9.0]])))])
    sink.write(1, [_result(1, _artifact(torch.tensor([[2.0, 2.0]])))])
    sink.close()

    np.testing.assert_array_equal(
        np.load(tmp_path / "trajectory.npy"),
        np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32),
    )


def test_sink_requires_open_and_open_requires_a_declared_schema(
    tmp_path: Path,
) -> None:
    sink = TensorArtifactOutputSink(tmp_path)
    sink.close()

    with pytest.raises(RuntimeError, match=r"open\(\).*before write"):
        sink.write(0, [_result(0)])

    with pytest.raises(ValueError, match="at least one tensor artifact"):
        sink.open(SessionDesc())

    assert list(tmp_path.iterdir()) == []


def test_staging_failure_preserves_an_existing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "trajectory.npy"
    destination.write_bytes(b"previous")
    sink = TensorArtifactOutputSink(tmp_path)
    sink.open(_session_desc())
    sink.write(0, [_result(0, _artifact(torch.zeros(1, 2)))])

    def fail_write(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("write failed")

    monkeypatch.setattr(artifact_sink_module, "_write_numpy", fail_write)

    with pytest.raises(OSError, match="write failed"):
        sink.close()

    assert destination.read_bytes() == b"previous"
    assert not list(tmp_path.glob("*.tmp"))


def test_replace_failure_removes_staged_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "trajectory.npy"
    destination.write_bytes(b"previous")
    sink = TensorArtifactOutputSink(tmp_path)
    sink.open(_session_desc())
    sink.write(0, [_result(0, _artifact(torch.zeros(1, 2)))])

    def fail_replace(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("replace failed")

    monkeypatch.setattr(artifact_sink_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        sink.close()

    assert destination.read_bytes() == b"previous"
    assert not list(tmp_path.glob("*.tmp"))


def test_manifest_replace_failure_restores_prior_artifact_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optional = TensorArtifactSchema(
        name="optional",
        dimension_names=("sample", "coordinate"),
    )
    trajectory_path = tmp_path / "trajectory.npy"
    optional_path = tmp_path / "optional.npy"
    np.save(trajectory_path, np.array([[1.0, 2.0]], dtype=np.float32))
    np.save(optional_path, np.array([[3.0, 4.0]], dtype=np.float32))
    prior_trajectory = trajectory_path.read_bytes()
    prior_optional = optional_path.read_bytes()

    sink = TensorArtifactOutputSink(tmp_path)
    sink.open(_session_desc(_TRAJECTORY, optional))
    sink.write(0, [_result(0, _artifact(torch.tensor([[9.0, 9.0]])))])

    real_replace = artifact_sink_module.os.replace

    def fail_final_manifest_replace(source: Any, destination: Any) -> None:
        if Path(destination).name == "tensor_artifacts.json":
            raise OSError("manifest replace failed")
        real_replace(source, destination)

    monkeypatch.setattr(
        artifact_sink_module.os,
        "replace",
        fail_final_manifest_replace,
    )

    with pytest.raises(OSError, match="manifest replace failed"):
        sink.close()

    assert trajectory_path.read_bytes() == prior_trajectory
    assert optional_path.read_bytes() == prior_optional
    assert _manifest(tmp_path)["complete"] is False
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "name",
    ["../trajectory", "/absolute", ".hidden", "contains space"],
)
def test_artifact_names_cannot_escape_the_output_directory(name: str) -> None:
    with pytest.raises(ValueError, match="Tensor artifact names"):
        TensorArtifactSchema(
            name=name,
            dimension_names=("sample", "coordinate"),
        )


def test_artifact_schema_validates_dimension_names() -> None:
    with pytest.raises(ValueError, match="'trajectory'.*non-empty"):
        TensorArtifactSchema(name="trajectory", dimension_names=("sample", " "))
    with pytest.raises(ValueError, match="'trajectory'.*unique"):
        TensorArtifactSchema(
            name="trajectory",
            dimension_names=("sample", "sample"),
        )


@pytest.mark.parametrize("axis", [2, -3])
def test_artifact_schema_validates_concatenation_axis(axis: int) -> None:
    with pytest.raises(ValueError, match="'trajectory'.*concatenate_axis"):
        TensorArtifactSchema(
            name="trajectory",
            dimension_names=("sample", "coordinate"),
            concatenate_axis=axis,
        )


def test_scalar_artifact_requires_no_concatenation_axis() -> None:
    with pytest.raises(ValueError, match="'summary'.*concatenate_axis"):
        TensorArtifactSchema(name="summary", dimension_names=())

    assert (
        TensorArtifactSchema(
            name="summary",
            dimension_names=(),
            concatenate_axis=None,
        ).concatenate_axis
        is None
    )


def test_artifact_output_validates_its_rank() -> None:
    with pytest.raises(ValueError, match="'trajectory'.*declares 2 dimensions"):
        TensorArtifactOutput(schema=_TRAJECTORY, tensor=torch.zeros(1, 2, 3))


def test_session_desc_rejects_duplicate_artifact_names() -> None:
    duplicate = TensorArtifactSchema(
        name="trajectory",
        dimension_names=("sample",),
    )

    with pytest.raises(ValueError, match="artifact names must be unique"):
        SessionDesc(tensor_artifact_schemas=(_TRAJECTORY, duplicate))
