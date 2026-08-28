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

"""CPU tests for LingBot-VA action artifact inspection."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from lingbot_va.action_artifact import (
    ROBOTWIN_ACTION_CHANNEL_NAMES,
    load_action_artifact,
    write_action_csv,
    write_action_plot,
)

matplotlib.use("Agg")
pytestmark = pytest.mark.ci_cpu


def _write_artifact(output_dir: Path, *, complete: bool = True) -> np.ndarray:
    """Write a representative committed or incomplete action artifact."""
    actions = np.arange(64, dtype=np.float32).reshape(4, 16)
    output_dir.mkdir()
    np.save(output_dir / "actions.npy", actions)
    manifest = {
        "schema_version": 1,
        "artifact_type": "flashdreams.runtime_v2.tensor_artifacts",
        "complete": complete,
        "generation": 0,
        "artifacts": [
            {
                "name": "actions",
                "path": "actions.npy",
                "emitted": True,
                "dimension_names": ["step", "channel"],
                "concatenate_axis": 0,
                "dtype": "float32",
                "shape": [4, 16],
            }
        ],
    }
    (output_dir / "tensor_artifacts.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return actions


def test_loads_complete_manifest_artifact(tmp_path: Path) -> None:
    expected = _write_artifact(tmp_path / "output")

    actual = load_action_artifact(tmp_path / "output")

    np.testing.assert_array_equal(actual, expected)


def test_rejects_incomplete_manifest_artifact(tmp_path: Path) -> None:
    _write_artifact(tmp_path / "output", complete=False)

    with pytest.raises(ValueError, match="incomplete"):
        load_action_artifact(tmp_path / "output")


def test_accepts_direct_legacy_numpy_output(tmp_path: Path) -> None:
    actions = np.zeros((32, 16), dtype=np.float32)
    output_path = tmp_path / "actions.npy"
    np.save(output_path, actions)

    np.testing.assert_array_equal(load_action_artifact(output_path), actions)


def test_writes_named_csv_and_plot(tmp_path: Path) -> None:
    actions = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(4, 16)

    csv_path = write_action_csv(actions, tmp_path / "actions.csv")
    plot_path = write_action_plot(actions, tmp_path / "actions.png")

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.reader(csv_file))
    assert rows[0] == ["step", *ROBOTWIN_ACTION_CHANNEL_NAMES]
    assert len(rows) == 5
    assert plot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
