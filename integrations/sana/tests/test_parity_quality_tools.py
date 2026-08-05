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

"""CPU-safe tests for SANA-WM parity quality tools."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

pytestmark = pytest.mark.ci_cpu

TOOLS_DIR = Path("integrations/sana/tests/parity_check")


def _load_tool(module_name: str) -> ModuleType:
    path = TOOLS_DIR / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(TOOLS_DIR))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(TOOLS_DIR))
    return module


def test_diff_parity_loads_upstream_streaming_npz(tmp_path: Path) -> None:
    module = _load_tool("diff_parity")
    frames = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)
    path = tmp_path / "frames.npz"
    np.savez_compressed(path, frames=frames, frame_indices=np.arange(2))

    loaded = module._load_frames(path)

    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded, frames)


def test_streaming_continuity_reports_boundary_spikes(tmp_path: Path) -> None:
    module = _load_tool("streaming_continuity")
    values = np.asarray([0, 1, 2, 20, 21, 22, 40], dtype=np.uint8)
    frames = np.repeat(values[:, None, None, None], 3, axis=3)

    summary = module.summarize_frames(frames, chunk_size=3)

    assert summary["boundary_mean_abs_delta"] == 18.0
    assert summary["nonboundary_p95_abs_delta"] == 1.0
    assert summary["boundary_to_nonboundary_p95_ratio"] == 18.0
    assert summary["boundary_transitions"] == [
        {"from_frame": 2, "to_frame": 3, "mean_abs_delta": 18.0},
        {"from_frame": 5, "to_frame": 6, "mean_abs_delta": 18.0},
    ]


def test_streaming_continuity_cli_writes_comparison_json(tmp_path: Path) -> None:
    module = _load_tool("streaming_continuity")
    upstream_frames = np.zeros((4, 1, 1, 3), dtype=np.uint8)
    flashdreams_frames = upstream_frames.copy()
    flashdreams_frames[2:] = 10
    upstream = tmp_path / "upstream.npz"
    flashdreams = tmp_path / "flashdreams.npy"
    output = tmp_path / "continuity.json"
    np.savez_compressed(upstream, frames=upstream_frames)
    np.save(flashdreams, flashdreams_frames)

    module.main(
        [
            "--upstream",
            str(upstream),
            "--flashdreams",
            str(flashdreams),
            "--chunk-size",
            "2",
            "--output",
            str(output),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["comparison"]["shape_match"] is True
    assert payload["upstream"]["boundary_mean_abs_delta"] == 0.0
    assert payload["flashdreams"]["boundary_mean_abs_delta"] == 10.0
