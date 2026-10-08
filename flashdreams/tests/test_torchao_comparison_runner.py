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

"""CPU checks for benchmark evidence preservation and result accounting."""

import json
import runpy
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.ci_cpu

_RUNNER = (
    Path(__file__).parents[1]
    / "benchmarks/accelerated/quantization/run_torchao_comparison.py"
)


@pytest.mark.parametrize(
    "result_kind",
    ("missing", "skipped", "truncated", "incomplete", "wrong_backend", "valid"),
)
def test_runner_requires_measurements(result_kind: str) -> None:
    """Fail zero-exit subprocesses without complete measurements and keep their logs."""
    main = runpy.run_path(str(_RUNNER))["main"]

    def run(command, **kwargs):
        if "--benchmark-json" in command:
            path = Path(command[command.index("--benchmark-json") + 1])
            _, backend, mode, _ = path.stem.split("-")
            info: dict[str, object] = dict.fromkeys(
                (
                    "preparation_ms",
                    "first_call_ms",
                    "capture_first_call_ms",
                    "cuda_event_median_ms",
                    "steady_peak_allocated_bytes",
                    "relative_l2",
                ),
                0.0,
            )
            info.update(backend=backend, execution=mode)
            record = {
                "extra_info": info,
                "stats": {"data": [0.001] * 50, "median": 0.001},
            }
            if result_kind == "wrong_backend":
                info["backend"] = "unrelated"
            if result_kind == "incomplete":
                record["stats"]["data"] = []
            if result_kind == "truncated":
                path.write_text('{"benchmarks":')
            elif result_kind != "missing":
                path.write_text(
                    json.dumps(
                        {"benchmarks": [] if result_kind == "skipped" else [record]}
                    )
                )
        return subprocess.CompletedProcess(command, 0)

    def check_output(command, **kwargs):
        return "test environment" if kwargs.get("text") else b""

    with TemporaryDirectory() as directory:
        output = Path(directory) / "results"
        with (
            patch.object(
                sys, "argv", [str(_RUNNER), "--output", str(output), "--repeats", "1"]
            ),
            patch("subprocess.run", side_effect=run),
            patch("subprocess.check_output", side_effect=check_output),
        ):
            if result_kind == "valid":
                main()
            else:
                with pytest.raises(SystemExit) as error:
                    main()
                assert error.value.code == 1
        manifest = json.loads((output / "manifest.json").read_text())
        assert len(manifest["runs"]) == 18
        assert all(row["returncode"] == 0 for row in manifest["runs"])
        assert all(
            ("artifact_error" in row) == (result_kind != "valid")
            for row in manifest["runs"]
        )
        assert len(list(output.glob("*.log"))) == 18
        if result_kind == "valid":
            assert len(json.loads((output / "measurements.json").read_text())) == 18


def test_runner_preserves_existing_directory() -> None:
    """Reject cached or interrupted output directories before writing anything."""
    main = runpy.run_path(str(_RUNNER))["main"]
    with TemporaryDirectory() as directory:
        cache = Path(directory) / "cache"
        cache.mkdir()
        existing = cache / "evidence"
        existing.write_text("keep")
        with (
            patch.object(sys, "argv", [str(_RUNNER), "--output", directory]),
            patch("subprocess.run") as run,
            patch("subprocess.check_output") as check_output,
        ):
            with pytest.raises(SystemExit) as error:
                main()
            assert error.value.code == 2
            run.assert_not_called()
            check_output.assert_not_called()
        assert existing.read_text() == "keep"
        assert list(Path(directory).iterdir()) == [cache]
