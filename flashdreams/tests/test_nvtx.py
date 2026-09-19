# SPDX-FileCopyrightText: Copyright (c) 2026 Praneeth Samineni.
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

"""NVTX range emission and the ``flashdreams-profile`` launcher."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import torch

from flashdreams.infra import nvtx
from tools.profiling import cli as profiling_cli

pytestmark = pytest.mark.ci_cpu


class _RecordingNvtx:
    """Stand-in for ``torch.cuda.nvtx`` that records push/pop order."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def range_push(self, name: str) -> None:
        self.events.append(f"push:{name}")

    def range_pop(self) -> None:
        self.events.append("pop")


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _RecordingNvtx:
    recording = _RecordingNvtx()
    monkeypatch.setattr(nvtx, "_ENABLED", True)
    monkeypatch.setattr(torch.cuda, "nvtx", recording)
    return recording


def test_nested_ranges_close_innermost_first(recorder: _RecordingNvtx) -> None:
    with nvtx.nvtx_range("model.step[0]"), nvtx.nvtx_range("denoise[0]"):
        pass

    assert recorder.events == ["push:model.step[0]", "push:denoise[0]", "pop", "pop"]


def test_range_pops_when_the_body_raises(recorder: _RecordingNvtx) -> None:
    """An unclosed range silently shifts every later row of the trace."""
    with pytest.raises(ValueError), nvtx.nvtx_range("model.step[0]"):
        raise ValueError

    assert recorder.events == ["push:model.step[0]", "pop"]


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a CPU-only build")
def test_ranges_are_safe_on_cpu_only_builds() -> None:
    """``range_push`` raises without CUDA, so the gate must never reach it."""
    with nvtx.nvtx_range("model.step[0]"):
        pass


def test_pipeline_emits_stage_ranges(recorder: _RecordingNvtx) -> None:
    null_model = pytest.importorskip("null_model")
    pipeline = null_model.NULL_MODEL_CONFIG.setup().to("cpu")
    cache = pipeline.initialize_cache()

    pipeline.generate(0, cache, input=torch.tensor([[1]]))
    pipeline.finalize(0, cache)

    assert recorder.events == [
        "push:pipeline.encode",
        "pop",
        "push:pipeline.diffuse",
        "push:denoise[0]",
        "pop",
        "pop",
        "push:pipeline.decode",
        "pop",
        "push:pipeline.finalize",
        "pop",
    ]


def test_profile_wraps_the_runner_console_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The v2 CLI module has no ``__main__`` guard, so ``python -m`` exits silently."""
    script = tmp_path / "flashdreams-run-v2"
    script.touch()
    script.chmod(0o755)
    monkeypatch.setattr(profiling_cli.sys, "executable", str(tmp_path / "python"))
    assert profiling_cli.resolve_runner() == str(script)

    command = profiling_cli.build_command(
        runner_args=["slug"],
        runner=str(script),
        report_path=tmp_path / "run.nsys-rep",
        trace=profiling_cli._DEFAULT_TRACE,
    )

    assert command == [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,osrt,vulkan",
        "--output",
        str(tmp_path / "run"),
        "--force-overwrite",
        "true",
        str(script),
        "slug",
    ]


@pytest.mark.parametrize(
    "make", [Path.touch, Path.mkdir], ids=["non-executable file", "directory"]
)
def test_profile_skips_a_sibling_runner_it_cannot_execute(
    make, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stray path next to the interpreter must not shadow the runner on PATH."""
    make(tmp_path / "flashdreams-run-v2")
    monkeypatch.setattr(profiling_cli.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(profiling_cli.shutil, "which", lambda name: f"/usr/bin/{name}")

    assert profiling_cli.resolve_runner() == "/usr/bin/flashdreams-run-v2"


def test_profile_reports_default_under_artifacts() -> None:
    report_path = profiling_cli.default_report_path()

    assert report_path.parent == Path("artifacts", "profiles")
    assert report_path.suffix == ".nsys-rep"


def test_default_reports_from_concurrent_runs_do_not_collide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent runs in the same second must not overwrite each other's report."""
    monkeypatch.setattr(profiling_cli.time, "strftime", lambda fmt: "20260918-120000")
    monkeypatch.setattr(profiling_cli.os, "getpid", lambda: 101)
    first = profiling_cli.default_report_path()
    monkeypatch.setattr(profiling_cli.os, "getpid", lambda: 102)

    assert profiling_cli.default_report_path() != first


def test_profile_writes_the_report_where_asked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(profiling_cli.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(profiling_cli.subprocess, "run", run)

    report_path = tmp_path / "profiles" / "run"
    assert profiling_cli.main(["--report-path", str(report_path), "--", "slug"]) == 0
    assert commands[0][commands[0].index("--output") + 1] == str(report_path)
    assert report_path.parent.is_dir()


def test_profile_stops_before_writing_when_nsys_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report the missing profiler without a traceback or an empty report directory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(profiling_cli.shutil, "which", lambda name: None)

    assert profiling_cli.main(["--", "slug"]) == 127
    assert "nsys not found" in capsys.readouterr().err
    assert not (tmp_path / "artifacts").exists()


def test_profile_stops_before_writing_when_the_runner_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Report the missing runner without a traceback or an empty report directory."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(profiling_cli.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(
        profiling_cli.shutil,
        "which",
        lambda name: "/bin/nsys" if name == "nsys" else None,
    )

    assert profiling_cli.main(["--", "slug"]) == 127
    assert "flashdreams-run-v2 not found" in capsys.readouterr().err
    assert not (tmp_path / "artifacts").exists()
