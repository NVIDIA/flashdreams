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

"""The profiler interface, its backends, and the ``flashdreams-profile`` launcher."""

from __future__ import annotations

import queue
import subprocess
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest
import torch

from flashdreams.api_v2.loop import IModelLoop
from flashdreams.infra import profiler as profiler_module
from flashdreams.infra.profiler import (
    CompositeProfiler,
    CudaEventProfiler,
    IProfiler,
    NullProfiler,
    NVTXProfiler,
    _reset_default_profiler,
    bind_inference_profiler,
    create_profiler,
    get_inference_profiler,
    set_flashdreams_inference_profiler,
    unbind_inference_profiler,
)
from tools.profiling import cli as profiling_cli

pytestmark = pytest.mark.ci_cpu


class _RecordingProfiler(IProfiler):
    """Records range names and nesting instead of profiling."""

    def __init__(self) -> None:
        self.events: list[str] = []

    @contextmanager
    def range(self, name: str) -> Iterator[None]:
        self.events.append(f"enter:{name}")
        try:
            yield
        finally:
            self.events.append(f"exit:{name}")


class _RecordingNvtx:
    """Stand-in for ``torch.cuda.nvtx`` that records push/pop order."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def range_push(self, name: str) -> None:
        self.events.append(f"push:{name}")

    def range_pop(self) -> None:
        self.events.append("pop")


@pytest.fixture
def nvtx_recorder(monkeypatch: pytest.MonkeyPatch) -> _RecordingNvtx:
    recording = _RecordingNvtx()
    monkeypatch.setattr(torch.cuda, "nvtx", recording)
    return recording


def test_nvtx_nested_ranges_close_innermost_first(
    nvtx_recorder: _RecordingNvtx,
) -> None:
    profiler = NVTXProfiler()

    with profiler.range("model.step[0]"), profiler.range("pipeline.diffuse"):
        pass

    assert nvtx_recorder.events == [
        "push:model.step[0]",
        "push:pipeline.diffuse",
        "pop",
        "pop",
    ]


def test_nvtx_range_pops_when_the_body_raises(nvtx_recorder: _RecordingNvtx) -> None:
    """An unclosed range silently shifts every later row of the trace."""
    with pytest.raises(ValueError), NVTXProfiler().range("model.step[0]"):
        raise ValueError

    assert nvtx_recorder.events == ["push:model.step[0]", "pop"]


def test_null_profiler_records_nothing() -> None:
    profiler = NullProfiler()

    with profiler.range("model.step[0]"):
        pass

    assert profiler.collect_stage_ms() == {}


def test_cuda_event_profiler_ignores_ranges_outside_the_pipeline() -> None:
    """Stage timings are consecutive pipeline stages, so other ranges are skipped."""
    profiler = CudaEventProfiler()

    with profiler.range("model.step[0]"):
        pass

    assert profiler.collect_stage_ms() == {}


class _StubEvents:
    """Stand-in for ``EventProfiler``: no CUDA, same record/summarize contract."""

    def __init__(self) -> None:
        self.stages: list[str] = []

    def record(self, stage: str) -> None:
        assert stage not in self.stages, f"stage {stage!r} already recorded"
        self.stages.append(stage)

    def sync_and_summarize(self) -> dict[str, float]:
        return {stage: 1.0 for stage in self.stages}


@pytest.fixture
def stub_events(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(profiler_module, "EventProfiler", _StubEvents)


def test_cuda_event_profiler_recovers_from_a_step_that_never_collected(
    stub_events: None,
) -> None:
    """A step raising between generate and finalize must not poison the next one."""
    profiler = CudaEventProfiler()

    with pytest.raises(RuntimeError), profiler.range("pipeline.encode"):
        raise RuntimeError("decode blew up before finalize collected")

    # The next step reuses the same stage names; this must not assert.
    with profiler.range("pipeline.encode"):
        pass
    with profiler.range("pipeline.diffuse"):
        pass

    assert sorted(profiler.collect_stage_ms()) == ["diffuse", "encode"]


def test_cuda_event_profiler_keeps_threads_apart(stub_events: None) -> None:
    """One profiler serves every session, so its stage state must be per-thread."""
    profiler = CudaEventProfiler()
    other: list[dict[str, float]] = []

    def worker() -> None:
        with profiler.range("pipeline.decode"):
            pass
        other.append(profiler.collect_stage_ms())

    with profiler.range("pipeline.encode"):
        pass
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert other == [{"decode": 1.0}], "the thread sees only its own stage"
    assert sorted(profiler.collect_stage_ms()) == ["encode"]


def test_composite_drains_every_backend(stub_events: None) -> None:
    """A backend left uncollected would carry its stages into the next step."""
    first, second = CudaEventProfiler(), CudaEventProfiler()
    profiler = CompositeProfiler((first, second))

    with profiler.range("pipeline.encode"):
        pass

    assert profiler.collect_stage_ms() == {"encode": 1.0}
    assert first.collect_stage_ms() == {} and second.collect_stage_ms() == {}


def test_code_without_a_session_still_profiles_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The batch runner and serving run pipelines with no session; they kept metrics."""
    monkeypatch.setenv("FLASHDREAMS_SYNC_AND_PROFILE", "1")
    monkeypatch.delenv("FLASHDREAMS_NVTX", raising=False)
    _reset_default_profiler()
    try:
        assert isinstance(get_inference_profiler(), CudaEventProfiler)
    finally:
        _reset_default_profiler()


def test_composite_enters_every_backend(nvtx_recorder: _RecordingNvtx) -> None:
    recording = _RecordingProfiler()
    profiler = CompositeProfiler((NVTXProfiler(), recording))

    with profiler.range("pipeline.decode"):
        pass

    assert nvtx_recorder.events == ["push:pipeline.decode", "pop"]
    assert recording.events == ["enter:pipeline.decode", "exit:pipeline.decode"]


@pytest.mark.parametrize(
    ("nvtx", "sync", "cuda", "expected"),
    [
        (None, None, True, NullProfiler),
        ("1", None, True, NVTXProfiler),
        ("1", None, False, NullProfiler),
        (None, "1", True, CudaEventProfiler),
        ("1", "1", True, CompositeProfiler),
    ],
)
def test_create_profiler_reads_the_environment(
    nvtx: str | None,
    sync: str | None,
    cuda: bool,
    expected: type[IProfiler],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``range_push`` raises without CUDA, so NVTX needs a CUDA build."""
    monkeypatch.delenv("FLASHDREAMS_NVTX", raising=False)
    monkeypatch.delenv("FLASHDREAMS_SYNC_AND_PROFILE", raising=False)
    if nvtx is not None:
        monkeypatch.setenv("FLASHDREAMS_NVTX", nvtx)
    if sync is not None:
        monkeypatch.setenv("FLASHDREAMS_SYNC_AND_PROFILE", sync)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)

    assert isinstance(create_profiler(), expected)


def test_the_default_profiler_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing asked for, inference outside an application records nothing."""
    monkeypatch.delenv("FLASHDREAMS_NVTX", raising=False)
    monkeypatch.delenv("FLASHDREAMS_SYNC_AND_PROFILE", raising=False)
    _reset_default_profiler()
    try:
        assert isinstance(get_inference_profiler(), NullProfiler)
    finally:
        _reset_default_profiler()


def test_a_thread_binds_the_profiler_it_was_given() -> None:
    """ContextVars do not cross threads, so the model thread binds its own."""
    recording = _RecordingProfiler()
    seen: list[IProfiler] = []

    def worker() -> None:
        bind_inference_profiler(recording)
        seen.append(get_inference_profiler())

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()

    assert seen == [recording]
    assert isinstance(get_inference_profiler(), NullProfiler), "caller is unaffected"


def test_pipeline_ranges_come_from_the_profiler_in_context() -> None:
    null_model = pytest.importorskip("null_model")
    pipeline = null_model.NULL_MODEL_CONFIG.setup().to("cpu")
    recording = _RecordingProfiler()
    cache = pipeline.initialize_cache()

    with set_flashdreams_inference_profiler(recording):
        pipeline.generate(0, cache, input=torch.tensor([[1]]))
        # No stage timings from this profiler, so finalize reports none.
        assert pipeline.finalize(0, cache) is None

    assert recording.events == [
        "enter:pipeline.encode",
        "exit:pipeline.encode",
        "enter:pipeline.diffuse",
        "exit:pipeline.diffuse",
        "enter:pipeline.decode",
        "exit:pipeline.decode",
        "enter:pipeline.finalize",
        "exit:pipeline.finalize",
    ]


def test_pipeline_reports_timings_a_caller_recorded_on_the_cache(
    stub_events: None,
) -> None:
    """FlashVSR overrides generate() and times its own stages onto the cache.

    finalize() is the shared one, so it must report those rather than the
    injected profiler's, which saw no pipeline stages at all.
    """
    null_model = pytest.importorskip("null_model")
    pipeline = null_model.NULL_MODEL_CONFIG.setup().to("cpu")
    recording = _RecordingProfiler()
    cache = pipeline.initialize_cache()

    with set_flashdreams_inference_profiler(recording):
        pipeline.generate(0, cache, input=torch.tensor([[1]]))
        cache.event_profiler = profiler_module.EventProfiler()
        cache.event_profiler.record("denoise")
        stats = pipeline.finalize(0, cache)

    assert stats is not None, "a caller's own timings must not be discarded"
    assert "denoise_ms" in stats and "finalize_ms" in stats


def test_loops_run_with_the_profiler_the_session_injects() -> None:
    class _Loop(IModelLoop[None]):
        def step(self, step_index: int, events: object) -> list[object]:
            return []

    loop = _Loop()
    recording = _RecordingProfiler()
    loop.register_session_loop_objects(
        state=None,
        frequency=0,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
        profiler=recording,
    )

    assert loop.profiler is recording


def test_a_finished_loop_leaves_the_caller_s_context_alone() -> None:
    """A loop driven on the caller's thread must not keep its binding afterwards."""
    recording = _RecordingProfiler()
    token = bind_inference_profiler(recording)
    assert get_inference_profiler() is recording
    unbind_inference_profiler(token)

    assert get_inference_profiler() is not recording


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
