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

"""Bounded many-process runtime checks for step admission, failures, and ownership."""

from __future__ import annotations

import json
import os
import threading
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from flashdreams.api_v2.client_window import IClientWindow
from flashdreams.api_v2.loop import IModelLoop
from flashdreams.api_v2.session import ISession
from flashdreams.core.distributed.parallel import build_context
from flashdreams.runtime_v2 import session_runner
from flashdreams.runtime_v2.coordination import StepAgreement
from flashdreams.runtime_v2.session_desc import PresentationMode, SessionDesc
from flashdreams.runtime_v2.step_result import StepResult
from flashdreams.runtime_v2.user_input_event import (
    CloseUserInputEvent,
    KeyboardInputState,
    KeyboardUserInputEvent,
    ResetUserInputEvent,
)
from flashdreams.runtime_v2.user_input_events import UserInputEvents

pytestmark = pytest.mark.ci_cpu


class _Model(IModelLoop):
    def _model_thread(self):
        assert threading.current_thread().name == "flashdreams-model-generation-thread"

    def is_finished(self):
        self._model_thread()
        if self.state.scenario == "prepare_failure" and self.state.ctx.rank == 1:
            raise ValueError("injected preparation failure")
        if self.state.scenario == "worker_finished" and self.state.ctx.rank == 1:
            return len(self.state.indices) == 1
        return self._step_index >= 3

    def _pace(self, last_run_started):
        result = super()._pace(last_run_started)
        if self.state.scenario == "stop_during_pacing" and self.state.ctx.is_main:
            self._shutdown_event.set()
        if (
            self.state.scenario == "reset"
            and self.state.ctx.is_main
            and len(self.state.indices) == 1
        ):
            assert self.state.reset_sent.wait(5), "UI did not deliver reset"
        return result

    def step(self, step_index, events):
        self._model_thread()
        if self.state.scenario == "step_failure" and self.state.ctx.rank == 1:
            raise ValueError("injected model failure")
        value = torch.tensor([self.state.ctx.rank + 1.0])
        dist.all_reduce(value)
        assert value.item() == 3.0
        self.state.indices.append(step_index)
        self.state.keys.extend(
            event.key
            for event in events.get_events()
            if isinstance(event, KeyboardUserInputEvent)
        )
        self.state.generated.set()
        if not self.state.ctx.is_main:
            return []
        return [
            StepResult(
                step_index=step_index,
                output=torch.zeros(1, 3, 2, 2),
                frame_count=1,
                output_layout=self.state.session_desc.output_layout,
            )
        ]

    def reset(self):
        self._model_thread()
        self.state.resets += 1

    def close(self):
        self.state.model_closes += 1


class _Session(ISession):
    def __init__(self, ctx, scenario):
        self.ctx = ctx
        self.scenario = scenario
        self.indices = []
        self.keys = []
        self.resets = 0
        self.model_closes = 0
        self.main_thread = threading.get_ident()
        self.generated = threading.Event()
        self.reset_sent = threading.Event()
        self.session_closes = 0

    @property
    def parallel_context(self):
        return self.ctx

    @property
    def session_desc(self):
        return SessionDesc(
            video_width=2,
            video_height=2,
            frames_per_second_for_ui=200,
            frames_per_second_for_step=200,
            presentation_mode=PresentationMode.ON_DEMAND,
        )

    def init(self):
        assert threading.get_ident() == self.main_thread
        if self.scenario == "init_failure" and self.ctx.is_main:
            raise ValueError("injected session initialization failure")
        self.register_model_loop(_Model, state=self)

    def close(self):
        assert threading.get_ident() == self.main_thread
        self.session_closes += 1


class _Window(IClientWindow):
    def __init__(self, session):
        self.session = session
        self.writes = 0

    def _main_thread(self):
        assert threading.get_ident() == self.session.main_thread
        assert self.session.ctx.is_main

    def open(self, desc):
        self._main_thread()

    def get_user_input_events(self):
        self._main_thread()
        if self.session.scenario == "window_failure":
            raise ValueError("injected window failure")
        if self.session.scenario == "close_before_start":
            return UserInputEvents([CloseUserInputEvent(timestamp=0)])
        if (
            self.session.scenario == "reset"
            and self.session.generated.is_set()
            and not self.session.reset_sent.is_set()
        ):
            self.session.reset_sent.set()
            return UserInputEvents(
                [
                    ResetUserInputEvent(timestamp=1),
                    KeyboardUserInputEvent(
                        timestamp=2, key="w", state=KeyboardInputState.PRESSED
                    ),
                ]
            )
        return UserInputEvents([])

    def write(self, result):
        self._main_thread()
        self.writes += 1

    def close(self):
        self._main_thread()


def _worker(rank, scenario, rendezvous, output):
    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        os.environ.pop(name, None)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=8),
    )
    records = []
    try:
        ctx = build_context(1, 2, torch.device("cpu"))
        for _ in range(2 if scenario == "repeat" else 1):
            session = _Session(ctx, scenario)
            window = _Window(session) if ctx.is_main else None
            ready_calls = 0
            original_ready = StepAgreement.ready

            def ready(agreement, **kwargs):
                nonlocal ready_calls
                admitted = original_ready(agreement, **kwargs)
                ready_calls += 1
                if (
                    scenario == "stop_after_admission"
                    and rank == 0
                    and ready_calls == 2
                ):
                    assert admitted
                    session._shutdown_event.set()
                return admitted

            failure = None
            try:
                with (
                    patch.object(StepAgreement, "ready", ready),
                    patch.object(
                        session_runner,
                        "StepAgreement",
                        lambda mesh: StepAgreement(mesh, timeout=timedelta(seconds=8)),
                    ),
                ):
                    session_runner.run_session(
                        session,
                        window,
                        steps=0
                        if scenario == "zero_steps"
                        else 2
                        if scenario == "bounded"
                        else None,
                    )
            except Exception as error:
                failure = str(error)
            assert not any(
                t.name == "flashdreams-model-generation-thread"
                for t in threading.enumerate()
            )
            records.append(
                {
                    "indices": session.indices,
                    "keys": session.keys,
                    "resets": session.resets,
                    "model_closes": session.model_closes,
                    "session_closes": session.session_closes,
                    "failure": failure,
                    "writes": 0 if window is None else window.writes,
                }
            )
        Path(output, f"rank{rank}.json").write_text(json.dumps(records))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "scenario",
    [
        "natural",
        "bounded",
        "zero_steps",
        "close_before_start",
        "stop_during_pacing",
        "stop_after_admission",
        "worker_finished",
        "reset",
        "prepare_failure",
        "step_failure",
        "init_failure",
        "window_failure",
        "repeat",
    ],
)
def test_runtime_ranks_stop_reset_and_fail_together(scenario, tmp_path):
    processes = [
        mp.get_context("spawn").Process(
            target=_worker,
            args=(rank, scenario, str(tmp_path / "rendezvous"), str(tmp_path)),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    try:
        for process in processes:
            process.join(timeout=45)
        assert [process.exitcode for process in processes] == [0, 0], (
            "runtime failed to exit within the process deadline"
        )
    finally:
        for process in processes:
            if process.is_alive():
                process.kill()
            process.join()
    results = [
        json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)
    ]
    for leader, worker in zip(*results, strict=True):
        assert leader["session_closes"] == worker["session_closes"] == 1
        assert worker["writes"] == 0
        if scenario.endswith("failure"):
            assert leader["failure"] is not None
            assert worker["failure"] is not None
            failing_rank = (
                leader if scenario in ("init_failure", "window_failure") else worker
            )
            assert "injected" in failing_rank["failure"]
            continue
        assert leader["failure"] is worker["failure"] is None
        assert leader["indices"] == worker["indices"]
        assert leader["model_closes"] == worker["model_closes"] == 1
        if scenario in ("zero_steps", "close_before_start", "stop_during_pacing"):
            assert leader["indices"] == []
        elif scenario in ("stop_after_admission", "worker_finished"):
            assert leader["indices"] == [0]
        elif scenario == "bounded":
            assert leader["indices"] == [0, 1]
        elif scenario == "reset":
            assert leader["resets"] == worker["resets"] == 1
            assert leader["keys"] == worker["keys"] == ["w"]
            assert leader["indices"].count(0) == 2
        else:
            assert leader["indices"] == [0, 1, 2]


def test_worker_cli_constructs_no_window_or_sink(monkeypatch):
    from types import SimpleNamespace

    from flashdreams.runtime_v2 import cli

    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(cli, "get_global_rank_for_logging", lambda: 1)
    monkeypatch.setattr(cli, "registered_application_slugs", lambda: [])
    monkeypatch.setattr(
        cli,
        "create_application",
        lambda slug: SimpleNamespace(session_desc=lambda: SessionDesc()),
    )
    received = []
    monkeypatch.setattr(
        cli,
        "ApplicationRunner",
        lambda application, window, **kwargs: SimpleNamespace(
            run=lambda *args, **run_kwargs: received.append((window, kwargs))
        ),
    )
    mode = cli.client_window_mode("webrtc")
    monkeypatch.setattr(
        mode, "create", lambda args: pytest.fail("worker created a window")
    )
    monkeypatch.setattr(
        cli, "MetricsOutputSink", lambda path: pytest.fail("worker created a sink")
    )
    cli.entrypoint(
        ["fake", "--mode", "webrtc", "--port", "8123", "--stats-path", "shared.json"]
    )
    assert received == [(None, {"metrics_output_sink": None})]
