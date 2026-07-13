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

from __future__ import annotations

import multiprocessing
import time

import pytest

from flashdreams.serving.webrtc import bootstrap

pytestmark = pytest.mark.ci_cpu


class _FakeSessionManager:
    def __init__(self) -> None:
        self.exit_called = False

    def send_exit_signal(self) -> None:
        self.exit_called = True

    def wait_for_termination(self) -> None:
        raise AssertionError("Rank 0 must not wait for a termination signal.")


def test_interrupted_server_terminates_child_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = multiprocessing.Process(
        target=time.sleep,
        args=(60.0,),
        name="checkpoint-download-worker",
    )
    manager = _FakeSessionManager()

    def interrupted_run_app(*args: object, **kwargs: object) -> None:
        del args, kwargs
        child.start()
        raise KeyboardInterrupt

    monkeypatch.setattr(bootstrap.web, "run_app", interrupted_run_app)

    try:
        with pytest.raises(KeyboardInterrupt):
            bootstrap.run_webrtc_server(
                world_rank=0,
                session_manager=manager,
                app=bootstrap.web.Application(),
                host="127.0.0.1",
                port=8080,
            )

        child.join(timeout=5.0)
        assert not child.is_alive()
        assert child.exitcode is not None
        assert manager.exit_called
    finally:
        if child.is_alive():
            child.kill()
        child.join()
