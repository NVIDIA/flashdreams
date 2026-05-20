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

from argparse import Namespace

import pytest
import torch
from lingbot.webrtc import server

pytestmark = pytest.mark.ci_gpu


class _FakeSessionManager:
    def __init__(self) -> None:
        self.wait_called = False
        self.exit_called = False

    def wait_for_termination(self) -> None:
        self.wait_called = True

    def send_exit_signal(self) -> None:
        self.exit_called = True


def _args(device: str = "cuda:0") -> Namespace:
    return Namespace(
        host="127.0.0.1",
        port=8080,
        config_name="LingBot-World-Fast",
        no_compile=False,
        device=device,
        warmup_chunks=10,
        warmup_timeout_s=600.0,
    )


def test_initialize_distributed_single_process_honors_default_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    set_device_calls: list[torch.device] = []
    monkeypatch.setattr(server.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(server.torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(
        server.torch.cuda, "set_device", lambda device: set_device_calls.append(device)
    )

    device, world_rank, context_parallel_size = server.initialize_distributed(
        default_device="cuda:2"
    )

    assert device == torch.device("cuda:2")
    assert world_rank == 0
    assert context_parallel_size == 1
    assert set_device_calls == [torch.device("cuda:2")]


def test_initialize_distributed_derives_cp_from_world_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")

    init_calls = 0

    def _fake_distributed_init() -> None:
        nonlocal init_calls
        init_calls += 1

    set_device_calls: list[torch.device] = []
    monkeypatch.setattr(server, "distributed_init", _fake_distributed_init)
    monkeypatch.setattr(server.dist, "get_rank", lambda: 3)
    monkeypatch.setattr(server.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(server.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(server.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        server.torch.cuda, "set_device", lambda device: set_device_calls.append(device)
    )

    device, world_rank, context_parallel_size = server.initialize_distributed()

    assert init_calls == 1
    assert device == torch.device("cuda:1")
    assert world_rank == 3
    assert context_parallel_size == 4
    assert set_device_calls == [torch.device("cuda:1")]


def test_initialize_distributed_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(server.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is required"):
        server.initialize_distributed()


def test_initialize_distributed_requires_rank_and_world_size_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(server.torch.cuda, "is_available", lambda: True)

    with pytest.raises(RuntimeError, match="both RANK and WORLD_SIZE"):
        server.initialize_distributed()


def test_initialize_distributed_rejects_cpu_default_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(server.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(server.torch.cuda, "device_count", lambda: 1)

    with pytest.raises(RuntimeError, match="CUDA device is required"):
        server.initialize_distributed(default_device="cpu")


def test_main_rank0_sends_exit_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_manager = _FakeSessionManager()
    runtime_configs = []

    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(server, "parse_args", lambda: _args(device="cuda:2"))
    monkeypatch.setattr(
        server,
        "initialize_distributed",
        lambda default_device: (torch.device("cuda:2"), 0, 1),
    )

    def _make_manager(runtime_config):
        runtime_configs.append(runtime_config)
        return fake_manager

    monkeypatch.setattr(server, "LingbotWebRTCSessionManager", _make_manager)
    monkeypatch.setattr(server, "create_app", lambda session_manager: object())
    monkeypatch.setattr(server.web, "run_app", lambda app, host, port: None)
    monkeypatch.setattr(server.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(server.dist, "is_initialized", lambda: False)

    server.main()

    assert fake_manager.exit_called is True
    assert fake_manager.wait_called is False
    assert runtime_configs[0].device == "cuda:2"
    assert runtime_configs[0].context_parallel_size == 1


def test_main_worker_rank_waits_for_termination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_manager = _FakeSessionManager()
    runtime_configs = []

    monkeypatch.setattr(server, "parse_args", _args)
    monkeypatch.setattr(
        server,
        "initialize_distributed",
        lambda default_device: (torch.device("cuda:1"), 1, 2),
    )
    monkeypatch.setattr(server.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(server.torch.cuda, "is_available", lambda: False)

    def _make_manager(runtime_config):
        runtime_configs.append(runtime_config)
        return fake_manager

    monkeypatch.setattr(server, "LingbotWebRTCSessionManager", _make_manager)

    server.main()

    assert fake_manager.wait_called is True
    assert fake_manager.exit_called is False
    assert runtime_configs[0].device == "cuda:1"
    assert runtime_configs[0].context_parallel_size == 2
