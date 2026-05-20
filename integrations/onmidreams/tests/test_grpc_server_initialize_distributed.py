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

import pytest
import torch
from onmidreams.grpc import server

pytestmark = pytest.mark.ci_gpu


def test_initialize_distributed_single_process_defaults(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    set_device_calls: list[torch.device] = []
    monkeypatch.setattr(server.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(server.torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(
        server.torch.cuda, "set_device", lambda device: set_device_calls.append(device)
    )

    device, world_rank, context_parallel_size = server.initialize_distributed(
        n_cameras=1
    )

    assert device == torch.device("cuda:0")
    assert world_rank == 0
    assert context_parallel_size == 1
    assert set_device_calls == [torch.device("cuda:0")]


def test_initialize_distributed_derives_cp_from_world_size(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "4")

    init_calls = 0

    def _fake_distributed_init() -> None:
        nonlocal init_calls
        init_calls += 1

    set_device_calls: list[torch.device] = []
    monkeypatch.setattr(server, "distributed_init", _fake_distributed_init)
    monkeypatch.setattr(server.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(server.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(server.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(server.torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(
        server.torch.cuda, "set_device", lambda device: set_device_calls.append(device)
    )

    device, world_rank, context_parallel_size = server.initialize_distributed(
        n_cameras=2
    )

    assert init_calls == 1
    assert device == torch.device("cuda:1")
    assert world_rank == 1
    assert context_parallel_size == 4
    assert set_device_calls == [torch.device("cuda:1")]


def test_initialize_distributed_validates_camera_divisibility(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "4")

    monkeypatch.setattr(server, "distributed_init", lambda: None)
    monkeypatch.setattr(server.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(server.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(server.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(server.torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(server.torch.cuda, "set_device", lambda device: None)

    with pytest.raises(ValueError, match="must divide n_cameras"):
        server.initialize_distributed(n_cameras=3)


def test_initialize_distributed_requires_cuda(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(server.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is required"):
        server.initialize_distributed(n_cameras=1)


def test_initialize_distributed_requires_rank_and_world_size_pair(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(server.torch.cuda, "is_available", lambda: True)

    with pytest.raises(RuntimeError, match="both RANK and WORLD_SIZE"):
        server.initialize_distributed(n_cameras=1)
