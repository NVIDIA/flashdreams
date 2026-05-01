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

from collections import deque
from enum import IntEnum
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from flashdreams.core.distributed.rank_orchestration import (
    PayloadBus,
    RankCoordinator,
    distributed_op,
)


class _Signal(IntEnum):
    STEP = 0
    EXIT = 1


class _FakeSignalBus:
    def __init__(self, recv_values: list[_Signal] | None = None) -> None:
        self.sent: list[_Signal] = []
        self._recv_values = deque(recv_values or [])

    def send(self, signal: _Signal) -> None:
        self.sent.append(signal)

    def recv(self) -> _Signal:
        if not self._recv_values:
            raise RuntimeError("No signal queued")
        return self._recv_values.popleft()


class _FakePayloadBus(PayloadBus):
    def __init__(self, *, queued_payloads: list[Any] | None = None) -> None:
        super().__init__(master_rank=0)
        self.sent_payloads: list[Any] = []
        self._queued_payloads = deque(queued_payloads or [])

    def broadcast_object(self, payload: Any) -> Any:
        self.sent_payloads.append(payload)
        if payload is not None:
            return payload
        if self._queued_payloads:
            return self._queued_payloads.popleft()
        return payload


class _DistributedTarget:
    def __init__(self, coordinator: RankCoordinator[_Signal]) -> None:
        self.rank_coordinator = coordinator
        self.observed_payloads: list[int] = []

    @distributed_op(_Signal.STEP)
    def step(self, payload: int) -> int:
        self.observed_payloads.append(payload)
        return payload * 2


def test_master_invoke_broadcasts_signal_and_payload() -> None:
    signal_bus = _FakeSignalBus()
    payload_bus = _FakePayloadBus()
    coordinator: RankCoordinator[_Signal] = RankCoordinator(  # ty:ignore[invalid-assignment]
        device=torch.device("cpu"),
        signal_type=_Signal,
        signal_bus=signal_bus,  # ty:ignore[invalid-argument-type]
        payload_bus=payload_bus,
        is_master=True,
    )
    target = _DistributedTarget(coordinator)

    result = target.step(21)

    assert result == 42
    assert signal_bus.sent == [_Signal.STEP]
    assert len(payload_bus.sent_payloads) == 1
    assert payload_bus.sent_payloads[0].args == (21,)
    assert payload_bus.sent_payloads[0].kwargs == {}
    assert target.observed_payloads == [21]


def test_worker_invoke_receives_broadcast_payload() -> None:
    signal_bus = _FakeSignalBus()
    payload_bus = _FakePayloadBus(
        queued_payloads=[SimpleNamespace(args=(7,), kwargs={})]
    )
    coordinator: RankCoordinator[_Signal] = RankCoordinator(  # ty:ignore[invalid-assignment]
        device=torch.device("cpu"),
        signal_type=_Signal,
        signal_bus=signal_bus,  # ty:ignore[invalid-argument-type]
        payload_bus=payload_bus,
        is_master=False,
    )
    target = _DistributedTarget(coordinator)

    result = target.step()

    assert result == 14
    assert signal_bus.sent == []
    assert payload_bus.sent_payloads == [None]
    assert target.observed_payloads == [7]


def test_worker_rejects_locally_provided_payload() -> None:
    signal_bus = _FakeSignalBus()
    payload_bus = _FakePayloadBus()
    coordinator: RankCoordinator[_Signal] = RankCoordinator(  # ty:ignore[invalid-assignment]
        device=torch.device("cpu"),
        signal_type=_Signal,
        signal_bus=signal_bus,  # ty:ignore[invalid-argument-type]
        payload_bus=payload_bus,
        is_master=False,
    )
    target = _DistributedTarget(coordinator)

    with pytest.raises(AssertionError):
        target.step(1)


def test_worker_loop_dispatches_registered_distributed_ops() -> None:
    signal_bus = _FakeSignalBus(recv_values=[_Signal.STEP, _Signal.EXIT])
    payload_bus = _FakePayloadBus(
        queued_payloads=[SimpleNamespace(args=(9,), kwargs={})]
    )
    coordinator: RankCoordinator[_Signal] = RankCoordinator(  # ty:ignore[invalid-assignment]
        device=torch.device("cpu"),
        signal_type=_Signal,
        signal_bus=signal_bus,  # ty:ignore[invalid-argument-type]
        payload_bus=payload_bus,
        is_master=False,
    )
    target = _DistributedTarget(coordinator)
    coordinator.register_distributed_ops(target)

    coordinator.worker_loop(exit_signal=_Signal.EXIT)

    assert target.observed_payloads == [9]


def test_worker_cannot_send_exit_signal() -> None:
    signal_bus = _FakeSignalBus()
    payload_bus = _FakePayloadBus()
    coordinator: RankCoordinator[_Signal] = RankCoordinator(  # ty:ignore[invalid-assignment]
        device=torch.device("cpu"),
        signal_type=_Signal,
        signal_bus=signal_bus,  # ty:ignore[invalid-argument-type]
        payload_bus=payload_bus,
        is_master=False,
    )

    with pytest.raises(RuntimeError):
        coordinator.send_exit(exit_signal=_Signal.EXIT)
