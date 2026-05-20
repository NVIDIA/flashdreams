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

import functools
import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Generic, TypeVar, cast

import torch
import torch.distributed as dist

_PayloadT = TypeVar("_PayloadT")
_ResultT = TypeVar("_ResultT")
_SignalT = TypeVar("_SignalT", bound=IntEnum)

_DISTRIBUTED_OP_ATTR = "__distributed_op_spec__"


@dataclass(frozen=True, slots=True)
class DistributedOpSpec(Generic[_SignalT]):
    signal: _SignalT
    method_name: str


@dataclass(frozen=True, slots=True)
class _InvocationPayload:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class SignalBus(Generic[_SignalT]):
    """Broadcast control signals from master to workers in strict order."""

    def __init__(
        self,
        *,
        device: torch.device,
        signal_type: type[_SignalT],
        master_rank: int = 0,
    ) -> None:
        self.device = device
        self.signal_type = signal_type
        self.master_rank = master_rank
        self._counter = 0

    def send(self, signal: _SignalT) -> None:
        encoded_signal = torch.tensor(
            [self._counter, int(signal)], dtype=torch.int64, device=self.device
        )
        if dist.is_initialized():
            dist.broadcast(encoded_signal, src=self.master_rank)
        self._counter += 1

    def recv(self) -> _SignalT:
        if not dist.is_initialized():
            raise RuntimeError(
                "Cannot receive distributed signals without process group"
            )

        packet = torch.tensor([self._counter, 0], dtype=torch.int64, device=self.device)
        dist.broadcast(packet, src=self.master_rank)
        received_counter = int(packet[0].item())
        if received_counter != self._counter:
            raise RuntimeError(
                f"Signal counter mismatch: got {received_counter}, expected {self._counter}"
            )
        self._counter += 1
        return self.signal_type(int(packet[1].item()))


class PayloadBus:
    """Broadcast picklable payloads from master to workers."""

    def __init__(self, *, master_rank: int = 0) -> None:
        self.master_rank = master_rank

    def broadcast_object(self, payload: _PayloadT) -> _PayloadT:
        if not dist.is_initialized():
            return payload

        payload_list = [payload]
        dist.broadcast_object_list(payload_list, src=self.master_rank)
        return payload_list[0]


@dataclass(slots=True)
class _RegisteredHandler:
    method_name: str
    callback: Callable[[], Any]


class RankCoordinator(Generic[_SignalT]):
    """Coordinates rank0-driven distributed operation invocation."""

    def __init__(
        self,
        *,
        device: torch.device,
        signal_type: type[_SignalT],
        is_master: bool,
        master_rank: int = 0,
        signal_bus: SignalBus[_SignalT] | None = None,
        payload_bus: PayloadBus | None = None,
    ) -> None:
        self.signal_bus = signal_bus or SignalBus(
            device=device,
            signal_type=signal_type,
            master_rank=master_rank,
        )
        self.payload_bus = payload_bus or PayloadBus(master_rank=master_rank)
        self.is_master = is_master
        self._lock = threading.RLock()
        self._handlers: dict[_SignalT, _RegisteredHandler] = {}

    def invoke(
        self,
        *,
        signal: _SignalT,
        payload: _PayloadT | None,
        handler: Callable[[_PayloadT], _ResultT],
    ) -> _ResultT:
        with self._lock:
            if self.is_master:
                self.signal_bus.send(signal)
            elif payload is not None:
                raise AssertionError(
                    f"Non-master rank cannot provide payload for signal {signal.name}"
                )

            synced_payload = self.payload_bus.broadcast_object(payload)
            if synced_payload is None:
                raise AssertionError(f"Synchronized payload for {signal.name} is None")
            return handler(cast(_PayloadT, synced_payload))

    def register(
        self, *, signal: _SignalT, method_name: str, callback: Callable[[], Any]
    ) -> None:
        existing = self._handlers.get(signal)
        if existing is not None and existing.method_name != method_name:
            raise ValueError(
                f"Signal {signal.name} already registered to {existing.method_name}"
            )
        self._handlers[signal] = _RegisteredHandler(
            method_name=method_name,
            callback=callback,
        )

    def register_distributed_ops(self, obj: Any) -> None:
        for method_name in dir(type(obj)):
            method_obj = getattr(type(obj), method_name, None)
            spec = getattr(method_obj, _DISTRIBUTED_OP_ATTR, None)
            if not isinstance(spec, DistributedOpSpec):
                continue
            bound_method = getattr(obj, method_name)
            self.register(
                signal=spec.signal,
                method_name=method_name,
                callback=lambda bound_method=bound_method: bound_method(),
            )

    def worker_loop(self, *, exit_signal: _SignalT) -> None:
        while True:
            signal = self.signal_bus.recv()
            if signal == exit_signal:
                break
            handler = self._handlers.get(signal)
            if handler is None:
                raise ValueError(
                    f"No distributed handler registered for signal {signal.name}"
                )
            handler.callback()

    def send_exit(self, *, exit_signal: _SignalT) -> None:
        if not self.is_master:
            raise RuntimeError("Only master rank can send exit signal")
        self.signal_bus.send(exit_signal)


def distributed_op(
    signal: _SignalT,
    *,
    coordinator_attr: str = "rank_coordinator",
) -> Callable[[Callable[..., _ResultT]], Callable[..., _ResultT]]:
    """Decorate a method as a distributed operation coordinated by rank0.

    Decorated methods can use regular positional/keyword arguments.
    Rank 0 arguments are serialized and broadcast to workers.
    """

    def decorator(method: Callable[..., _ResultT]) -> Callable[..., _ResultT]:
        @functools.wraps(method)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> _ResultT:
            coordinator = getattr(self, coordinator_attr)
            if not isinstance(coordinator, RankCoordinator):
                raise TypeError(
                    f"Expected `{coordinator_attr}` to be RankCoordinator, got {type(coordinator)}"
                )
            payload: _InvocationPayload | None
            if coordinator.is_master:
                payload = _InvocationPayload(args=tuple(args), kwargs=dict(kwargs))
            else:
                if args or kwargs:
                    raise AssertionError(
                        f"Non-master rank cannot provide call arguments for signal {signal.name}"
                    )
                payload = None

            def invoke_method(invocation_payload: _InvocationPayload) -> _ResultT:
                return method(
                    self, *invocation_payload.args, **invocation_payload.kwargs
                )

            return coordinator.invoke(
                signal=signal,
                payload=payload,
                handler=invoke_method,
            )

        setattr(
            wrapper,
            _DISTRIBUTED_OP_ATTR,
            DistributedOpSpec(
                signal=signal,
                method_name=getattr(method, "__name__", "distributed_op"),
            ),
        )
        return wrapper

    return decorator
