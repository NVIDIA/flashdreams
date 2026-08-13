# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from flashdreams.runtime.demo import bootstrap as runtime_bootstrap
from flashdreams.serving.webrtc import bootstrap

pytestmark = pytest.mark.ci_cpu


def test_initialize_cuda_distributed_single_process_uses_default_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)

    set_device_calls: list[torch.device] = []
    logging_ranks: list[int | None] = []
    monkeypatch.setattr(bootstrap.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bootstrap.torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(
        bootstrap.torch.cuda,
        "set_device",
        lambda device: set_device_calls.append(device),
    )

    context = bootstrap.initialize_cuda_distributed(
        default_device="cuda:2",
        configure_logging_fn=lambda *, world_rank: logging_ranks.append(world_rank),
    )

    assert context.device == torch.device("cuda:2")
    assert context.world_rank == 0
    assert context.world_size == 1
    assert set_device_calls == [torch.device("cuda:2")]
    assert logging_ranks == [0]


def test_initialize_cuda_distributed_defaults_unspecified_cuda_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    set_device_calls: list[torch.device] = []
    monkeypatch.setattr(bootstrap.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bootstrap.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        bootstrap.torch.cuda,
        "set_device",
        lambda device: set_device_calls.append(device),
    )

    context = bootstrap.initialize_cuda_distributed(
        default_device="cuda",
        configure_logging_fn=lambda *, world_rank: None,
    )

    assert context.device == torch.device("cuda:0")
    assert set_device_calls == [torch.device("cuda:0")]


def test_initialize_cuda_distributed_uses_local_rank_for_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")

    init_calls = 0

    def _fake_distributed_init() -> None:
        nonlocal init_calls
        init_calls += 1

    set_device_calls: list[torch.device] = []
    monkeypatch.setattr(bootstrap.dist, "get_rank", lambda: 3)
    monkeypatch.setattr(bootstrap.dist, "get_world_size", lambda: 8)
    monkeypatch.setattr(bootstrap.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bootstrap.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        bootstrap.torch.cuda,
        "set_device",
        lambda device: set_device_calls.append(device),
    )

    context = bootstrap.initialize_cuda_distributed(
        distributed_init_fn=_fake_distributed_init,
        configure_logging_fn=lambda *, world_rank: None,
    )

    assert init_calls == 1
    assert context.device == torch.device("cuda:1")
    assert context.world_rank == 3
    assert context.world_size == 8
    assert set_device_calls == [torch.device("cuda:1")]


def test_initialize_cuda_distributed_requires_rank_world_size_pair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(bootstrap.torch.cuda, "is_available", lambda: True)

    with pytest.raises(RuntimeError, match="both RANK and WORLD_SIZE"):
        bootstrap.initialize_cuda_distributed()


def test_initialize_cuda_distributed_rejects_cpu_default_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(bootstrap.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bootstrap.torch.cuda, "device_count", lambda: 1)

    with pytest.raises(RuntimeError, match="CUDA device is required"):
        bootstrap.initialize_cuda_distributed(default_device="cpu")


def test_cleanup_cuda_distributed_destroys_group_without_barrier() -> None:
    fake_torch = _FakeTorch()
    fake_dist = _FakeDist()

    runtime_bootstrap.cleanup_cuda_distributed(
        world_rank=0,
        synchronize_distributed=False,
        torch_module=fake_torch,
        dist_module=fake_dist,
    )

    assert fake_torch.cuda.empty_cache_calls == 1
    assert fake_torch.cuda.synchronize_calls == 1
    assert fake_dist.barrier_calls == 0
    assert fake_dist.destroy_process_group_calls == 1


def test_run_webrtc_server_cleans_process_state_when_startup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    manager = _FakeServerLifecycle(events)
    cleanup_calls: list[dict[str, object]] = []

    def fail_to_run_app(*_: object, **__: object) -> None:
        raise RuntimeError("server startup failed")

    def record_cleanup(**kwargs: object) -> None:
        events.append("process-cleanup")
        cleanup_calls.append(kwargs)

    monkeypatch.setattr(bootstrap.web, "run_app", fail_to_run_app)
    monkeypatch.setattr(bootstrap, "cleanup_cuda_distributed", record_cleanup)

    with pytest.raises(RuntimeError, match="server startup failed"):
        bootstrap.run_webrtc_server(
            world_rank=0,
            session_manager=manager,
            app=bootstrap.web.Application(),
            host="127.0.0.1",
            port=8080,
        )

    assert manager.send_exit_signal_calls == 1
    assert events == ["send-exit", "shutdown", "process-cleanup"]
    assert len(cleanup_calls) == 1
    cleanup_call = cleanup_calls[0]
    assert cleanup_call["world_rank"] == 0
    assert cleanup_call["synchronize_distributed"] is False
    assert cleanup_call["torch_module"] is bootstrap.torch
    assert cleanup_call["dist_module"] is bootstrap.dist


class _FakeCuda:
    def __init__(self) -> None:
        self.empty_cache_calls = 0
        self.synchronize_calls = 0

    def is_available(self) -> bool:
        return True

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def synchronize(self) -> None:
        self.synchronize_calls += 1


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()


class _FakeDist:
    def __init__(self) -> None:
        self.barrier_calls = 0
        self.destroy_process_group_calls = 0

    def is_available(self) -> bool:
        return True

    def is_initialized(self) -> bool:
        return True

    def barrier(self) -> None:
        self.barrier_calls += 1

    def destroy_process_group(self) -> None:
        self.destroy_process_group_calls += 1


class _FakeServerLifecycle:
    def __init__(self, events: list[str] | None = None) -> None:
        self.events = events
        self.send_exit_signal_calls = 0
        self.shutdown_calls = 0

    def send_exit_signal(self) -> None:
        self.send_exit_signal_calls += 1
        if self.events is not None:
            self.events.append("send-exit")

    def wait_for_termination(self) -> None:
        raise AssertionError("rank 0 should not wait for termination")

    async def shutdown(self) -> None:
        self.shutdown_calls += 1
        if self.events is not None:
            self.events.append("shutdown")
