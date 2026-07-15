# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from flashdreams.core import distributed

pytestmark = pytest.mark.ci_cpu


@pytest.fixture(autouse=True)
def _restore_loguru(monkeypatch: pytest.MonkeyPatch):
    yield
    monkeypatch.delenv("LOGURU_LEVEL", raising=False)
    distributed.configure_loguru_for_distributed(world_rank=0)


def test_configure_loguru_for_distributed_demotes_non_rank0_records(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("LOGURU_LEVEL", "INFO")
    distributed.configure_loguru_for_distributed(world_rank=1)
    distributed.logger.warning("hidden non-rank0 warning")
    assert "hidden non-rank0 warning" not in capfd.readouterr().err

    monkeypatch.setenv("LOGURU_LEVEL", "DEBUG")
    distributed.configure_loguru_for_distributed(world_rank=1)
    distributed.logger.warning("visible non-rank0 warning")
    err = capfd.readouterr().err
    assert "visible non-rank0 warning" in err
    assert "DEBUG" in err


def test_configure_loguru_for_distributed_keeps_rank0_record_levels(
    monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
):
    monkeypatch.setenv("LOGURU_LEVEL", "INFO")
    distributed.configure_loguru_for_distributed(world_rank=0)
    distributed.logger.warning("visible rank0 warning")
    err = capfd.readouterr().err
    assert "visible rank0 warning" in err
    assert "WARNING" in err


def test_get_global_rank_for_logging_uses_env_before_distributed_init(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setattr(distributed, "is_distributed_initialized", lambda: False)

    assert distributed.get_global_rank_for_logging() == 7


def test_shutdown_synchronizes_successful_run_before_destroy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized = True
    calls: list[object] = []

    def destroy() -> None:
        nonlocal initialized
        calls.append("destroy")
        initialized = False

    monkeypatch.setattr(distributed, "is_distributed_initialized", lambda: initialized)
    monkeypatch.setattr(
        distributed.dist,
        "barrier",
        lambda **kwargs: calls.append(("barrier", kwargs)),
    )
    monkeypatch.setattr(distributed.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(distributed.dist, "destroy_process_group", destroy)
    monkeypatch.setattr(distributed.torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(
        distributed.atexit,
        "unregister",
        lambda callback: calls.append(("unregister", callback)),
    )

    distributed.shutdown(synchronize=True)

    assert calls == [
        ("barrier", {"device_ids": [3]}),
        "destroy",
        ("unregister", distributed._safe_destroy_pg),
    ]


def test_shutdown_skips_destroy_after_unsynchronized_rank_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    monkeypatch.setattr(distributed, "is_distributed_initialized", lambda: True)
    monkeypatch.setattr(distributed.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(
        distributed.dist,
        "destroy_process_group",
        lambda: calls.append("destroy"),
    )
    monkeypatch.setattr(
        distributed.atexit,
        "unregister",
        lambda callback: calls.append(("unregister", callback)),
    )

    distributed.shutdown(synchronize=False)

    assert calls == [("unregister", distributed._safe_destroy_pg)]


def test_shutdown_destroys_group_when_readiness_barrier_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized = True
    destroyed = False

    def fail_barrier(**kwargs: object) -> None:
        del kwargs
        raise RuntimeError("barrier failed")

    def destroy() -> None:
        nonlocal initialized, destroyed
        initialized = False
        destroyed = True

    monkeypatch.setattr(distributed, "is_distributed_initialized", lambda: initialized)
    monkeypatch.setattr(distributed.dist, "barrier", fail_barrier)
    monkeypatch.setattr(distributed.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(distributed.dist, "destroy_process_group", destroy)
    monkeypatch.setattr(distributed.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(distributed.atexit, "unregister", lambda _: None)

    with pytest.raises(RuntimeError, match="barrier failed"):
        distributed.shutdown(synchronize=True)

    assert destroyed


def test_shutdown_terminates_nccl_process_after_successful_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialized = True
    calls: list[str] = []

    def exit_process(code: int) -> None:
        calls.append(f"exit:{code}")
        raise SystemExit(code)

    monkeypatch.setattr(distributed, "is_distributed_initialized", lambda: initialized)
    monkeypatch.setattr(
        distributed.dist, "barrier", lambda **_: calls.append("barrier")
    )
    monkeypatch.setattr(distributed.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(distributed.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(distributed.atexit, "unregister", lambda _: None)
    monkeypatch.setattr(distributed.os, "_exit", exit_process)

    with pytest.raises(SystemExit) as exc_info:
        distributed.shutdown(synchronize=True, terminate_process=True)

    assert exc_info.value.code == 0
    assert calls == ["barrier", "exit:0"]


def test_shutdown_does_not_mask_failed_nccl_barrier_with_success_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_barrier(**kwargs: object) -> None:
        del kwargs
        raise RuntimeError("barrier failed")

    monkeypatch.setattr(distributed, "is_distributed_initialized", lambda: True)
    monkeypatch.setattr(distributed.dist, "barrier", fail_barrier)
    monkeypatch.setattr(distributed.dist, "get_backend", lambda: "nccl")
    monkeypatch.setattr(distributed.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(distributed.atexit, "unregister", lambda _: None)
    monkeypatch.setattr(
        distributed.os,
        "_exit",
        lambda code: pytest.fail(f"unexpected success exit {code}"),
    )

    with pytest.raises(RuntimeError, match="barrier failed"):
        distributed.shutdown(synchronize=True, terminate_process=True)
