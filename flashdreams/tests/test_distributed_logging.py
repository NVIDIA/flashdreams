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
