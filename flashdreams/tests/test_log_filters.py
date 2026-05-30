# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import pytest

from flashdreams.core import log_filters

pytestmark = pytest.mark.ci_cpu


@pytest.fixture
def autotune_logger() -> logging.Logger:
    """Return the Inductor autotune logger with the demoter freshly attached."""
    return logging.getLogger("torch._inductor.select_algorithm")


def _make_record(level: int, msg: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="torch._inductor.select_algorithm",
        level=level,
        pathname=__file__,
        lineno=0,
        msg=msg,
        args=None,
        exc_info=None,
    )


def test_demotes_runtime_error_during_autotuning_to_warning():
    record = _make_record(
        logging.ERROR,
        "Runtime error during autotuning: \npermute(sparse_coo) failed. \nIgnoring this choice.",
    )
    log_filters._DowngradeInductorAutotuneFallback().filter(record)
    assert record.levelno == logging.WARNING
    assert record.levelname == "WARNING"


def test_demotes_cuda_compilation_error_during_autotuning_to_warning():
    record = _make_record(
        logging.ERROR,
        "CUDA compilation error during autotuning: \nptxas: foo. \nIgnoring this choice.",
    )
    log_filters._DowngradeInductorAutotuneFallback().filter(record)
    assert record.levelno == logging.WARNING
    assert record.levelname == "WARNING"


def test_leaves_unrelated_error_records_untouched():
    record = _make_record(logging.ERROR, "kernel launch failed: invalid configuration")
    log_filters._DowngradeInductorAutotuneFallback().filter(record)
    assert record.levelno == logging.ERROR
    assert record.levelname == "ERROR"


def test_leaves_warning_records_untouched():
    record = _make_record(
        logging.WARNING,
        "Runtime error during autotuning: \nshape mismatch. \nIgnoring this choice.",
    )
    log_filters._DowngradeInductorAutotuneFallback().filter(record)
    assert record.levelno == logging.WARNING
    assert record.levelname == "WARNING"


def test_installer_is_idempotent(autotune_logger: logging.Logger):
    # The module already self-installed at import; calling again shouldn't
    # stack a second filter instance.
    before = sum(
        isinstance(f, log_filters._DowngradeInductorAutotuneFallback)
        for f in autotune_logger.filters
    )
    log_filters.install_inductor_autotune_demote()
    log_filters.install_inductor_autotune_demote()
    after = sum(
        isinstance(f, log_filters._DowngradeInductorAutotuneFallback)
        for f in autotune_logger.filters
    )
    assert before == after == 1


def test_end_to_end_logger_emits_at_warning_level(autotune_logger: logging.Logger):
    # Attach our own capturing handler directly to the autotune logger
    # rather than relying on pytest's ``caplog`` (which needs record
    # propagation to the root logger). Other imports in a shared test
    # session — notably torch's logging setup — reconfigure stdlib
    # logging and break that propagation, so capture at the source.
    # The logger-level filter runs in ``Logger.handle`` before handlers
    # are invoked, so a handler on this logger sees the mutated record.
    captured: list[logging.LogRecord] = []

    class _CaptureHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            captured.append(record)

    handler = _CaptureHandler(level=logging.DEBUG)
    autotune_logger.addHandler(handler)
    previous_level = autotune_logger.level
    autotune_logger.setLevel(logging.DEBUG)
    try:
        autotune_logger.error(
            "Runtime error during autotuning: \n%s. \nIgnoring this choice.",
            "permute(sparse_coo)",
        )
    finally:
        autotune_logger.removeHandler(handler)
        autotune_logger.setLevel(previous_level)

    assert len(captured) == 1
    assert captured[0].levelno == logging.WARNING
    assert captured[0].levelname == "WARNING"
    assert "permute(sparse_coo)" in captured[0].getMessage()
