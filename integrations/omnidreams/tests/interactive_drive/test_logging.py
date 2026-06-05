# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import sys

from loguru import logger
from omnidreams.interactive_drive.log import configure_logging


def test_configure_logging_uses_compact_default(
    monkeypatch,
    capfd,
) -> None:
    monkeypatch.setenv("LOGURU_LEVEL", "INFO")

    configure_logging()
    try:
        logger.info("[world-model] compact prefix")

        err = capfd.readouterr().err
        assert "[world-model] compact prefix" in err
        assert "INFO" in err
        assert "omnidreams.interactive_drive" not in err
    finally:
        logger.remove()
        logger.add(sys.stderr)
