# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import os
import sys

from loguru import logger

_LOG_FORMAT = "{time:HH:mm:ss.SSS} | {level:<7} | {message}"


def configure_logging() -> None:
    """Use compact Loguru output for interactive-drive CLI sessions."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=os.environ.get("LOGURU_LEVEL", "INFO"),
        format=_LOG_FORMAT,
    )
