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

"""Minimal loguru-compatible logger backed by the standard library.

The renderer only needs a small slice of the loguru API: a module-level
``logger`` whose ``debug``/``info``/``warning``/``error`` methods accept
loguru-style brace (``{}``) format strings with positional or keyword
arguments, e.g.::

    logger.info("Loaded {} in {:.1f}s", name, elapsed)

Providing that here lets us drop the third-party ``loguru`` dependency without
changing any call sites. Messages are written to stderr; the level defaults to
``INFO`` and can be overridden with the ``LUDUS_RENDERER_LOG_LEVEL`` environment
variable (e.g. ``DEBUG`` to restore loguru's verbose default).
"""

import logging
import os
import sys

_LOGGER_NAME = "ludus_renderer"


def _make_logger() -> logging.Logger:
    log = logging.getLogger(_LOGGER_NAME)
    # Attach our own stderr handler once so diagnostics show up when the
    # renderer is used directly (examples/CLI), without hijacking the root
    # logger or other libraries' configuration.
    if not log.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        log.addHandler(handler)
        log.propagate = False
        level_name = os.environ.get("LUDUS_RENDERER_LOG_LEVEL", "INFO").upper()
        log.setLevel(getattr(logging, level_name, logging.INFO))
    return log


class _BraceLogger:
    """Adapter exposing loguru-style ``{}`` formatting over stdlib logging."""

    def __init__(self) -> None:
        self._log = _make_logger()

    @staticmethod
    def _fmt(message: object, args: tuple, kwargs: dict) -> str:
        text = str(message)
        if args or kwargs:
            try:
                text = text.format(*args, **kwargs)
            except (IndexError, KeyError, ValueError):
                # Never let a logging call raise because of a format mismatch.
                pass
        return text

    def _emit(self, level: int, message: object, args: tuple, kwargs: dict) -> None:
        if self._log.isEnabledFor(level):
            self._log.log(level, self._fmt(message, args, kwargs))

    def debug(self, message: object, *args, **kwargs) -> None:
        self._emit(logging.DEBUG, message, args, kwargs)

    def info(self, message: object, *args, **kwargs) -> None:
        self._emit(logging.INFO, message, args, kwargs)

    def warning(self, message: object, *args, **kwargs) -> None:
        self._emit(logging.WARNING, message, args, kwargs)

    def error(self, message: object, *args, **kwargs) -> None:
        self._emit(logging.ERROR, message, args, kwargs)

    def exception(self, message: object, *args, **kwargs) -> None:
        if self._log.isEnabledFor(logging.ERROR):
            self._log.error(self._fmt(message, args, kwargs), exc_info=True)


logger = _BraceLogger()
