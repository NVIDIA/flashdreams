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

"""Process-wide stdlib :mod:`logging` filters installed at import time.

Currently houses one filter: :class:`_DowngradeInductorAutotuneFallback`.
:func:`install_inductor_autotune_demote` is invoked at module load so any
FlashDreams entry point that transitively imports :mod:`flashdreams.core`
inherits the demotion without explicit setup.
"""

from __future__ import annotations

import logging

_AUTOTUNE_LOGGER_NAME = "torch._inductor.select_algorithm"

# Inductor logs two structurally-equivalent autotuner fallbacks at ERROR
# level (one via ``log.error`` for ``RuntimeError``, one via
# ``log.exception`` for ``CUDACompileError``). Both are expected: the
# autotuner tried a template, it raised, the template is skipped, and a
# valid candidate is selected. Match by message prefix because Inductor
# resolves the formatted message lazily and the exception body varies.
_AUTOTUNE_FALLBACK_PREFIXES: tuple[str, ...] = (
    "Runtime error during autotuning",
    "CUDA compilation error during autotuning",
)


class _DowngradeInductorAutotuneFallback(logging.Filter):
    """Demote benign Inductor autotuner-fallback ERROR records to WARNING.

    ``torch._inductor.select_algorithm`` surfaces candidate-kernel
    rejections at ERROR level even though they're an expected part of a
    healthy autotune pass. Users hitting these mid-warmup reasonably
    assume something is broken; downgrading the level keeps the trail
    without the alarm.

    Mutating ``record.levelno`` / ``record.levelname`` inside a filter
    is supported by :mod:`logging`: the record is mutated before any
    handler's own level check runs, so handlers that emit at WARNING
    will still emit (just under a less alarming label).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.ERROR:
            return True
        message = record.getMessage()
        if any(message.startswith(prefix) for prefix in _AUTOTUNE_FALLBACK_PREFIXES):
            record.levelno = logging.WARNING
            record.levelname = "WARNING"
        return True


def install_inductor_autotune_demote() -> None:
    """Attach the autotune-fallback demoter to the Inductor logger.

    Idempotent: if a filter of this class is already attached (e.g.
    because :mod:`flashdreams.core.log_filters` was re-imported during a
    long test session), we leave the existing instance in place rather
    than stacking duplicates.
    """
    autotune_logger = logging.getLogger(_AUTOTUNE_LOGGER_NAME)
    for existing in autotune_logger.filters:
        if isinstance(existing, _DowngradeInductorAutotuneFallback):
            return
    autotune_logger.addFilter(_DowngradeInductorAutotuneFallback())


install_inductor_autotune_demote()
