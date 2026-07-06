# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional video-quality evaluator registry.

The first implementation keeps learned/reference/VLM evaluators out of the
blocking path. Future evaluators can register callables here without changing
the runner loop.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Evaluator = Callable[..., dict[str, Any]]

_EVALUATORS: dict[str, Evaluator] = {}


def register_evaluator(name: str, evaluator: Evaluator) -> None:
    """Register an optional evaluator by name."""
    if not name:
        raise ValueError("Evaluator name must be non-empty")
    if name in _EVALUATORS:
        raise ValueError(f"Evaluator {name!r} is already registered")
    _EVALUATORS[name] = evaluator


def get_evaluator(name: str) -> Evaluator:
    """Return a registered evaluator."""
    try:
        return _EVALUATORS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown evaluator {name!r}") from exc


def registered_evaluators() -> tuple[str, ...]:
    """Return registered evaluator names."""
    return tuple(sorted(_EVALUATORS))
