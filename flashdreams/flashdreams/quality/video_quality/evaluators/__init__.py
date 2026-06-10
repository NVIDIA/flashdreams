# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable video-quality evaluator registry.

Evaluators are opaque to the benchmark runner: each implements the
:class:`~flashdreams.quality.video_quality.evaluators.base.Evaluator` protocol and
is looked up by name from the manifest. Real learned/reference/VLM evaluators can
register here without changing the runner loop.
"""

from __future__ import annotations

from flashdreams.quality.video_quality.evaluators.base import (
    CheckResult,
    Evaluator,
    EvalSample,
)

__all__ = [
    "CheckResult",
    "Evaluator",
    "EvalSample",
    "get_evaluator",
    "register_default_evaluators",
    "register_evaluator",
    "registered_evaluators",
]

_EVALUATORS: dict[str, Evaluator] = {}


def register_evaluator(
    name: str, evaluator: Evaluator, *, replace: bool = False
) -> None:
    """Register an evaluator by name."""
    if not name:
        raise ValueError("Evaluator name must be non-empty")
    if name in _EVALUATORS and not replace:
        raise ValueError(f"Evaluator {name!r} is already registered")
    _EVALUATORS[name] = evaluator


def get_evaluator(name: str) -> Evaluator:
    """Return a registered evaluator."""
    try:
        return _EVALUATORS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown evaluator {name!r}; registered: {registered_evaluators()}"
        ) from exc


def registered_evaluators() -> tuple[str, ...]:
    """Return registered evaluator names."""
    return tuple(sorted(_EVALUATORS))


def register_default_evaluators() -> None:
    """Register built-in placeholder evaluators (idempotent).

    These are skeleton stubs that satisfy the contract so the pipeline runs
    end-to-end; real evaluators replace them in a later pass.
    """
    from flashdreams.quality.video_quality.evaluators.stub import (
        StubPresenceEvaluator,
    )

    for evaluator in (StubPresenceEvaluator(),):
        register_evaluator(evaluator.name, evaluator, replace=True)
