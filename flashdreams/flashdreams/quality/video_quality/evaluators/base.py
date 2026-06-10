# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime contract for video-quality benchmark evaluators (the black-box seam).

An evaluator receives an :class:`EvalSample` (a generated clip plus its
conditioning) and the subset of manifest
:class:`~flashdreams.quality.video_quality.manifest.Check` items addressed to it,
and returns one :class:`CheckResult` per check. The runner does not care *how* an
evaluator decides -- a cheap heuristic, an object detector, or a hosted VLM all
satisfy this same interface -- so real evaluators can be dropped in later without
touching the pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from flashdreams.quality.video_quality.manifest import Check

RGBVideo = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class EvalSample:
    """One generated clip plus everything an evaluator may inspect."""

    scenario_id: str
    frames: RGBVideo
    fps: float | None = None
    reference_frames: RGBVideo | None = None
    conditioning: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckResult:
    """The outcome of one yes/no check against a generated clip."""

    check_id: str
    category: str
    question: str
    expected: str
    answer: str
    correct: bool
    control_channel: str
    tier: str
    evaluator: str
    detail: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_check(
        cls,
        check: Check,
        *,
        answer: str,
        evaluator: str,
        detail: dict[str, Any] | None = None,
    ) -> CheckResult:
        """Build a result for ``check`` from an evaluator's raw ``answer``."""
        normalized = answer.strip().lower()
        return cls(
            check_id=check.id,
            category=check.category,
            question=check.question,
            expected=check.expected,
            answer=normalized,
            correct=normalized == check.expected.strip().lower(),
            control_channel=check.control_channel,
            tier=check.tier,
            evaluator=evaluator,
            detail=dict(detail or {}),
        )


@runtime_checkable
class Evaluator(Protocol):
    """Pluggable, opaque evaluator. Real impls (VLM/detector) land next week."""

    name: str

    def evaluate(
        self, sample: EvalSample, checks: Sequence[Check]
    ) -> list[CheckResult]:
        """Answer ``checks`` for ``sample``; return one result per check."""
        ...
