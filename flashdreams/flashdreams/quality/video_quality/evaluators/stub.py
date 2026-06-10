# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Placeholder evaluators for pipe-cleaning the benchmark.

These are deliberately fake: they satisfy the ``Evaluator`` contract and produce
deterministic, varied yes/no answers so the end-to-end plumbing and KPI
aggregation can be exercised before the real VLM/detector evaluators exist.
REPLACE in the evaluator pass.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from flashdreams.quality.video_quality.evaluators.base import CheckResult, EvalSample
from flashdreams.quality.video_quality.manifest import Check


class StubPresenceEvaluator:
    """Deterministic fake 'object presence' evaluator (placeholder).

    Answers are a stable pseudo-random function of (scenario, check, a cheap
    frame signal) so a pipe-clean run yields a realistic mix of correct/incorrect
    results without any real perception. This is NOT a real evaluator.
    """

    name = "stub_presence"

    def evaluate(
        self, sample: EvalSample, checks: Sequence[Check]
    ) -> list[CheckResult]:
        brightness = float(np.mean(sample.frames)) if sample.frames.size else 0.0
        results: list[CheckResult] = []
        for check in checks:
            seed = f"{sample.scenario_id}:{check.id}:{int(brightness)}"
            digest = hashlib.sha256(seed.encode()).digest()[0]
            answer = "yes" if digest % 2 == 0 else "no"
            results.append(
                CheckResult.from_check(
                    check,
                    answer=answer,
                    evaluator=self.name,
                    detail={"placeholder": True, "mean_brightness": brightness},
                )
            )
        return results
