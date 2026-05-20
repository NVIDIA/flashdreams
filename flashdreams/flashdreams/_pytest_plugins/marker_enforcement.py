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

"""Pytest plugin that enforces CI tier markers on every test.

Every test must carry exactly one of ``ci_cpu``, ``ci_gpu``, or
``manual`` -- either via a module-level ``pytestmark`` or a per-function
decorator.  ``manual`` may be combined with ``ci_cpu`` or ``ci_gpu``
to opt a single function out of a module-level tier (manual takes
precedence).  Having both ``ci_cpu`` and ``ci_gpu`` is rejected.
Tests without any tier marker are rejected.
"""

from __future__ import annotations

import pytest

_CI_TIER_MARKERS = frozenset({"ci_cpu", "ci_gpu", "manual"})


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Reject unmarked tests and ci_cpu + ci_gpu conflicts."""
    missing: list[str] = []
    conflicting: list[str] = []

    for item in items:
        item_markers = {m.name for m in item.iter_markers()} & _CI_TIER_MARKERS

        if not item_markers:
            missing.append(item.nodeid)
        elif {"ci_cpu", "ci_gpu"} <= item_markers:
            conflicting.append(item.nodeid)

    errors: list[str] = []
    if missing:
        formatted = "\n".join(f"  {nodeid}" for nodeid in missing)
        errors.append(
            f"Tests missing a CI tier marker (ci_cpu, ci_gpu, or manual):\n{formatted}"
        )
    if conflicting:
        formatted = "\n".join(f"  {nodeid}" for nodeid in conflicting)
        errors.append(
            f"Tests marked with both ci_cpu and ci_gpu "
            f"(pick exactly one, or use manual to opt out):\n"
            f"{formatted}"
        )

    if errors:
        raise pytest.UsageError("\n\n".join(errors))
