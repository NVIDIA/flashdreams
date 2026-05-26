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

"""Process-global runner registry storage and the single registration primitive.

This module is a leaf: it has no integration / plugin imports, so any code
path (in-tree integration ``runner.py``, the plugin layer, tests) can import
:func:`register_runner` and the underlying :data:`_SUPPORTED_RUNNERS`
dict without circular-import risk.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from loguru import logger

from flashdreams.infra.runner import RunnerConfig

RunnerSource = Literal["builtin", "plugin"]
"""Where a runner came from. Drives the collision policy in
:func:`register_runner`."""


_SUPPORTED_RUNNERS: dict[str, RunnerConfig] = {}
"""In-tree runner registry. Populated at integration-runner import time via
:func:`register_runner` calls (``source="builtin"``).

Treat as immutable after all in-tree integration modules have been imported.
:func:`flashdreams.configs.runner_configs.all_runners` layers plugin
discoveries on top in a per-call local dict rather than mutating this
global, so test isolation is automatic and multiple ``all_runners()``
invocations see deterministic builtin state.
"""


def register_runner(
    name: str,
    runner: RunnerConfig,
    *,
    source: RunnerSource,
    target: dict[str, RunnerConfig] | None = None,
) -> None:
    """Register one runner under ``name`` with source-aware collision policy.

    Args:
        name: Registry slug (typically ``runner.runner_name``).
        runner: Runner config to register.
        source: ``"builtin"`` → in-tree integration; collisions raise
            (programmer bug, fail at import time). ``"plugin"`` →
            third-party (entry point or env-var); collisions are
            logged and skipped, so a plugin can never silently shadow
            an existing slug.
        target: Destination dict. Defaults to the process-global
            :data:`_SUPPORTED_RUNNERS` (used by in-tree integration
            ``runner.py`` modules at import time). Pass an explicit
            dict to register against a per-call snapshot
            (used by :func:`flashdreams.configs.runner_configs.all_runners`
            for the plugin layer).

    Raises:
        ValueError: ``source="builtin"`` and ``name`` is already in
            ``target``.
    """
    if target is None:
        target = _SUPPORTED_RUNNERS
    if name in target:
        if source == "builtin":
            existing = type(target[name]).__name__
            new = type(runner).__name__
            raise ValueError(
                f"Duplicate built-in runner_name {name!r}: already "
                f"registered as {existing}, new entry is {new}."
            )
        logger.warning(
            f"Skipping plugin runner {name!r}: already registered "
            f"(in-tree or earlier-registered plugin wins)."
        )
        return
    target[name] = runner


def supported_runners() -> Mapping[str, RunnerConfig]:
    """Return a shallow snapshot of the in-tree runner registry.

    The returned dict is a copy; mutating it does not affect the
    underlying registry.
    """
    return dict(_SUPPORTED_RUNNERS)
