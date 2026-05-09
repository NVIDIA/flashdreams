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

"""Specification dataclass external packages use to register runners."""

from __future__ import annotations

from dataclasses import dataclass

from flashdreams.infra.runner import RunnerConfig


@dataclass
class RunnerSpecification:
    """Runner specification used to register an end-to-end driver out-of-tree.

    Mirrors :class:`nerfstudio.plugins.types.MethodSpecification`.
    External Python packages expose one ``RunnerSpecification`` per
    shipped variant via the ``flashdreams.runner_configs`` entry-point
    group; the discovery layer in :mod:`flashdreams.plugins.registry`
    loads them and merges them into the central
    ``BUILTIN_RUNNERS`` dict the ``flashdreams-run`` CLI dispatches over.

    The registry key is ``config.runner_name`` (not the entry-point
    name), so the same slug works whether a runner ships in-tree, via
    an entry point, or via the ``FLASHDREAMS_RUNNER_CONFIGS`` env-var
    backdoor.
    """

    config: RunnerConfig
    """Fully-resolved runner config (carrying its wrapped pipeline
    config). ``config.runner_name`` is the registry key and the
    ``flashdreams-run`` subcommand name."""

    description: str
    """One-line help string shown next to the subcommand in
    ``flashdreams-run --help``."""
