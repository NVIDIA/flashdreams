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

"""Controller-contract compatibility and transport import boundaries."""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.ci_cpu


def test_active_sources_import_without_legacy_runtime_or_servers() -> None:
    """Keep the V2 host path independent of legacy runtime implementations."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib
import sys

prefix = "flashdreams_controller_input."
sys.modules["handdreams"] = None
sys.modules["cosmos_interactive_internal"] = None
for name in ("source", "webxr_server", "isaac_teleop_bridge"):
    sys.modules[prefix + name] = None
for name in (
    "controller_types",
):
    importlib.import_module(prefix + name)
for name in (
    "controller_server",
    "quest_source",
    "isaac_teleop_source",
    "controller_source_v2_window",
    "quest_remote_v2_window",
    "isaac_teleop_v2_window",
):
    importlib.import_module("flashdreams_controller_input." + name)
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
