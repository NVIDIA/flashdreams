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

"""Explicit CloudXR EULA acceptance across the sidecar boundary."""

import sys
from types import ModuleType
from typing import Any
from unittest.mock import Mock, sentinel

import pytest
from flashdreams_controller_input import isaac_teleop_bridge
from flashdreams_controller_input.isaac_teleop_source import IsaacTeleopSource

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize(
    "options",
    [{}, {"accept_eula": False}, {"accept_eula": True}],
    ids=["omitted", "false", "true"],
)
def test_eula_acceptance_reaches_sdk_without_becoming_implicit(
    monkeypatch: pytest.MonkeyPatch, options: dict[str, Any]
) -> None:
    """Preserve the caller's acceptance through the real sidecar command and parser."""
    launcher = Mock(return_value=sentinel.launcher)
    package = ModuleType("isaacteleop")
    cloudxr = ModuleType("isaacteleop.cloudxr")
    monkeypatch.setattr(package, "cloudxr", cloudxr, raising=False)
    monkeypatch.setattr(cloudxr, "CloudXRLauncher", launcher, raising=False)
    monkeypatch.setitem(sys.modules, "isaacteleop", package)
    monkeypatch.setitem(sys.modules, "isaacteleop.cloudxr", cloudxr)
    monkeypatch.setattr(
        isaac_teleop_bridge, "_prepare_cloudxr_64k_preload", lambda: None
    )
    monkeypatch.delenv("CONTROL_TOKEN", raising=False)

    source = IsaacTeleopSource(**options)
    command = source._bridge_command(python=sys.executable, port=8091)
    expected = options.get("accept_eula", False)
    assert ("--accept-eula" in command) is expected
    args = isaac_teleop_bridge.build_parser().parse_args(command[2:])
    assert args.accept_eula is expected

    if not expected:
        with pytest.raises(ValueError, match="requires --accept-eula"):
            isaac_teleop_bridge._cloudxr_context(args)
        launcher.assert_not_called()
        return
    assert isaac_teleop_bridge._cloudxr_context(args) is sentinel.launcher

    launcher.assert_called_once()
    assert launcher.call_args.kwargs["accept_eula"] is expected
