# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for application commands before v2 window setup."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from flashdreams.api_v2.application import IApplication
from flashdreams.runtime_v2 import cli

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize(
    "flag", ("-h", "--help", "--export-test", "--export-test=path")
)
def test_application_commands_skip_window_setup(monkeypatch, flag: str) -> None:
    app = SimpleNamespace(
        commandline_only_flags=IApplication.commandline_only_flags | {"--export-test"},
        init=Mock(),
    )
    mode = Mock()
    monkeypatch.setattr(cli, "create_application", lambda slug: app)
    monkeypatch.setattr(cli, "client_window_mode", lambda name: mode)

    cli.entrypoint(["test-app", "--", flag])

    app.init.assert_called_once_with([flag])
    mode.check_arguments.assert_not_called()
    mode.create.assert_not_called()


def test_normal_arguments_still_require_a_valid_window(monkeypatch) -> None:
    app = SimpleNamespace(commandline_only_flags=IApplication.commandline_only_flags)
    mode = Mock()
    mode.check_arguments.side_effect = ValueError("missing output path")
    monkeypatch.setattr(cli, "create_application", lambda slug: app)
    monkeypatch.setattr(cli, "client_window_mode", lambda name: mode)

    with pytest.raises(SystemExit) as exc:
        cli.entrypoint(["test-app", "--", "--game-option"])

    assert exc.value.code == 2
    mode.check_arguments.assert_called_once()
    mode.create.assert_not_called()
