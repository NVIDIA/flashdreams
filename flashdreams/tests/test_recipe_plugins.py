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

"""Tests for the runner-config plugin layer.

Covers the two discovery seams the CLI relies on -- the entry-point
group and the ``FLASHDREAMS_RUNNER_CONFIGS`` env-var backdoor -- and
checks the precedence the loader documents (in-tree wins over plugin).
The CLI factory is also exercised end-to-end so a tyro-API regression
shows up here, not at the user's terminal.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator

import pytest

from flashdreams.configs.runner_configs import (
    BUILTIN_RUNNERS,
    _annotated_base_runner_union,
    all_runners,
    merge_runners,
)
from flashdreams.infra.config import derive_config
from flashdreams.infra.runner import RunnerConfig
from flashdreams.plugins import discover_runners
from flashdreams.plugins.registry import ENV_VAR
from flashdreams.recipes.template.runner import TEMPLATE_OFFLINE_RUNNER


@pytest.fixture
def fake_plugin_module() -> Iterator[str]:
    """Install a throwaway ``flashdreams._test_plugin`` module on ``sys.modules``.

    The env-var backdoor resolves the runner via
    ``importlib.import_module``, so the config object has to be reachable
    by import. Bolting it onto ``sys.modules`` is the lightest-weight
    way to give the resolver something to find without writing a file
    to ``site-packages``.
    """
    module_name = "flashdreams._test_plugin"
    module = types.ModuleType(module_name)
    setattr(
        module,
        "PLUGIN_CONFIG",
        derive_config(
            TEMPLATE_OFFLINE_RUNNER,
            runner_name="test-plugin-offline",
            description="Test plugin runner (env-var registration).",
        ),
    )

    def _factory() -> RunnerConfig:
        return derive_config(
            TEMPLATE_OFFLINE_RUNNER,
            runner_name="test-plugin-factory-offline",
            description="Test plugin runner (factory-style env-var registration).",
        )

    setattr(module, "PLUGIN_FACTORY", _factory)
    sys.modules[module_name] = module
    try:
        yield module_name
    finally:
        sys.modules.pop(module_name, None)


def test_runner_config_carries_description() -> None:
    """``RunnerConfig.description`` is the public CLI-description surface."""
    cfg = derive_config(
        TEMPLATE_OFFLINE_RUNNER,
        runner_name="description-shape-check",
        description="shape check",
    )
    assert cfg.runner_name == "description-shape-check"
    assert cfg.description == "shape check"


def test_env_var_backdoor_loads_config(
    monkeypatch: pytest.MonkeyPatch, fake_plugin_module: str
) -> None:
    """``FLASHDREAMS_RUNNER_CONFIGS=slug=mod:attr`` registers the config."""
    monkeypatch.setenv(
        ENV_VAR,
        f"test-plugin-offline={fake_plugin_module}:PLUGIN_CONFIG",
    )
    runners = discover_runners()
    assert "test-plugin-offline" in runners
    assert runners["test-plugin-offline"].runner_name == "test-plugin-offline"
    assert runners["test-plugin-offline"].description.startswith("Test plugin runner")


def test_env_var_backdoor_accepts_factory(
    monkeypatch: pytest.MonkeyPatch, fake_plugin_module: str
) -> None:
    """The backdoor invokes a callable to obtain the config (matches ns)."""
    monkeypatch.setenv(
        ENV_VAR,
        f"test-plugin-factory-offline={fake_plugin_module}:PLUGIN_FACTORY",
    )
    runners = discover_runners()
    assert "test-plugin-factory-offline" in runners


def test_env_var_backdoor_skips_bad_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed env-var entry must not break the discovery call."""
    monkeypatch.setenv(ENV_VAR, "broken-entry-without-equals-sign")
    runners = discover_runners()
    # The bad entry is logged-and-skipped; the good ones (entry-points,
    # if any are installed) are still returned.
    assert isinstance(runners, dict)


def test_all_runners_does_not_let_plugins_shadow_builtins(
    monkeypatch: pytest.MonkeyPatch, fake_plugin_module: str
) -> None:
    """Built-in runners win over a same-slug plugin (overwrite=False)."""
    # Register a plugin that *claims* to be the in-tree template-offline
    # runner but with a different description; the loader must keep the
    # built-in version intact.
    module = sys.modules[fake_plugin_module]
    setattr(
        module,
        "SHADOW_CONFIG",
        derive_config(
            TEMPLATE_OFFLINE_RUNNER,
            runner_name="template-offline",
            description="SHOULD-BE-IGNORED: plugin tried to shadow a built-in.",
        ),
    )
    monkeypatch.setenv(
        ENV_VAR,
        f"template-offline={fake_plugin_module}:SHADOW_CONFIG",
    )
    runners = all_runners()
    assert runners["template-offline"] is BUILTIN_RUNNERS["template-offline"]


def test_all_runners_returns_sorted_view() -> None:
    """``all_runners`` returns a deterministic, alphabetically sorted view."""
    runners = all_runners()
    assert list(runners) == sorted(runners)
    # Every built-in is present (regardless of plugin discovery).
    for slug in BUILTIN_RUNNERS:
        assert slug in runners


def test_merge_runners_overwrite_semantics() -> None:
    """``merge_runners`` honours the ``overwrite`` flag on key collisions."""
    base_cfg = derive_config(TEMPLATE_OFFLINE_RUNNER, runner_name="merge-base")
    override_cfg = derive_config(TEMPLATE_OFFLINE_RUNNER, runner_name="merge-base")
    out = merge_runners(
        {"merge-base": base_cfg},
        {"merge-base": override_cfg},
        overwrite=True,
    )
    assert out["merge-base"] is override_cfg

    out = merge_runners(
        {"merge-base": base_cfg},
        {"merge-base": override_cfg},
        overwrite=False,
    )
    assert out["merge-base"] is base_cfg


def test_annotated_base_runner_union_builds() -> None:
    """The tyro union factory must succeed -- a regression here breaks ``flashdreams-run``."""
    union = _annotated_base_runner_union()
    assert union is not None
