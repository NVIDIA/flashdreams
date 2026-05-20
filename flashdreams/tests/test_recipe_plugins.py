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

from flashdreams.configs.registry import register_runner, supported_runners
from flashdreams.configs.runner_configs import (
    _annotated_base_runner_union,
    all_runners,
)
from flashdreams.infra.config import derive_config
from flashdreams.infra.runner import RunnerConfig
from flashdreams.plugins import discover_runners
from flashdreams.plugins.registry import ENV_VAR
from flashdreams.recipes.template.config import TEMPLATE_OFFLINE_RUNNER

pytestmark = pytest.mark.ci_cpu


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


def test_discover_runners_skips_runner_name_collision(
    monkeypatch: pytest.MonkeyPatch, fake_plugin_module: str
) -> None:
    """Two env-var entries claiming the same ``runner_name`` -> first wins."""
    module = sys.modules[fake_plugin_module]
    setattr(
        module,
        "DUP_FIRST",
        derive_config(
            TEMPLATE_OFFLINE_RUNNER,
            runner_name="dup-collision-slug",
            description="first",
        ),
    )
    setattr(
        module,
        "DUP_SECOND",
        derive_config(
            TEMPLATE_OFFLINE_RUNNER,
            runner_name="dup-collision-slug",
            description="second",
        ),
    )
    monkeypatch.setenv(
        ENV_VAR,
        (
            f"first={fake_plugin_module}:DUP_FIRST,"
            f"second={fake_plugin_module}:DUP_SECOND"
        ),
    )
    runners = discover_runners()
    assert runners["dup-collision-slug"].description == "first"


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
    assert runners["template-offline"] is supported_runners()["template-offline"]


def test_all_runners_returns_sorted_view() -> None:
    """``all_runners`` returns a deterministic, alphabetically sorted view."""
    runners = all_runners()
    assert list(runners) == sorted(runners)
    # Every built-in is present (regardless of plugin discovery).
    for slug in supported_runners():
        assert slug in runners


def test_register_runner_builtin_collision_raises() -> None:
    """``source="builtin"`` with a duplicate slug is a programmer bug."""
    base_cfg = derive_config(TEMPLATE_OFFLINE_RUNNER, runner_name="reg-base")
    dup_cfg = derive_config(TEMPLATE_OFFLINE_RUNNER, runner_name="reg-base")
    target: dict[str, RunnerConfig] = {"reg-base": base_cfg}
    with pytest.raises(ValueError, match="Duplicate built-in runner_name"):
        register_runner("reg-base", dup_cfg, source="builtin", target=target)
    assert target["reg-base"] is base_cfg


def test_register_runner_plugin_collision_skips() -> None:
    """``source="plugin"`` skips slugs already in the target (in-tree wins)."""
    base_cfg = derive_config(TEMPLATE_OFFLINE_RUNNER, runner_name="reg-base")
    plugin_cfg = derive_config(TEMPLATE_OFFLINE_RUNNER, runner_name="reg-base")
    target: dict[str, RunnerConfig] = {"reg-base": base_cfg}
    register_runner("reg-base", plugin_cfg, source="plugin", target=target)
    assert target["reg-base"] is base_cfg


def test_register_runner_inserts_new_slugs() -> None:
    """Both sources insert non-conflicting slugs into ``target``."""
    a = derive_config(TEMPLATE_OFFLINE_RUNNER, runner_name="reg-new-a")
    b = derive_config(TEMPLATE_OFFLINE_RUNNER, runner_name="reg-new-b")
    target: dict[str, RunnerConfig] = {}
    register_runner("reg-new-a", a, source="builtin", target=target)
    register_runner("reg-new-b", b, source="plugin", target=target)
    assert target["reg-new-a"] is a
    assert target["reg-new-b"] is b


def test_annotated_base_runner_union_builds() -> None:
    """The tyro union factory must succeed -- a regression here breaks ``flashdreams-run``."""
    union = _annotated_base_runner_union()
    assert union is not None
