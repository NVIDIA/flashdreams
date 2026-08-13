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

"""Discover FlashDreams plugins registered via entry points."""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from collections.abc import Callable
from functools import lru_cache
from typing import TYPE_CHECKING, Any, TypeVar, cast

from loguru import logger

from flashdreams.infra.postprocess import VideoPostProcessorConfig
from flashdreams.infra.runner import RunnerConfig

if TYPE_CHECKING:
    from flashdreams.demo.application import Application

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points  # type: ignore[import-not-found]
else:
    from importlib.metadata import entry_points

ENTRY_POINT_GROUP = "flashdreams.runner_configs"
"""Entry-point group external packages register :class:`RunnerConfig`
instances under (matches nerfstudio's ``nerfstudio.method_configs``
naming)."""

POSTPROCESS_PRESET_GROUP = "flashdreams.postprocess_presets"
"""Setuptools entry-point group for video post-processor presets.

Each entry maps a preset slug (``ep.name``) to a
:class:`~flashdreams.infra.postprocess.VideoPostProcessorConfig`
instance (or a zero-arg factory returning one). Slugs are resolved by
:func:`discover_postprocess_presets` and selected via
``RunnerConfig.postprocess.preset`` / ``--postprocess.preset``."""

ENV_VAR = "FLASHDREAMS_RUNNER_CONFIGS"
"""Env-var backdoor for in-development runners that aren't installed yet.

Format: ``slug=module.path:attribute,slug2=other.module:attr``. The
attribute is loaded with ``getattr(import_module(module), attr)``; if
it is callable (and not already a :class:`RunnerConfig`) it is invoked
with no arguments to obtain the config. The ``slug=`` prefix is purely
for log readability -- the registry key always comes from
``cfg.runner_name``."""


APPLICATION_ENTRY_POINT_GROUP = "flashdreams.applications"
"""Setuptools entry-point group for public demo applications."""

PluginT = TypeVar("PluginT")


def load_plugins(group: str, expected_type: type[PluginT]) -> dict[str, PluginT]:
    """Load entry-point plugins and keep only values of ``expected_type``.

    Entry points may expose either a ready object or a zero-argument factory
    returning one. Bad plugins are logged and skipped so a partial environment
    does not break unrelated commands.
    """
    plugins: dict[str, PluginT] = {}
    for name, origin, value in _load_entry_point_plugins(group, expected_type):
        if name in plugins:
            logger.warning(f"Skipping duplicate {group} plugin {origin}.")
            continue
        plugins[name] = value
    return plugins


def discover_runners() -> dict[str, RunnerConfig]:
    """Discover externally-registered runner configs.

    Looks at every entry point under :data:`ENTRY_POINT_GROUP` and at
    the ``slug=module:attr`` pairs in :data:`ENV_VAR`. Bad entries are
    logged and skipped -- the CLI must keep working even when a third-
    party plugin is broken.

    Each loaded value must be a :class:`RunnerConfig` (or a zero-arg
    factory returning one); the subcommand description is read off
    ``cfg.description``.

    Resolution order:

    1. Entry points sorted by ``ep.name`` so the winner of a collision
       is deterministic across installs.
    2. ``FLASHDREAMS_RUNNER_CONFIGS`` env-var entries, in declared
       order.

    On a ``runner_name`` collision the *first* seen config wins; the
    later one is logged and skipped, including the ``module:attr``
    origin of both configs so the plugin author can find and rename the
    duplicate.

    Returns:
        A dict keyed by ``cfg.runner_name``.
    """
    runners: dict[str, RunnerConfig] = {}
    # Tracks the ``module:attr`` (entry-point) or ``slug=module:attr``
    # (env-var) origin of each accepted runner so we can name *both*
    # sides of a collision in the warning message.
    origins: dict[str, str] = {}

    def _accept(cfg: RunnerConfig, origin: str) -> None:
        """Insert ``cfg`` unless its ``runner_name`` is already taken."""
        existing = origins.get(cfg.runner_name)
        if existing is not None:
            logger.warning(
                f"Skipping runner {cfg.runner_name!r} from {origin}: "
                f"slug already registered by {existing}. Rename one of "
                f"the two configs to disambiguate."
            )
            return
        runners[cfg.runner_name] = cfg
        origins[cfg.runner_name] = origin

    for _name, origin, cfg in _load_entry_point_plugins(
        ENTRY_POINT_GROUP,
        RunnerConfig,
    ):
        _accept(cfg, origin)

    raw = os.environ.get(ENV_VAR)
    if raw:
        for definition in raw.split(","):
            definition = definition.strip()
            if not definition:
                continue
            origin = f"{ENV_VAR} entry {definition!r}"
            try:
                slug, path = definition.split("=", 1)
                module_name, attr = path.split(":", 1)
                logger.info(
                    f"Loading runner {slug!r} from {module_name}:{attr} ({ENV_VAR})"
                )
                attr_value = getattr(importlib.import_module(module_name), attr)
                if callable(attr_value) and not isinstance(attr_value, RunnerConfig):
                    attr_value = attr_value()
                if not isinstance(attr_value, RunnerConfig):
                    raise TypeError(
                        f"{module_name}:{attr} is not a RunnerConfig "
                        f"(got {type(attr_value).__name__})."
                    )
                _accept(attr_value, origin)
            except Exception:  # noqa: BLE001
                logger.warning(
                    f"Failed to load runner from {origin}:\n{traceback.format_exc()}"
                )

    return runners


def discover_applications() -> dict[str, "Application"]:
    """Discover public demo applications registered by installed packages."""
    from flashdreams.demo.application import Application

    return load_plugins(APPLICATION_ENTRY_POINT_GROUP, Application)


def discover_postprocess_presets() -> dict[str, VideoPostProcessorConfig]:
    """Discover named post-processor presets from entry points.

    Returns:
        Mapping from preset slug to :class:`VideoPostProcessorConfig`.
    """
    return load_plugins(POSTPROCESS_PRESET_GROUP, VideoPostProcessorConfig)


def _load_entry_point_plugin(
    ep: Any,
    *,
    group: str,
    origin: str,
    expected_type: type[PluginT],
) -> PluginT | None:
    module_name = ep.value.split(":", 1)[0]
    top_level_module = module_name.split(".", 1)[0]
    try:
        value = ep.load()
    except ModuleNotFoundError as exc:
        missing_name = exc.name or ""
        if (
            missing_name == top_level_module
            or missing_name == module_name
            or missing_name.startswith(f"{top_level_module}.")
        ):
            logger.debug(
                f"Skipping unavailable {group} plugin {origin}: "
                f"module {missing_name!r} is not installed in this environment."
            )
            return None
        logger.debug(
            f"Failed to load {group} plugin {origin}:\n{traceback.format_exc()}"
        )
        return None
    except Exception:  # noqa: BLE001 - keep discovery alive on bad plugins
        logger.debug(
            f"Failed to load {group} plugin {origin}:\n{traceback.format_exc()}"
        )
        return None
    if callable(value) and (
        isinstance(value, type) or not isinstance(value, expected_type)
    ):
        value = _call_plugin_factory(value, group=group, origin=origin)
        if value is None:
            return None
    if not isinstance(value, expected_type):
        logger.warning(
            f"Skipping {group} plugin {origin}: expected a "
            f"{expected_type.__name__}, got {type(value).__name__}."
        )
        return None
    return cast(PluginT, value)


def _load_entry_point_plugins(
    group: str,
    expected_type: type[PluginT],
) -> list[tuple[str, str, PluginT]]:
    discovered = sorted(entry_points(group=group), key=lambda ep: ep.name)
    plugins: list[tuple[str, str, PluginT]] = []
    for ep in discovered:
        origin = f"entry point {ep.name!r} -> {ep.value}"
        value = _load_entry_point_plugin(
            ep,
            group=group,
            origin=origin,
            expected_type=expected_type,
        )
        if value is not None:
            plugins.append((ep.name, origin, value))
    return plugins


def _call_plugin_factory(
    factory: Callable[[], object],
    *,
    group: str,
    origin: str,
) -> object | None:
    try:
        return factory()
    except Exception:  # noqa: BLE001
        logger.warning(
            f"Calling {group} plugin {origin} as a factory raised:"
            f"\n{traceback.format_exc()}"
        )
        return None


@lru_cache(maxsize=None)
def resolve_postprocess_preset(name: str) -> VideoPostProcessorConfig:
    """Resolve one registered post-processor preset by slug.

    Args:
        name: Preset slug registered under
            :data:`POSTPROCESS_PRESET_GROUP`.

    Raises:
        ValueError: ``name`` is not registered.
    """
    presets = discover_postprocess_presets()
    try:
        return presets[name]
    except KeyError as exc:
        available = ", ".join(sorted(presets)) or "(none registered)"
        raise ValueError(
            f"Unknown postprocess preset {name!r}. Available presets: {available}"
        ) from exc
