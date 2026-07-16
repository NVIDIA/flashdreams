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

"""Serving-model configuration and plugin discovery."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import cast

from flashdreams.serving.api import ModelDescriptor
from flashdreams.serving.backend import WorkerFactory

ENTRY_POINT_GROUP = "flashdreams.serve_configs"
"""Entry-point group containing :class:`ServeModelConfig` instances."""

ENV_VAR = "FLASHDREAMS_SERVE_CONFIGS"
"""Comma-separated ``slug=module:attribute`` development registrations."""


@dataclass(frozen=True, slots=True)
class ServeModelConfig:
    """Bind a public model descriptor to its lazy worker factory."""

    descriptor: ModelDescriptor
    """Public discovery and placement description."""

    worker_factory: WorkerFactory
    """Factory that constructs an unloaded model worker."""


def discover_serve_configs() -> dict[str, ServeModelConfig]:
    """Discover installed and development serving model configurations."""
    configs: dict[str, ServeModelConfig] = {}
    for entry_point in sorted(
        entry_points(group=ENTRY_POINT_GROUP), key=lambda value: value.name
    ):
        _accept_config(configs, entry_point.load(), entry_point.value)
    raw = os.environ.get(ENV_VAR, "")
    for definition in filter(None, (item.strip() for item in raw.split(","))):
        _slug, path = definition.split("=", 1)
        module_name, attribute = path.split(":", 1)
        value = getattr(importlib.import_module(module_name), attribute)
        _accept_config(configs, value, path)
    return configs


def _accept_config(
    configs: dict[str, ServeModelConfig], value: object, origin: str
) -> None:
    if callable(value) and not isinstance(value, ServeModelConfig):
        value = cast(Callable[[], object], value)()
    if not isinstance(value, ServeModelConfig):
        raise TypeError(
            f"Serving config {origin!r} returned {type(value).__name__}, "
            "not ServeModelConfig."
        )
    slug = value.descriptor.id
    if slug in configs:
        raise ValueError(f"Duplicate serving model slug {slug!r} from {origin!r}.")
    configs[slug] = value
