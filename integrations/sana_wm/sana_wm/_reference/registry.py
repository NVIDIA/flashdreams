# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small registry subset used by the vendored SANA-WM inference code."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


class Registry:
    """Minimal registry with the methods used by SANA-WM inference."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.module_dict: dict[str, Any] = {}

    def register_module(
        self,
        name: str | None = None,
        module: T | None = None,
    ) -> Callable[[T], T] | T:
        if module is not None:
            self._register(module, name)
            return module

        def decorator(obj: T) -> T:
            self._register(obj, name)
            return obj

        return decorator

    def get(self, key: str) -> Any:
        return self.module_dict.get(key)

    def build(
        self,
        cfg: dict[str, Any] | str,
        default_args: dict[str, Any] | None = None,
    ) -> Any:
        return build_from_cfg(cfg, self, default_args=default_args)

    def _register(self, obj: Any, name: str | None) -> None:
        key = name or obj.__name__
        if key in self.module_dict and self.module_dict[key] is not obj:
            raise KeyError(f"{key!r} is already registered in {self.name}")
        self.module_dict[key] = obj


def build_from_cfg(
    cfg: dict[str, Any] | str,
    registry: Registry,
    default_args: dict[str, Any] | None = None,
) -> Any:
    args = dict(default_args or {})
    if isinstance(cfg, str):
        obj_type = cfg
    else:
        cfg = dict(cfg)
        obj_type = cfg.pop("type")
        args.update(cfg)

    cls = registry.get(obj_type) if isinstance(obj_type, str) else obj_type
    if cls is None:
        raise KeyError(f"{obj_type!r} is not registered in {registry.name}")
    return cls(**args)
