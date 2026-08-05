# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal MMCV compatibility used by the SANA-WM parity harness.

The upstream inference entrypoint only needs registry construction plus a few
utility functions at import time. This shim avoids installing old mmcv builds in
the isolated parity venv.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Callable


class Registry:
    """Small subset of ``mmcv.Registry`` used by SANA model registration."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.module_dict: dict[str, Any] = {}

    def __contains__(self, key: str) -> bool:
        return key in self.module_dict

    def get(self, key: str) -> Any:
        return self.module_dict.get(key)

    def register_module(
        self,
        module: Any | None = None,
        *,
        name: str | None = None,
        force: bool = False,
    ) -> Callable[[Any], Any] | Any:
        """Register ``module`` or return a class decorator."""

        def _register(obj: Any) -> Any:
            key = name or obj.__name__
            if key in self.module_dict and not force:
                raise KeyError(f"{key!r} is already registered in {self.name}.")
            self.module_dict[key] = obj
            return obj

        if module is not None:
            return _register(module)
        return _register

    def build(
        self,
        cfg: dict[str, Any],
        *,
        default_args: dict[str, Any] | None = None,
    ) -> Any:
        return build_from_cfg(cfg, self, default_args=default_args)


class Config(dict):
    """Tiny dict-backed stand-in for import-time ``mmcv.Config`` references."""

    @classmethod
    def fromfile(cls, filename: str | os.PathLike[str]) -> "Config":
        raise NotImplementedError(
            "mmcv.Config.fromfile is not implemented in the parity harness shim."
        )


def build_from_cfg(
    cfg: dict[str, Any],
    registry: Registry,
    *,
    default_args: dict[str, Any] | None = None,
) -> Any:
    """Instantiate a registered object from a config dict."""

    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be a dict, got {type(cfg).__name__}.")
    if "type" not in cfg:
        raise KeyError("cfg must contain key 'type'.")
    args = dict(default_args or {})
    args.update({k: v for k, v in cfg.items() if k != "type"})
    obj_type = cfg["type"]
    if isinstance(obj_type, str):
        obj_cls = registry.get(obj_type)
        if obj_cls is None:
            raise KeyError(f"{obj_type!r} is not registered in {registry.name}.")
    else:
        obj_cls = obj_type
    return obj_cls(**args)


def mkdir_or_exist(path: str | os.PathLike[str]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def dump(obj: Any, file: str | os.PathLike[str]) -> None:
    with open(file, "wb") as handle:
        pickle.dump(obj, handle)


def load(file: str | os.PathLike[str]) -> Any:
    with open(file, "rb") as handle:
        return pickle.load(handle)


__all__ = [
    "Config",
    "Registry",
    "build_from_cfg",
    "dump",
    "load",
    "mkdir_or_exist",
]
