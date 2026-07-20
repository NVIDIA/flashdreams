# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lifecycle helpers for one-shot encoders used during cache initialization."""

from __future__ import annotations

import gc
import importlib
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any


def setup_one_shot_encoder(
    config: Any,
    *,
    device: Any | Callable[[], Any] | None = None,
    torch_module: Any | None = None,
) -> Any:
    """Instantiate an encoder config and move torch modules to ``device``."""
    encoder = config.setup()
    if device is None:
        return encoder

    torch = torch_module if torch_module is not None else _maybe_import_torch()
    module_cls = getattr(getattr(torch, "nn", None), "Module", None)
    if module_cls is not None and isinstance(encoder, module_cls):
        resolved_device = device() if callable(device) else device
        encoder = encoder.to(device=resolved_device)
    return encoder


def ensure_one_shot_encoder(
    encoder: Any | None,
    config: Any | None,
    *,
    device: Any | Callable[[], Any] | None = None,
    name: str = "encoder",
    required: bool = True,
    torch_module: Any | None = None,
) -> Any | None:
    """Return ``encoder`` if loaded, otherwise instantiate it from ``config``."""
    if encoder is not None:
        return encoder
    if config is None:
        if required:
            raise RuntimeError(
                f"{name} is not loaded and no config is available to reload it."
            )
        return None
    return setup_one_shot_encoder(
        config,
        device=device,
        torch_module=torch_module,
    )


def release_one_shot_encoder_references(
    owner: Any,
    *attrs: str,
    device: Any | None = None,
    synchronize_cuda: bool = False,
    empty_cuda_cache: bool = True,
    torch_module: Any | None = None,
) -> tuple[str, ...]:
    """Set encoder attributes to ``None`` and release reclaimed CUDA memory.

    Returns the names that previously held non-``None`` values. Attributes are
    set to ``None`` even when they were already empty so later access sees a
    useful unloaded state instead of a missing attribute.
    """
    released: list[str] = []
    for attr in attrs:
        if getattr(owner, attr, None) is not None:
            released.append(attr)
        setattr(owner, attr, None)

    collect_and_release_cuda_memory(
        device=device,
        synchronize_cuda=synchronize_cuda,
        empty_cuda_cache=empty_cuda_cache,
        torch_module=torch_module,
    )
    return tuple(released)


def collect_and_release_cuda_memory(
    *,
    device: Any | None = None,
    synchronize_cuda: bool = False,
    empty_cuda_cache: bool = True,
    torch_module: Any | None = None,
) -> None:
    """Run GC and optionally return freed CUDA blocks to the caching allocator."""
    gc.collect()
    if not empty_cuda_cache and not synchronize_cuda:
        return

    torch = torch_module if torch_module is not None else _maybe_import_torch()
    cuda = getattr(torch, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if cuda is None or not callable(is_available) or not is_available():
        return

    if synchronize_cuda:
        synchronize = getattr(cuda, "synchronize", None)
        if callable(synchronize):
            if device is None:
                synchronize()
            else:
                synchronize(device=device)
    if empty_cuda_cache:
        empty_cache = getattr(cuda, "empty_cache", None)
        if callable(empty_cache):
            empty_cache()


def move_tensors_to_cpu(value: Any, *, torch_module: Any | None = None) -> Any:
    """Recursively move torch tensors in ``value`` to CPU."""
    torch = torch_module if torch_module is not None else _maybe_import_torch()
    is_tensor = getattr(torch, "is_tensor", None)
    if callable(is_tensor) and is_tensor(value):
        return value.cpu()
    if isinstance(value, dict):
        return {
            key: move_tensors_to_cpu(item, torch_module=torch)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [move_tensors_to_cpu(item, torch_module=torch) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors_to_cpu(item, torch_module=torch) for item in value)
    return value


def run_one_shot_encoder_stage(
    stage: Callable[[], Any],
    *,
    release: Callable[[], Any] | None = None,
    cpu_result: bool = True,
    torch_module: Any | None = None,
) -> Any:
    """Run an encoder-only stage under ``no_grad`` and release encoders after it."""
    torch = torch_module if torch_module is not None else _maybe_import_torch()
    no_grad = getattr(torch, "no_grad", None)
    context = no_grad() if callable(no_grad) else nullcontext()
    try:
        with context:
            result = stage()
        if cpu_result:
            result = move_tensors_to_cpu(result, torch_module=torch)
        return result
    finally:
        if release is not None:
            release_result = release()
            del release_result


def _maybe_import_torch() -> Any | None:
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None
