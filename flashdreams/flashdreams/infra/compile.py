# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``torch.compile`` helper that preserves the wrapped module's static type."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

import torch
import torch.nn as nn

M = TypeVar("M", bound=nn.Module)

_INDUCTOR_CACHE_SUBDIR = "torchinductor"
"""FlashDreams cache subdirectory for coupled FX-graph and Triton artifacts."""

_TRITON_BUNDLE_PATCH_MARKER = "_flashdreams_complete_static_bundles"
"""Class marker preventing duplicate installation of the Triton bundle repair."""

CompileMode = Literal[
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]
"""Valid ``mode`` values accepted by ``torch.compile``.

- ``default``: balanced; no Inductor autotune.
- ``reduce-overhead``: Inductor with CUDA-graph capture; lower per-call overhead.
- ``max-autotune``: full Inductor autotune + CUDA graphs.
- ``max-autotune-no-cudagraphs``: full Inductor autotune, skip CUDA graphs
  (use this when the caller wraps the result in its own
  :class:`flashdreams.infra.cuda_graph.CUDAGraphWrapper`).
"""


def _configure_dynamo_cache() -> None:
    """Apply explicit FlashDreams overrides to Dynamo's cache limits."""
    from torch._dynamo import config

    for name in ("recompile_limit", "accumulated_recompile_limit"):
        env_name = f"FLASHDREAMS_{name.upper()}"
        value = os.environ.get(env_name)
        if value is not None:
            try:
                limit = int(value)
            except ValueError:
                raise ValueError(f"{env_name} must be a nonnegative integer") from None
            if limit < 0:
                raise ValueError(f"{env_name} must be a nonnegative integer")
            setattr(config, name, limit)


def _configure_inductor_cache() -> None:
    """Place Inductor artifacts in the persistent FlashDreams cache by default."""
    from torch._inductor.runtime.cache_dir_utils import default_cache_dir

    cache_root = Path(
        os.path.expanduser(
            os.environ.get("FLASHDREAMS_CACHE_DIR", "~/.cache/flashdreams")
        )
    )
    configured_cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if configured_cache is not None and configured_cache != default_cache_dir():
        return

    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache_root / _INDUCTOR_CACHE_SUBDIR)


def _add_static_autotuner_hashes_to_winners(
    bundler: Any,
    *,
    to_path_key: Callable[[str], str],
) -> None:
    """Retain every cubin referenced by serialized static autotuners.

    PyTorch's Triton bundler filters artifacts whenever any kernel records an
    autotuning winner. The serialized static autotuners still retain all their
    compile results, so filtering those cubins produces an internally incomplete
    FX-graph cache entry. Add their hashes to the retained set before collection.

    Args:
        bundler: Active private ``TritonBundler`` class.
        to_path_key: Convert a raw Triton hash to its cache-directory key.
    """
    winners = getattr(bundler, "_winners", None)
    static_autotuners = getattr(bundler, "_static_autotuners", None)
    if not winners or not static_autotuners:
        return

    for entry in static_autotuners:
        for result in getattr(entry.kernel, "compile_results", ()):
            kernel_hash = getattr(getattr(result, "kernel", None), "hash", None)
            if isinstance(kernel_hash, str):
                winners.add(to_path_key(kernel_hash))


def _patch_triton_bundle_collection() -> None:
    """Make PyTorch FX-graph bundles self-contained for static Triton launchers."""
    try:
        from torch._inductor.runtime.triton_heuristics import (
            triton_hash_to_path_key,
        )
        from torch._inductor.triton_bundler import TritonBundler
    except ImportError:
        return

    if getattr(TritonBundler, _TRITON_BUNDLE_PATCH_MARKER, False):
        return

    original_collect = getattr(TritonBundler, "collect", None)
    if original_collect is None:
        return

    @classmethod
    def collect_with_complete_static_bundles(
        cls: Any, *args: Any, **kwargs: Any
    ) -> Any:
        _add_static_autotuner_hashes_to_winners(
            cls,
            to_path_key=triton_hash_to_path_key,
        )
        return original_collect(*args, **kwargs)

    setattr(TritonBundler, "collect", collect_with_complete_static_bundles)
    setattr(TritonBundler, _TRITON_BUNDLE_PATCH_MARKER, True)


def compile_module(
    module: M,
    *,
    mode: CompileMode = "max-autotune-no-cudagraphs",
    dynamic: bool | None = None,
) -> M:
    """``torch.compile`` returning the same static type as ``module``.

    ``torch.compile`` wraps a module in an ``OptimizedModule`` proxy whose
    forward signature mirrors the wrapped module; the static type widens.
    This helper hides that single cast at one site so callers stay clean.

    ``FLASHDREAMS_RECOMPILE_LIMIT`` and
    ``FLASHDREAMS_ACCUMULATED_RECOMPILE_LIMIT`` override Dynamo's cache limits.
    Unset variables preserve the existing limits.

    Args:
        module: ``nn.Module`` to compile.
        mode: One of the four ``torch.compile`` modes; see :data:`CompileMode`.
        dynamic: Dynamic-shape policy forwarded to ``torch.compile``.

    Returns:
        The compiled module, statically typed as the same ``M`` so attribute
        access on the wrapped module continues to type-check at call sites.
    """
    _configure_dynamo_cache()
    _configure_inductor_cache()
    _patch_triton_bundle_collection()
    return cast(M, torch.compile(module, mode=mode, dynamic=dynamic))
