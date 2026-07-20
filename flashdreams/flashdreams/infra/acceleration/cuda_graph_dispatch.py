# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA-graph dispatch policy helpers for autoregressive inference."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

CUDAGraphWrapper: Any | None = None


def _cuda_graph_wrapper_cls() -> Any:
    if CUDAGraphWrapper is not None:
        return CUDAGraphWrapper
    return _default_cuda_graph_wrapper_cls()


@lru_cache(maxsize=1)
def _default_cuda_graph_wrapper_cls() -> Any:
    from flashdreams.infra.cuda_graph import CUDAGraphWrapper as wrapper_cls

    return wrapper_cls


def cuda_graph_capture_ar_index(
    *,
    sink_size_t: int,
    window_size_t: int,
    len_t: int,
    size_name: str = "sink_size_t + window_size_t",
    len_name: str = "len_t",
) -> int:
    """Return the first AR index whose KV cache is in steady state."""
    chunks_total = sink_size_t + window_size_t
    assert len_t > 0, f"{len_name} ({len_t}) must be positive."
    assert chunks_total % len_t == 0, (
        f"{size_name} ({chunks_total}) must be divisible by {len_name} "
        f"({len_t}) so the KV cache can fit a whole number of AR chunks."
    )
    return chunks_total // len_t


class CUDAGraphDispatch:
    """Dispatch filling AR steps eagerly and steady AR steps through CUDA graphs.

    The dispatch owns one wrapper for the conditional branch and one for the
    CFG-unconditional branch. Both wrappers call the same underlying function,
    but they must not share static buffers because the branch caches diverge.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        enabled: bool,
        capture_ar_idx: int,
        warmup_iters: int,
        build: bool = True,
    ) -> None:
        self.fn = fn
        self.enabled = enabled
        self.capture_ar_idx = capture_ar_idx
        self.warmup_iters = warmup_iters
        self._cond_call: Any | None = None
        self._uncond_call: Any | None = None
        if enabled and build:
            self.rebuild()

    @property
    def cond_call(self) -> Any | None:
        return self._cond_call

    @property
    def uncond_call(self) -> Any | None:
        return self._uncond_call

    def rebuild(self, *, capture_ar_idx: int | None = None) -> None:
        """Build fresh wrappers, optionally updating the steady-state threshold."""
        if capture_ar_idx is not None:
            self.capture_ar_idx = capture_ar_idx
        if not self.enabled:
            self._cond_call = None
            self._uncond_call = None
            return
        wrapper_cls = _cuda_graph_wrapper_cls()
        self._cond_call = wrapper_cls(self.fn, warmup_iters=self.warmup_iters)
        self._uncond_call = wrapper_cls(self.fn, warmup_iters=self.warmup_iters)

    def disable(self, *, fn: Callable[..., Any] | None = None) -> None:
        """Disable graph dispatch and drop wrapper references."""
        if fn is not None:
            self.fn = fn
        self.enabled = False
        self._cond_call = None
        self._uncond_call = None

    def reset(self) -> None:
        """Reset captured graphs and staged buffers for a fresh cache."""
        if not self.enabled:
            return
        if self._cond_call is None or self._uncond_call is None:
            self.rebuild()
            return
        self._cond_call.reset()
        self._uncond_call.reset()

    def select(
        self,
        autoregressive_index: int,
        *,
        uncond: bool,
    ) -> Callable[..., Any]:
        """Return eager ``fn``, wrapper ``drain``, or wrapper replay callable."""
        if not self.enabled:
            return self.fn
        wrapper = self._uncond_call if uncond else self._cond_call
        if wrapper is None:
            raise RuntimeError(
                "CUDA graph dispatch was selected before wrappers were built."
            )
        if autoregressive_index < self.capture_ar_idx:
            return wrapper.drain
        return wrapper
