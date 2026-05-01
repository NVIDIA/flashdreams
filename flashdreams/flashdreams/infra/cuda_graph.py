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

"""Reusable CUDA-graph capture wrapper for stateful inference callables."""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten


class CUDAGraphWrapper:
    """Capture a stateful CUDA callable into a replayable graph.

    The callable runs eagerly for ``warmup_iters`` calls so kernels JIT-load
    and the allocator stabilises. The next call captures the whole forward
    into a ``torch.cuda.CUDAGraph`` against static input buffers; every
    same-shape call after that copies inputs into those buffers and replays
    the graph, returning clones of the captured outputs.

    Captured kernels reference the static input pointers, the callable's
    parameters, the output buffer pointers, and any in-place-mutated buffers
    (e.g. cache slots). Those internal buffers must have stable storage
    addresses — typically by writing through ``copy_`` once a slot's shape
    stabilises.

    Input staging: only top-level tensor positional args and kwargs are
    copied into static buffers. Everything else (ints, ``None``, dicts,
    custom objects) passes through verbatim. This is intentional: callers
    routinely pass mutable state through containers (e.g. a streaming cache
    as a ``dict[int, Tensor]``) and recursing in would break the in-place
    semantics. Pass any per-call-varying tensor as its own arg.

    A change in the staged-tensor signature drops the graph and restarts
    warmup. ``reset`` does the same explicitly — call it when external
    state (e.g. a fresh streaming cache) is swapped out.

    Compatibility with ``torch.compile``: a compiled ``fn`` is fine, but
    Inductor + triton trigger lazy autotunes (illegal during capture) on the
    first call per shape. Drain them on the eager path via ``drain``
    (or an unwrapped call) before the wrapped path captures, otherwise
    capture fails with ``cudaErrorStreamCaptureUnsupported``.

    Typical usage example:

        wrapper = CUDAGraphWrapper(model.forward, warmup_iters=2)

        # Rollout 1: drain Inductor autotune on the eager path.
        for chunk in first_rollout_chunks:
            y = wrapper.drain(chunk, timesteps=t, cache=cache)

        # Rollout 2+: capture + replay.
        for chunk in steady_rollout_chunks:
            y = wrapper(chunk, timesteps=t, cache=cache)
    """

    def __init__(self, fn: Callable[..., Any], warmup_iters: int = 2):
        self.fn = fn
        self.warmup_iters = warmup_iters
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        # Input staging state.
        self._static_args: list[Any] = []  # one slot per positional arg
        self._static_kwargs: dict[str, Any] = {}  # one slot per kwarg name
        # Output staging state.
        self._static_out_leaves: Optional[list[Any]] = None
        self._out_spec: Any = None
        self._warmup_remaining = warmup_iters

    def reset(self) -> None:
        self._graph = None
        self._static_args = []
        self._static_kwargs = {}
        self._static_out_leaves = None
        self._out_spec = None
        self._warmup_remaining = self.warmup_iters

    ## Input staging

    @staticmethod
    def _slot_compatible(slot: Any, fresh: Any) -> bool:
        """Can ``slot`` absorb ``fresh``?

        A tensor slot accepts a tensor of the same shape and dtype; a
        non-tensor slot accepts any non-tensor value (forwarded verbatim).
        """
        if isinstance(slot, torch.Tensor):
            return (
                isinstance(fresh, torch.Tensor)
                and slot.shape == fresh.shape
                and slot.dtype == fresh.dtype
            )
        return not isinstance(fresh, torch.Tensor)

    def _slots_compatible_with(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> bool:
        if len(self._static_args) != len(args):
            return False
        if set(self._static_kwargs) != set(kwargs):
            return False
        for slot, fresh in zip(self._static_args, args):
            if not self._slot_compatible(slot, fresh):
                return False
        for name, slot in self._static_kwargs.items():
            if not self._slot_compatible(slot, kwargs[name]):
                return False
        return True

    @staticmethod
    def _make_slot(value: Any) -> Any:
        """Static buffer for a tensor; pass-through value for non-tensors."""
        if isinstance(value, torch.Tensor):
            return torch.empty_like(value).contiguous()
        return value

    def _stage(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Copy top-level tensors into static buffers; forward non-tensors verbatim.

        Reallocates buffers and drops the captured graph if the staged-
        tensor signature changes.
        """
        if not self._slots_compatible_with(args, kwargs):
            self.reset()
            self._static_args = [self._make_slot(a) for a in args]
            self._static_kwargs = {k: self._make_slot(v) for k, v in kwargs.items()}

        staged_args: list[Any] = []
        for slot, fresh in zip(self._static_args, args):
            if isinstance(slot, torch.Tensor):
                slot.copy_(fresh)
                staged_args.append(slot)
            else:
                staged_args.append(fresh)

        staged_kwargs: dict[str, Any] = {}
        for name, fresh in kwargs.items():
            slot = self._static_kwargs[name]
            if isinstance(slot, torch.Tensor):
                slot.copy_(fresh)
                staged_kwargs[name] = slot
            else:
                staged_kwargs[name] = fresh

        return tuple(staged_args), staged_kwargs

    ## Output handling

    def _clone_output(self) -> Any:
        assert self._static_out_leaves is not None and self._out_spec is not None
        cloned = [
            leaf.clone() if isinstance(leaf, torch.Tensor) else leaf
            for leaf in self._static_out_leaves
        ]
        return tree_unflatten(cloned, self._out_spec)

    ## Public entry points

    def drain(self, *args: Any, **kwargs: Any) -> Any:
        """Eager autotune drain through the shared static buffers.

        Used during the first rollout so Inductor's lazy triton autotunes
        run on the eager path against the same buffers + strides that
        ``__call__`` will later capture against. Without this, capture
        would trigger a second Inductor specialisation and a second
        multi-second autotune when the wrapper takes over.

        Does not consume ``warmup_iters`` and does not capture.
        """
        args, kwargs = self._stage(args, kwargs)
        return self.fn(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        args, kwargs = self._stage(args, kwargs)

        if self._graph is not None:
            self._graph.replay()
            # Clone: the next replay would overwrite the static output buffers.
            return self._clone_output()

        if self._warmup_remaining > 0:
            self._warmup_remaining -= 1
            return self.fn(*args, **kwargs)

        # Capture: trace one full forward against the static buffers.
        # cudaStreamBeginCapture only records kernels — it does not execute
        # them — so the static outputs and in-place cache updates are no-ops
        # here. Replay once immediately to actually compute the output and
        # advance the cache.
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            out = self.fn(*args, **kwargs)
        out_leaves, self._out_spec = tree_flatten(out)
        self._static_out_leaves = out_leaves
        self._graph.replay()
        return self._clone_output()
