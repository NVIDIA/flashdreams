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

"""Attention kernels with selectable QKV layouts and backends."""

from __future__ import annotations

from functools import cache
from importlib import import_module
from typing import Callable, Literal, cast

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental import context_parallel

SageAttentionOp = Callable[..., Tensor | tuple[Tensor, Tensor]]


@cache
def _load_sage2_op() -> SageAttentionOp:
    """Load and cache the optional SageAttention 2 entry point.

    Returns:
        SageAttention's backend-selecting attention callable.

    Raises:
        RuntimeError: SageAttention 2 or its CUDA extension cannot be imported.
    """
    try:
        sageattn = getattr(import_module("sageattention"), "sageattn")
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "The 'sage2' attention backend requires SageAttention 2.x with its "
            "CUDA extension built. Install SageAttention before selecting "
            "backend='sage2'."
        ) from exc
    return cast(SageAttentionOp, sageattn)


def _validate_sage2_inputs(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    tensor_layout: Literal["HND", "NHD"],
) -> None:
    """Validate QKV constraints imposed by SageAttention 2.

    Args:
        query: Query tensor in ``tensor_layout``.
        key: Key tensor in ``tensor_layout``.
        value: Value tensor in ``tensor_layout``.
        tensor_layout: SageAttention layout identifier.

    Raises:
        ValueError: QKV rank, shape, dtype, device, or stride is unsupported.
    """
    tensors = (query, key, value)
    if any(tensor.ndim != 4 for tensor in tensors):
        raise ValueError("SageAttention 2 requires rank-4 query, key, and value.")

    supported_dtypes = (torch.float16, torch.bfloat16)
    if any(tensor.dtype not in supported_dtypes for tensor in tensors):
        raise ValueError(
            "SageAttention 2 requires query, key, and value to use float16 or bfloat16."
        )
    if not (query.dtype == key.dtype == value.dtype):
        raise ValueError(
            "SageAttention 2 requires query, key, and value to have the same dtype."
        )

    head_dims = (query.shape[-1], key.shape[-1], value.shape[-1])
    if not (head_dims[0] == head_dims[1] == head_dims[2]):
        raise ValueError(
            "SageAttention 2 requires query, key, and value to have the same "
            "head dimension."
        )
    if not 0 < head_dims[0] <= 128:
        raise ValueError(
            f"SageAttention 2 supports head dimensions from 1 through 128; got "
            f"{head_dims[0]}."
        )

    sequence_dim = 2 if tensor_layout == "HND" else 1
    heads_dim = 1 if tensor_layout == "HND" else 2
    if not (query.shape[0] == key.shape[0] == value.shape[0]):
        raise ValueError(
            "SageAttention 2 requires query, key, and value to have the same batch "
            "size."
        )
    if key.shape[sequence_dim] != value.shape[sequence_dim]:
        raise ValueError(
            "SageAttention 2 requires key and value sequence lengths to match."
        )
    if key.shape[heads_dim] != value.shape[heads_dim]:
        raise ValueError("SageAttention 2 requires key and value head counts to match.")
    query_heads = query.shape[heads_dim]
    key_heads = key.shape[heads_dim]
    if key_heads == 0 or query_heads == 0 or query_heads % key_heads != 0:
        raise ValueError(
            "SageAttention 2 requires the query head count to be a positive "
            "multiple of the key/value head count."
        )

    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError(
            "SageAttention 2 requires query, key, and value on a CUDA device."
        )
    if not (query.device == key.device == value.device):
        raise ValueError(
            "SageAttention 2 requires query, key, and value on the same CUDA device."
        )
    if any(tensor.stride(-1) != 1 for tensor in tensors):
        raise ValueError(
            "SageAttention 2 requires the last dimension of query, key, and value "
            "to be contiguous."
        )


class NativeAttention(torch.nn.Module):
    """Native attention module with configurable QKV layout and kernel backend."""

    def __init__(
        self,
        qkv_format: Literal["bhsd", "bshd"] = "bhsd",
        backend: Literal["math", "efficient", "cudnn", "flash", "sage2"] = "cudnn",
    ) -> None:
        """Configure attention format and backend.

        Args:
            qkv_format: Layout of the QKV tensors; ``"bhsd"`` is ``(B, H, S, D)``,
                ``"bshd"`` is ``(B, S, H, D)``.
            backend: Attention kernel backend. ``"sage2"`` loads the optional
                SageAttention 2 package.
        """
        super().__init__()
        assert qkv_format in ["bhsd", "bshd"], f"Invalid qkv format: {qkv_format}"
        assert backend in ["math", "efficient", "cudnn", "flash", "sage2"], (
            f"Invalid backend: {backend}"
        )
        self.qkv_format = qkv_format
        self.backend = backend
        self.device_mesh: DeviceMesh | None = None
        self._sage2_op = _load_sage2_op() if backend == "sage2" else None

    def set_context_parallel_group(self, cp_group: ProcessGroup | None) -> None:
        """Enable or disable context parallelism for ring attention.

        Args:
            cp_group: Process group for context parallel; use None to disable.
        """
        if cp_group is None:
            self.device_mesh = None
        else:
            self.device_mesh = DeviceMesh.from_group(cp_group, device_type="cuda")

            # Need to disable load balance for torch context parallel to work.
            from torch.distributed.tensor.experimental._attention import (
                _cp_options,
                set_rotate_method,
            )

            _cp_options.enable_load_balance = False
            set_rotate_method("allgather")

    def is_context_parallel_enabled(self) -> bool:
        """Return True if context parallelism is active."""
        return self.device_mesh is not None

    def context_parallel_size(self) -> int:
        """Return the context parallel world size, or 1 if disabled."""
        return self.device_mesh.size() if self.device_mesh is not None else 1

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Run attention and preserve the configured QKV layout.

        Args:
            query: Query tensor in configured ``qkv_format``.
            key: Key tensor in configured ``qkv_format``.
            value: Value tensor in configured ``qkv_format``.

        Returns:
            Attention output in the same format as inputs.
        """
        # SageAttention consumes NHD directly; SDPA expects HND.
        sage_nhd = self.backend == "sage2" and self.qkv_format == "bshd"
        if self.qkv_format == "bshd" and not sage_nhd:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
        out = self._impl(query=query, key=key, value=value)
        if self.qkv_format == "bshd" and not sage_nhd:
            out = out.transpose(1, 2)
        return out

    def _run_sage2(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        tensor_layout: Literal["HND", "NHD"],
        return_lse: bool,
    ) -> tuple[Tensor, Tensor | None]:
        """Run SageAttention 2 and normalize its optional LSE result.

        Args:
            query: Query tensor in ``tensor_layout``.
            key: Key tensor in ``tensor_layout``.
            value: Value tensor in ``tensor_layout``.
            tensor_layout: SageAttention layout identifier.
            return_lse: Return natural-log softmax normalization factors.

        Returns:
            Attention output and an optional ``[B, H, S]`` LSE tensor.

        Raises:
            RuntimeError: SageAttention returns an incompatible result.
            ValueError: QKV inputs violate SageAttention constraints.
        """
        if self._sage2_op is None:
            raise RuntimeError("SageAttention 2 was not initialized for this module.")
        _validate_sage2_inputs(query, key, value, tensor_layout)
        result = self._sage2_op(
            query,
            key,
            value,
            tensor_layout=tensor_layout,
            is_causal=False,
            return_lse=return_lse,
        )

        if return_lse:
            if not isinstance(result, tuple) or len(result) != 2:
                raise RuntimeError(
                    "SageAttention 2 must return an (output, LSE) pair when "
                    "return_lse=True."
                )
            out, lse = result
            if not isinstance(out, Tensor) or not isinstance(lse, Tensor):
                raise RuntimeError("SageAttention 2 returned non-tensor output or LSE.")
        else:
            if not isinstance(result, Tensor):
                raise RuntimeError(
                    "SageAttention 2 must return one tensor when return_lse=False."
                )
            out = result
            lse = None

        if out.shape != query.shape:
            raise RuntimeError(
                "SageAttention 2 returned an output shape different from the query: "
                f"{tuple(out.shape)} != {tuple(query.shape)}."
            )
        if lse is not None:
            sequence_dim = 2 if tensor_layout == "HND" else 1
            heads_dim = 1 if tensor_layout == "HND" else 2
            expected_lse_shape = (
                query.shape[0],
                query.shape[heads_dim],
                query.shape[sequence_dim],
            )
            if lse.shape != expected_lse_shape:
                raise RuntimeError(
                    "SageAttention 2 returned an incompatible LSE shape: "
                    f"{tuple(lse.shape)} != {tuple(expected_lse_shape)}."
                )
        return out, lse

    def _impl(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        """Attention implementation.

        Args:
            query: Query tensor, shape ``[B, H, S, D]`` (CP-shared).
            key: Key tensor, shape ``[B, H, S, D]`` (CP-sharded).
            value: Value tensor, shape ``[B, H, S, D]`` (CP-sharded).

        Returns:
            Attention output.
        """
        if self.backend == "sage2":
            if self.device_mesh is not None:
                raise RuntimeError(
                    "NativeAttention's context-parallel wrapper only supports SDPA "
                    "backends. Use ContextParallelAttention with backend='sage2'."
                )
            tensor_layout: Literal["HND", "NHD"] = (
                "NHD" if self.qkv_format == "bshd" else "HND"
            )
            return self._run_sage2(
                query,
                key,
                value,
                tensor_layout=tensor_layout,
                return_lse=False,
            )[0]

        sdpa_backend = {
            "math": torch.nn.attention.SDPBackend.MATH,
            "efficient": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            "cudnn": torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
            "flash": torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        }[self.backend]

        with torch.nn.attention.sdpa_kernel(sdpa_backend):
            if self.device_mesh is not None:
                # Pass a dummy buffer to satisfy context_parallel's buffers[0].device
                # check (required in PyTorch 2.9+ where buffers cannot be empty).
                _dummy = torch.empty(self.device_mesh.size(), device=query.device)
                with context_parallel(
                    self.device_mesh,
                    buffers=[
                        _dummy,
                    ],
                    buffer_seq_dims=[
                        0,
                    ],
                    no_restore_buffers={_dummy},
                ):
                    out = F.scaled_dot_product_attention(query, key, value)
            else:
                out = F.scaled_dot_product_attention(query, key, value)

        return out
