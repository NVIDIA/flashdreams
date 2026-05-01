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

"""Ring (sequence-parallel) attention built on the native SDPA primitive."""

from contextlib import nullcontext
from typing import Any, Callable, ContextManager, Literal, cast

import torch
import torch.distributed._functional_collectives as funcol
from torch import Tensor
from torch.distributed.tensor.device_mesh import DeviceMesh

from flashdreams.core.attention.native import NativeAttention

# For type checking, cast the native ops to the expected signature.
_sdpa_cudnn: Callable[..., tuple[Tensor, ...]] = cast(
    "Callable[..., tuple[Tensor, ...]]",
    torch.ops.aten._scaled_dot_product_cudnn_attention,
)
_sdpa_flash: Callable[..., tuple[Any, ...]] = cast(
    "Callable[..., tuple[Any, ...]]",
    torch.ops.aten._scaled_dot_product_flash_attention,
)


def torch_sdpa_cudnn(
    query: Tensor, key: Tensor, value: Tensor, return_lse: bool = False
) -> tuple[Tensor, Tensor | None]:
    """Scaled dot-product attention via CuDNN backend (supports LSE for ring merge).

    Args:
        query: Query tensor, shape ``[B, H, S, D]``.
        key: Key tensor, shape ``[B, H, S, D]``.
        value: Value tensor, shape ``[B, H, S, D]``.
        return_lse: If True, return (output, log_sum_exp) for ring merge.

    Returns:
        Attention output, or ``(output, lse)`` when ``return_lse=True``.
    """
    out, lse, *_ = _sdpa_cudnn(
        query,
        key,
        value,
        None,  # attn_bias
        True,  # compute_log_sumexp
    )
    return out, (lse if return_lse else None)


def torch_sdpa_flash(
    query: Tensor, key: Tensor, value: Tensor, return_lse: bool = False
) -> tuple[Tensor, Tensor | None]:
    """Scaled dot-product attention via Flash Attention backend (returns LSE for ring merge).

    Args:
        query: Query tensor, shape ``[B, H, S, D]``.
        key: Key tensor, shape ``[B, H, S, D]``.
        value: Value tensor, shape ``[B, H, S, D]``.
        return_lse: If True, return (output, log_sum_exp) for ring merge.

    Returns:
        Attention output, or ``(output, lse)`` when ``return_lse=True``.
    """
    out, lse, *_ = _sdpa_flash(query, key, value)
    return out, (lse if return_lse else None)


class RingAttention(NativeAttention):
    """Context-parallel ring attention module with configurable QKV layout and SDPA backend."""

    def __init__(
        self,
        qkv_format: Literal["bhsd", "bshd"] = "bhsd",
        backend: Literal["cudnn", "flash"] = "cudnn",
        convert_to_fp32: bool = True,
    ) -> None:
        """Configure ring attention format and backend.

        Args:
            qkv_format: "bshd" (B, S, H, D) or "bhsd" (B, H, S, D). Default is "bhsd".
            backend: "cudnn" or "flash" for the manual ring SDPA ops.
            convert_to_fp32: Use float32 for LSE merge across ring steps.
        """
        super().__init__()
        assert qkv_format in ["bhsd", "bshd"], f"Invalid qkv format: {qkv_format}"
        assert backend in ["cudnn", "flash"], f"Invalid backend: {backend}"
        self.qkv_format = qkv_format
        self.backend = backend
        self.device_mesh: DeviceMesh | None = None
        self.convert_to_fp32 = convert_to_fp32

    def _impl(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Ring attention: all-gather KV across CP ranks, run local SDPA, merge with LSE.

        Q is replicated per rank; K/V are sharded. Each rank runs SDPA over gathered KV
        and merges partial outputs using log-sum-exp. Expects q,k,v in ``[B, H, S, D]``.

        Args:
            query: Query tensor, shape ``[B, H, S, D]`` (CP-shared).
            key: Key tensor, shape ``[B, H, S, D]`` (CP-sharded).
            value: Value tensor, shape ``[B, H, S, D]`` (CP-sharded).

        Returns:
            Attention output tensor.
        """
        attn_op = {
            "cudnn": torch_sdpa_cudnn,
            "flash": torch_sdpa_flash,
        }[self.backend]

        if self.device_mesh is None:
            return attn_op(query, key, value, return_lse=False)[0]

        rank = self.device_mesh.get_rank()
        world_size = self.device_mesh.size()
        group = self.device_mesh.get_group()
        if world_size == 1:
            return attn_op(query, key, value, return_lse=False)[0]

        next_rank = (rank + 1) % world_size
        prev_out = prev_lse = None

        kv_buffer_local = torch.cat([key.flatten(), value.flatten()]).contiguous()
        kv_buffer_gathered = funcol.all_gather_tensor(
            kv_buffer_local, gather_dim=0, group=group
        )
        kv_buffer = kv_buffer_gathered.chunk(world_size)

        for i in range(world_size):
            if i > 0:
                kv = kv_buffer[next_rank]
                key = kv[: key.numel()].reshape_as(key)
                value = kv[key.numel() :].reshape_as(value)
                next_rank = (next_rank + 1) % world_size

            out, lse = attn_op(query, key, value, return_lse=True)
            if lse is None:
                raise AssertionError("LSE is None")

            precision_context: ContextManager[None]
            if self.convert_to_fp32:
                precision_context = torch.autocast(device_type="cuda", enabled=False)
            else:
                precision_context = nullcontext()

            with precision_context:
                if self.convert_to_fp32:
                    out = out.to(torch.float32)
                    lse = lse.to(torch.float32)

                if prev_out is not None and prev_lse is not None:
                    out = prev_out - torch.nn.functional.sigmoid(lse - prev_lse) * (
                        prev_out - out
                    )
                    lse = prev_lse - torch.nn.functional.logsigmoid(prev_lse - lse)
            prev_out = out
            prev_lse = lse

        out = out.to(query.dtype)
        return out
