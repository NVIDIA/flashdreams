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

"""Context-parallel attention (ring / ulysses) built on native SDPA primitives."""

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
    """Scaled dot-product attention via CuDNN backend."""
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
    """Scaled dot-product attention via Flash Attention backend."""
    out, lse, *_ = _sdpa_flash(query, key, value)
    return out, (lse if return_lse else None)


class ContextParallelAttention(NativeAttention):
    """Context-parallel attention with selectable method and SDPA backend."""

    def __init__(
        self,
        qkv_format: Literal["bhsd", "bshd"] = "bhsd",
        backend: Literal["cudnn", "flash"] = "cudnn",
        method: Literal["ring", "ulysses"] = "ring",
        convert_to_fp32: bool = True,
    ) -> None:
        """Configure context-parallel attention method and backend.

        Args:
            qkv_format: Layout of the QKV tensors; ``"bhsd"`` or ``"bshd"``.
            backend: SDPA backend; ``"cudnn"`` or ``"flash"``.
            method: Context-parallelism strategy; ``"ring"`` or ``"ulysses"``.
            convert_to_fp32: Promote LSE accumulators to fp32 during ring merges.
        """
        super().__init__()
        assert qkv_format in ["bhsd", "bshd"], f"Invalid qkv format: {qkv_format}"
        assert backend in ["cudnn", "flash"], f"Invalid backend: {backend}"
        assert method in ["ring", "ulysses"], f"Invalid cp method: {method}"
        self.qkv_format = qkv_format
        self.backend = backend
        self.method = method
        self.device_mesh: DeviceMesh | None = None
        self.convert_to_fp32 = convert_to_fp32

    @staticmethod
    def _wait_collective(tensor: Tensor) -> Tensor:
        """Wait for functional collective outputs when needed."""
        wait_fn = getattr(tensor, "wait", None)
        if callable(wait_fn):
            return cast(Tensor, wait_fn())
        return tensor

    def _impl(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        if self.method == "ring":
            return self._impl_ring(query, key, value)
        if self.method == "ulysses":
            return self._impl_ulysses(query, key, value)
        raise ValueError(f"Unsupported context parallel method: {self.method}")

    def _impl_ring(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Ring attention: all-gather KV across CP ranks and LSE-merge outputs."""
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

    def _impl_ulysses(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """Ulysses attention: all-to-all QKV, local SDPA, all-to-all output restore."""
        attn_op = {
            "cudnn": torch_sdpa_cudnn,
            "flash": torch_sdpa_flash,
        }[self.backend]

        if self.device_mesh is None:
            return attn_op(query, key, value, return_lse=False)[0]

        world_size = self.device_mesh.size()
        if world_size == 1:
            return attn_op(query, key, value, return_lse=False)[0]

        B, H, Sq_local, D = query.shape
        _, _, Sk_local, _ = key.shape
        if H % world_size != 0:
            raise ValueError(
                f"Number of heads ({H}) must be divisible by CP size ({world_size}) for Ulysses."
            )
        H_local = H // world_size
        group = self.device_mesh.get_group()

        query = (
            query.reshape(B, world_size, H_local, Sq_local, D)
            .permute(1, 3, 0, 2, 4)
            .contiguous()
        )
        key = (
            key.reshape(B, world_size, H_local, Sk_local, D)
            .permute(1, 3, 0, 2, 4)
            .contiguous()
        )
        value = (
            value.reshape(B, world_size, H_local, Sk_local, D)
            .permute(1, 3, 0, 2, 4)
            .contiguous()
        )
        query, key, value = (
            self._wait_collective(funcol.all_to_all_single(x, None, None, group=group))
            for x in (query, key, value)
        )
        query, key, value = (
            x.flatten(0, 1).permute(1, 2, 0, 3).contiguous()
            for x in (query, key, value)
        )

        out, _ = attn_op(query, key, value, return_lse=True)

        out = (
            out.reshape(B, H_local, world_size, Sq_local, D)
            .permute(2, 1, 0, 3, 4)
            .contiguous()
        )
        out = self._wait_collective(
            funcol.all_to_all_single(out, None, None, group=group)
        )
        out = out.flatten(0, 1).permute(1, 0, 2, 3).contiguous()
        return out
