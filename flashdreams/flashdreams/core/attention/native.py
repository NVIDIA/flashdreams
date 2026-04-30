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

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup
from torch.distributed.tensor.device_mesh import DeviceMesh


class NativeAttention(torch.nn.Module):
    """Native attention module with configurable QKV layout and SDPA backend."""

    def __init__(
        self,
        qkv_format: Literal["bhsd", "bshd"] = "bhsd",
        backend: Literal["math", "efficient", "cudnn", "flash"] = "cudnn",
    ) -> None:
        """Configure attention format and backend.

        Args:
            qkv_format: "bshd" (B, S, H, D) or "bhsd" (B, H, S, D). Default is "bhsd".
            backend: One of "math", "efficient", "cudnn", "flash" for ``sdpa_kernel``.
        """
        super().__init__()
        assert qkv_format in ["bhsd", "bshd"], f"Invalid qkv format: {qkv_format}"
        assert backend in ["math", "efficient", "cudnn", "flash"], (
            f"Invalid backend: {backend}"
        )
        self.qkv_format = qkv_format
        self.backend = backend
        self.device_mesh: DeviceMesh | None = None

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
        """Run ring attention (or single-rank SDPA when CP is disabled).

        Args:
            query: Query tensor in configured ``qkv_format``.
            key: Key tensor in configured ``qkv_format``.
            value: Value tensor in configured ``qkv_format``.

        Returns:
            Attention output in the same format as inputs.
        """
        # SDPA / low-level ops expect (B, H, S, D). "bshd" is (B, S, H, D) → transpose once.
        if self.qkv_format == "bshd":
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
        out = self._impl(query=query, key=key, value=value)
        if self.qkv_format == "bshd":
            out = out.transpose(1, 2)
        return out

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
                with torch.distributed.tensor.experimental.context_parallel(  # ty:ignore[possibly-missing-submodule]
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
