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

"""Shared mask, compilation, and kernel policy for FlexAttention."""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention


@dataclass(frozen=True, slots=True)
class FlexAttentionOptions:
    """Compilation, mask-block, and kernel policy for FlexAttention."""

    block_size: int = 128
    """Square fallback used when building a block mask."""

    mask_block_m: int | None = None
    """Optional query block size; ``None`` uses ``block_size``."""

    mask_block_n: int | None = None
    """Optional key/value block size; ``None`` uses ``block_size``."""

    compile_dynamic: bool | None = None
    """Dynamic-shape policy forwarded to :func:`torch.compile`."""

    block_m: int | None = None
    """Optional forward query tile size; ``None`` lets PyTorch choose."""

    block_n: int | None = None
    """Optional forward key/value tile size; ``None`` lets PyTorch choose."""

    num_warps: int | None = None
    """Optional Triton warp count; ``None`` lets PyTorch choose."""

    num_stages: int | None = None
    """Optional Triton pipeline-stage count; ``None`` lets PyTorch choose."""

    prescale_qk: bool | None = None
    """Whether to apply the attention scale before the QK reduction."""

    use_tma: bool | None = None
    """Whether to request TMA from FlexAttention; ``None`` uses its default."""

    backend: str | None = None
    """Optional FlexAttention kernel backend: ``TRITON`` or ``FLASH``."""

    rows_guaranteed_safe: bool = False
    """Skip empty-row guards when every query sees at least one key."""

    @property
    def mask_block_size(self) -> int | tuple[int, int]:
        """Return square or asymmetric ``(query, key/value)`` mask blocks."""
        block_m = self.block_size if self.mask_block_m is None else self.mask_block_m
        block_n = self.block_size if self.mask_block_n is None else self.mask_block_n
        return block_m if block_m == block_n else (block_m, block_n)

    @property
    def kernel_options(self) -> dict[str, int | bool | str]:
        """Return only explicitly selected FlexAttention kernel options."""
        pairs = {
            "BLOCK_M": self.block_m,
            "BLOCK_N": self.block_n,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "PRESCALE_QK": self.prescale_qk,
            "USE_TMA": self.use_tma,
            "BACKEND": self.backend,
        }
        options = {name: value for name, value in pairs.items() if value is not None}
        if self.rows_guaranteed_safe:
            options["ROWS_GUARANTEED_SAFE"] = True
        return options

    def __post_init__(self) -> None:
        """Reject invalid values before they reach a compiled kernel."""
        for name, value in (
            ("block_size", self.block_size),
            ("mask_block_m", self.mask_block_m),
            ("mask_block_n", self.mask_block_n),
            ("block_m", self.block_m),
            ("block_n", self.block_n),
        ):
            if value is not None and (value < 16 or value & (value - 1)):
                raise ValueError(
                    f"FlexAttention {name} must be a power of two >= 16; got {value}"
                )
        if self.compile_dynamic is not None and not isinstance(
            self.compile_dynamic, bool
        ):
            raise TypeError(
                f"compile_dynamic must be bool or None; got {self.compile_dynamic!r}"
            )
        if self.num_warps is not None and self.num_warps not in (1, 2, 4, 8):
            raise ValueError(
                f"num_warps must be one of 1, 2, 4, or 8; got {self.num_warps}"
            )
        if self.num_stages is not None and self.num_stages <= 0:
            raise ValueError(f"num_stages must be positive; got {self.num_stages}")
        for name, value in (
            ("prescale_qk", self.prescale_qk),
            ("use_tma", self.use_tma),
        ):
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{name} must be bool or None; got {value!r}")
        if self.backend is not None and self.backend not in ("TRITON", "FLASH"):
            raise ValueError(
                f"backend must be TRITON, FLASH, or None; got {self.backend!r}"
            )
        if not isinstance(self.rows_guaranteed_safe, bool):
            raise TypeError(
                f"rows_guaranteed_safe must be bool; got {self.rows_guaranteed_safe!r}"
            )


@functools.cache
def compiled_flex_attention(*, dynamic: bool | None = None) -> Callable[..., Tensor]:
    """Compile FlexAttention once for each dynamic-shape policy."""
    return torch.compile(cast(Callable[..., Tensor], flex_attention), dynamic=dynamic)


__all__ = ["FlexAttentionOptions", "compiled_flex_attention"]
