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

"""Low-precision linear helpers for SANA-WM."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import torch
import triton
import triton.language as tl
from torch import nn

FP8_MAX_E4M3 = 448.0
FP8_SCALE_EPS = 1.0e-12
FP4_MAX_E2M1 = 6.0
NVFP4_BLOCK_SIZE = 16
NVFP4_E4M3_SCALE_EPS = 0.015625
NVFP4_GLOBAL_SCALE_EPS = 1.0e-12
_RHT16_SCALE = 0.25
_RHT16_SIGNS = (
    1.0,
    -1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
    1.0,
    1.0,
    -1.0,
    -1.0,
    -1.0,
    -1.0,
    1.0,
    1.0,
)


@dataclass(frozen=True)
class TorchScaledMMFP8Recipe:
    """FP8 recipe marker for SANA-WM Linear replacement."""

    name: str = "torch_scaled_mm_fp8"
    precision: Literal["fp8"] = "fp8"


@dataclass(frozen=True)
class TorchScaledMMFP4Recipe:
    """FP4 recipe marker for SANA-WM Linear replacement."""

    name: str = "torch_scaled_mm_fp4"
    precision: Literal["fp4"] = "fp4"
    use_rht: bool = True
    use_global_scale: bool = True
    weight_scale_2d: bool = True


QuantRecipe = TorchScaledMMFP8Recipe | TorchScaledMMFP4Recipe


@triton.jit
def _pack_fp32_to_fp4_pairs(values):
    packed = tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b8 byte0, byte1, byte2, byte3;
        cvt.rn.satfinite.e2m1x2.f32 byte0, $5, $1;
        cvt.rn.satfinite.e2m1x2.f32 byte1, $6, $2;
        cvt.rn.satfinite.e2m1x2.f32 byte2, $7, $3;
        cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $4;
        mov.b32 $0, {byte0, byte1, byte2, byte3};
        }
        """,
        constraints=("=r,r,r,r,r,r,r,r,r"),
        args=values,
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )
    return packed


@triton.jit
def _quantize_nvfp4_kernel(
    input_ptr,
    qdata_ptr,
    scale_ptr,
    global_scale_ptr,
    stride_m,
    stride_k,
    rows: tl.constexpr,
    cols: tl.constexpr,
    mask_scales: tl.constexpr,
    scale_2d: tl.constexpr,
    has_global_scale: tl.constexpr,
):
    fp4_max = 6.0
    fp8_max = 448.0
    scale_eps = 0.015625
    global_scale = 1.0
    if has_global_scale:
        global_scale = tl.load(global_scale_ptr).to(tl.float32)

    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1)

    offsets_m = pid_m * 128 + tl.arange(0, 128)[:, None]
    offsets_k = pid_k * 64 + tl.arange(0, 64)[None, :]
    mask = (offsets_m < rows) & (offsets_k < cols)
    values = tl.load(
        input_ptr + offsets_m * stride_m + offsets_k * stride_k,
        mask=mask,
        other=0.0,
    )
    values = values.to(tl.float32)
    if scale_2d:
        values_4d = values.reshape(8, 16, 4, 16)
        block_amax = tl.max(tl.max(tl.abs(values_4d), axis=3), axis=1)
        scales_f32 = block_amax / fp4_max
        if has_global_scale:
            scales_f32 = scales_f32 / global_scale
        scales_f32 = tl.clamp(scales_f32, scale_eps, fp8_max)
        scales_tile = scales_f32.to(tl.float8e4nv)
        scales = tl.broadcast_to(tl.expand_dims(scales_tile, 1), (8, 16, 4))
        scales = scales.reshape(128, 4)
    else:
        values_3d = values.reshape(128, 4, 16)
        block_amax = tl.max(tl.abs(values_3d), axis=2)
        scales_f32 = block_amax / fp4_max
        if has_global_scale:
            scales_f32 = scales_f32 / global_scale
        scales_f32 = tl.clamp(scales_f32, scale_eps, fp8_max)
        scales = scales_f32.to(tl.float8e4nv)

    values = values.reshape(128, 4, 16)
    scale_product = scales.to(tl.float32)
    if has_global_scale:
        scale_product = scale_product * global_scale
    quantized = tl.div_rn(values, scale_product[:, :, None])

    if mask_scales:
        scale_offsets_k = pid_k * 4 + tl.arange(0, 4)[None, :]
        scale_mask = (offsets_m < rows) & (scale_offsets_k < tl.cdiv(cols, 16))
        scales = tl.where(scale_mask, scales, 0.0)

    packed_scales = scales.reshape(4, 32, 4).permute(1, 0, 2).reshape(32, 16)
    scale_m = tl.arange(0, 32)[:, None]
    scale_k = tl.arange(0, 16)[None, :]
    tl.store(
        scale_ptr
        + (pid_m * tl.num_programs(0) + pid_k) * (32 * 16)
        + scale_m * 16
        + scale_k,
        packed_scales,
    )

    packed = _pack_fp32_to_fp4_pairs(quantized.reshape(128, 32, 2).split())
    q_offsets_m = pid_m * 128 + tl.arange(0, 128)[:, None]
    q_offsets_k = pid_k * 32 + tl.arange(0, 32)[None, :]
    q_mask = (q_offsets_m < rows) & (q_offsets_k < cols // 2)
    tl.store(qdata_ptr + q_offsets_m * (cols // 2) + q_offsets_k, packed, mask=q_mask)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _swizzled_nvfp4_scale_shape(rows: int, cols: int) -> tuple[int, int]:
    return _ceil_div(rows, 128) * 32, _ceil_div(cols, 64) * 16


def quantize_nvfp4_swizzled(
    input: torch.Tensor,
    *,
    global_scale: torch.Tensor | None = None,
    scale_2d: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a 2D CUDA tensor to NVFP4 qdata plus swizzled E4M3 scales."""
    _require_fp4_dtype()
    if input.dim() != 2:
        raise ValueError(
            f"NVFP4 quantization requires a 2D tensor, got {input.dim()}D."
        )
    if not input.is_cuda:
        raise RuntimeError("NVFP4 quantization requires CUDA input.")
    rows, cols = input.shape
    if cols % NVFP4_BLOCK_SIZE != 0:
        raise ValueError(
            f"NVFP4 quantization requires last dim divisible by {NVFP4_BLOCK_SIZE}, got {cols}."
        )
    input_2d = input.contiguous()
    scale_rows, scale_cols = _swizzled_nvfp4_scale_shape(rows, cols)
    qdata = torch.empty((rows, cols // 2), device=input.device, dtype=torch.uint8)
    scales = torch.empty(
        (scale_rows, scale_cols),
        device=input.device,
        dtype=torch.float8_e4m3fn,
    )
    grid = (_ceil_div(cols, 64), _ceil_div(rows, 128))
    _quantize_nvfp4_kernel[grid](
        input_2d,
        qdata,
        scales,
        global_scale if global_scale is not None else input_2d,
        input_2d.stride(0),
        input_2d.stride(1),
        rows,
        cols,
        mask_scales=(rows % 128 != 0 or cols % 64 != 0),
        scale_2d=scale_2d,
        has_global_scale=global_scale is not None,
    )
    return qdata, scales


def nvfp4_global_scale(input: torch.Tensor) -> torch.Tensor:
    """Return the per-tensor FP32 global scale for hierarchical NVFP4."""
    amax = input.detach().abs().amax().to(torch.float32)
    scale = amax / (FP8_MAX_E4M3 * FP4_MAX_E2M1)
    return scale.clamp_min(NVFP4_GLOBAL_SCALE_EPS).reshape(())


def apply_rht16(input: torch.Tensor) -> torch.Tensor:
    """Apply the tiled 16-wide random Hadamard transform used by NVFP4."""
    if input.shape[-1] % 16 != 0:
        raise ValueError(
            f"RHT16 requires the last dimension to be divisible by 16, got {input.shape[-1]}."
        )
    original_shape = input.shape
    work = input.to(torch.float32).reshape(-1, 16)
    signs = torch.tensor(_RHT16_SIGNS, device=input.device, dtype=work.dtype)
    work = work * signs
    width = 1
    while width < 16:
        work = work.reshape(-1, 16 // (2 * width), 2, width)
        left = work[:, :, 0, :]
        right = work[:, :, 1, :]
        work = torch.stack((left + right, left - right), dim=2).reshape(-1, 16)
        width *= 2
    return (work * _RHT16_SCALE).reshape(original_shape).contiguous()


class TorchScaledMMFP8Linear(nn.Module):
    """Inference-only FP8 Linear using ``torch._scaled_mm``.

    The module stores the source weight as E4M3 FP8 with per-output-channel
    scales. At runtime it quantizes the flattened activation rows to E4M3 with
    per-row scales, then calls ``torch._scaled_mm`` and returns BF16 output.
    This intentionally mirrors the shape contract of ``nn.Linear`` so it can
    replace eligible SANA-WM linear layers without changing call sites.
    """

    in_features: int
    out_features: int
    out_dtype: torch.dtype
    weight: torch.Tensor
    weight_fp8: torch.Tensor
    weight_scale: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self,
        *,
        weight: torch.Tensor,
        weight_fp8: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if weight.shape != weight_fp8.shape:
            raise ValueError(
                "weight and weight_fp8 must have matching shape, got "
                f"{tuple(weight.shape)} and {tuple(weight_fp8.shape)}."
            )
        if weight_fp8.dim() != 2:
            raise ValueError(f"weight_fp8 must be 2D, got {tuple(weight_fp8.shape)}.")
        if weight_scale.shape != (weight_fp8.shape[0], 1):
            raise ValueError(
                "weight_scale must have shape "
                f"({weight_fp8.shape[0]}, 1), got {tuple(weight_scale.shape)}."
            )
        if bias is not None and bias.shape != (weight_fp8.shape[0],):
            raise ValueError(
                f"bias must have shape ({weight_fp8.shape[0]},), got {tuple(bias.shape)}."
            )
        self.out_features = int(weight_fp8.shape[0])
        self.in_features = int(weight_fp8.shape[1])
        self.out_dtype = out_dtype
        self.register_buffer("weight", weight.detach().contiguous())
        self.register_buffer("weight_fp8", weight_fp8.contiguous())
        self.register_buffer(
            "weight_scale", weight_scale.to(torch.float32).contiguous()
        )
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().to(out_dtype).contiguous())

    @classmethod
    def from_linear(
        cls,
        source: nn.Linear,
        *,
        out_dtype: torch.dtype,
    ) -> "TorchScaledMMFP8Linear":
        """Create an FP8 replacement from a source ``nn.Linear``."""
        _require_fp8_dtype()
        weight_f32 = source.weight.detach().to(torch.float32)
        weight_scale = (
            weight_f32.abs().amax(dim=1, keepdim=True).clamp_min(FP8_SCALE_EPS)
            / FP8_MAX_E4M3
        )
        weight_fp8 = torch.clamp(
            weight_f32 / weight_scale,
            -FP8_MAX_E4M3,
            FP8_MAX_E4M3,
        ).to(torch.float8_e4m3fn)
        bias = source.bias.detach() if source.bias is not None else None
        replacement = cls(
            weight=source.weight.detach().to(device=source.weight.device),
            weight_fp8=weight_fp8.to(device=source.weight.device),
            weight_scale=weight_scale.to(device=source.weight.device),
            bias=bias.to(device=source.weight.device) if bias is not None else None,
            out_dtype=out_dtype,
        )
        replacement.train(source.training)
        return replacement

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the quantized linear projection."""
        if input.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input last dim {self.in_features}, got {input.shape[-1]}."
            )
        if not input.is_cuda:
            raise RuntimeError("TorchScaledMMFP8Linear requires CUDA input.")
        _require_scaled_mm()
        _require_fp8_dtype()

        leading_shape = input.shape[:-1]
        input_2d = input.reshape(-1, self.in_features)
        if input_2d.numel() == 0:
            return input.new_empty(
                (*leading_shape, self.out_features),
                dtype=self.out_dtype,
            )

        input_f32 = input_2d.to(torch.float32)
        input_scale = (
            input_f32.abs().amax(dim=1, keepdim=True).clamp_min(FP8_SCALE_EPS)
            / FP8_MAX_E4M3
        )
        input_fp8 = torch.clamp(
            input_f32 / input_scale,
            -FP8_MAX_E4M3,
            FP8_MAX_E4M3,
        ).to(torch.float8_e4m3fn)
        input_fp8 = input_fp8.contiguous()

        # ``_scaled_mm`` expects the B operand as a column-major transpose view,
        # so keep the transpose stride instead of forcing it contiguous.
        output = torch._scaled_mm(
            input_fp8,
            self.weight_fp8.t(),
            scale_a=input_scale.contiguous(),
            scale_b=self.weight_scale.t().contiguous(),
            out_dtype=self.out_dtype,
        )
        if self.bias is not None:
            output = output + self.bias.to(device=output.device, dtype=output.dtype)
        return output.reshape(*leading_shape, self.out_features)


class TorchScaledMMFP4Linear(nn.Module):
    """Inference-only NVFP4 Linear using Triton quantization and ``_scaled_mm``."""

    in_features: int
    out_features: int
    out_dtype: torch.dtype
    use_rht: bool
    weight: torch.Tensor
    weight_qdata: torch.Tensor
    weight_scale: torch.Tensor
    weight_global_scale: torch.Tensor | None
    bias: torch.Tensor | None

    def __init__(
        self,
        *,
        weight: torch.Tensor,
        weight_qdata: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_global_scale: torch.Tensor | None,
        bias: torch.Tensor | None,
        out_dtype: torch.dtype,
        use_rht: bool,
    ) -> None:
        super().__init__()
        expected_weight_shape = (weight_qdata.shape[0], weight_qdata.shape[1] * 2)
        if weight.shape != expected_weight_shape:
            raise ValueError(
                "weight must match unpacked weight_qdata shape "
                f"{expected_weight_shape}, got {tuple(weight.shape)}."
            )
        if weight_qdata.dim() != 2:
            raise ValueError(
                f"weight_qdata must be 2D, got {tuple(weight_qdata.shape)}."
            )
        if weight_qdata.shape[1] * 2 % NVFP4_BLOCK_SIZE != 0:
            raise ValueError(
                "weight_qdata must represent a K dimension divisible by "
                f"{NVFP4_BLOCK_SIZE}, got packed shape {tuple(weight_qdata.shape)}."
            )
        if bias is not None and bias.shape != (weight_qdata.shape[0],):
            raise ValueError(
                f"bias must have shape ({weight_qdata.shape[0]},), got {tuple(bias.shape)}."
            )
        self.out_features = int(weight_qdata.shape[0])
        self.in_features = int(weight_qdata.shape[1] * 2)
        self.out_dtype = out_dtype
        self.use_rht = use_rht
        self.register_buffer("weight", weight.detach().contiguous())
        self.register_buffer("weight_qdata", weight_qdata.contiguous())
        self.register_buffer("weight_scale", weight_scale.contiguous())
        if weight_global_scale is None:
            self.register_buffer("weight_global_scale", None)
        else:
            self.register_buffer(
                "weight_global_scale",
                weight_global_scale.detach().to(torch.float32).reshape(()),
            )
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().to(out_dtype).contiguous())

    @classmethod
    def from_linear(
        cls,
        source: nn.Linear,
        *,
        out_dtype: torch.dtype,
        use_rht: bool = True,
        use_global_scale: bool = True,
        weight_scale_2d: bool = True,
    ) -> "TorchScaledMMFP4Linear":
        """Create an NVFP4 replacement from a source ``nn.Linear``."""
        _require_fp4_dtype()
        weight_for_quant = source.weight.detach()
        if use_rht:
            weight_for_quant = apply_rht16(weight_for_quant)
        weight_global_scale = (
            nvfp4_global_scale(weight_for_quant) if use_global_scale else None
        )
        weight_qdata, weight_scale = quantize_nvfp4_swizzled(
            weight_for_quant,
            global_scale=weight_global_scale,
            scale_2d=weight_scale_2d,
        )
        bias = source.bias.detach() if source.bias is not None else None
        replacement = cls(
            weight=source.weight.detach().to(device=source.weight.device),
            weight_qdata=weight_qdata,
            weight_scale=weight_scale,
            weight_global_scale=weight_global_scale,
            bias=bias.to(device=source.weight.device) if bias is not None else None,
            out_dtype=out_dtype,
            use_rht=use_rht,
        )
        replacement.train(source.training)
        return replacement

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the quantized linear projection."""
        if input.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input last dim {self.in_features}, got {input.shape[-1]}."
            )
        if not input.is_cuda:
            raise RuntimeError("TorchScaledMMFP4Linear requires CUDA input.")
        _require_scaled_mm()
        _require_fp4_dtype()

        leading_shape = input.shape[:-1]
        input_2d = input.reshape(-1, self.in_features)
        if input_2d.numel() == 0:
            return input.new_empty(
                (*leading_shape, self.out_features),
                dtype=self.out_dtype,
            )

        input_for_quant = apply_rht16(input_2d) if self.use_rht else input_2d
        input_global_scale = (
            nvfp4_global_scale(input_for_quant)
            if self.weight_global_scale is not None
            else None
        )
        input_qdata, input_scale = quantize_nvfp4_swizzled(
            input_for_quant,
            global_scale=input_global_scale,
        )
        add_bias_after_scale = (
            self.bias is not None
            and self.weight_global_scale is not None
            and input_global_scale is not None
        )
        output = torch._scaled_mm(
            input_qdata.view(torch.float4_e2m1fn_x2),
            self.weight_qdata.t().view(torch.float4_e2m1fn_x2),
            input_scale.view(torch.float8_e4m3fn),
            self.weight_scale.view(torch.float8_e4m3fn),
            bias=None
            if add_bias_after_scale
            else (
                self.bias.to(device=input.device, dtype=self.out_dtype)
                if self.bias is not None
                else None
            ),
            out_dtype=self.out_dtype,
        )
        if input_global_scale is not None and self.weight_global_scale is not None:
            output = output * (
                input_global_scale.to(device=output.device, dtype=output.dtype)
                * self.weight_global_scale.to(device=output.device, dtype=output.dtype)
            )
        if add_bias_after_scale:
            output = output + self.bias.to(device=output.device, dtype=output.dtype)
        return output.reshape(*leading_shape, self.out_features)


def replace_linear_with_torch_fp8(
    module: nn.Module,
    *,
    recipe: Any,
    params_dtype: torch.dtype,
    skip_patterns: tuple[str, ...],
    include_patterns: tuple[str, ...] | None = None,
    prefix: str = "",
) -> tuple[int, int]:
    """Replace eligible ``nn.Linear`` modules with ``TorchScaledMMFP8Linear``."""
    return _replace_linear_with_quant(
        module,
        recipe=TorchScaledMMFP8Recipe(),
        params_dtype=params_dtype,
        skip_patterns=skip_patterns,
        include_patterns=include_patterns,
        prefix=prefix,
    )


def replace_linear_with_torch_fp4(
    module: nn.Module,
    *,
    recipe: Any,
    params_dtype: torch.dtype,
    skip_patterns: tuple[str, ...],
    include_patterns: tuple[str, ...] | None = None,
    prefix: str = "",
) -> tuple[int, int]:
    """Replace eligible ``nn.Linear`` modules with ``TorchScaledMMFP4Linear``."""
    return _replace_linear_with_quant(
        module,
        recipe=TorchScaledMMFP4Recipe(),
        params_dtype=params_dtype,
        skip_patterns=skip_patterns,
        include_patterns=include_patterns,
        prefix=prefix,
    )


def replace_linear_with_quant(
    module: nn.Module,
    *,
    recipe: QuantRecipe,
    params_dtype: torch.dtype,
    skip_patterns: tuple[str, ...],
    include_patterns: tuple[str, ...] | None = None,
    prefix: str = "",
) -> tuple[int, int]:
    """Replace eligible ``nn.Linear`` modules with the requested backend."""
    return _replace_linear_with_quant(
        module,
        recipe=recipe,
        params_dtype=params_dtype,
        skip_patterns=skip_patterns,
        include_patterns=include_patterns,
        prefix=prefix,
    )


def _replace_linear_with_quant(
    module: nn.Module,
    *,
    recipe: QuantRecipe,
    params_dtype: torch.dtype,
    skip_patterns: tuple[str, ...],
    include_patterns: tuple[str, ...] | None,
    prefix: str,
) -> tuple[int, int]:
    converted = 0
    skipped = 0
    for name, child in list(module.named_children()):
        child_prefix = f"{prefix}.{name}" if prefix else name
        if _name_matches(skip_patterns, child_prefix):
            skipped += 1
            continue
        if isinstance(child, nn.Linear):
            if include_patterns is not None and not _name_matches(
                include_patterns,
                child_prefix,
            ):
                skipped += 1
                continue
            if child.in_features % 16 != 0 or child.out_features % 16 != 0:
                skipped += 1
                continue
            if recipe.precision == "fp4" and child.in_features % 32 != 0:
                skipped += 1
                continue
            out_dtype = (
                params_dtype
                if params_dtype in {torch.bfloat16, torch.float16}
                else torch.bfloat16
            )
            if recipe.precision == "fp8":
                replacement = TorchScaledMMFP8Linear.from_linear(
                    child,
                    out_dtype=out_dtype,
                )
            elif recipe.precision == "fp4":
                replacement = TorchScaledMMFP4Linear.from_linear(
                    child,
                    out_dtype=out_dtype,
                    use_rht=recipe.use_rht,
                    use_global_scale=recipe.use_global_scale,
                    weight_scale_2d=recipe.weight_scale_2d,
                )
            else:
                raise ValueError(f"Unsupported SANA-WM quant recipe: {recipe!r}.")
            setattr(module, name, replacement)
            converted += 1
            continue
        child_converted, child_skipped = _replace_linear_with_quant(
            child,
            recipe=recipe,
            params_dtype=params_dtype,
            skip_patterns=skip_patterns,
            include_patterns=include_patterns,
            prefix=child_prefix,
        )
        converted += child_converted
        skipped += child_skipped
    return converted, skipped


def _name_matches(patterns: tuple[str, ...], name: str) -> bool:
    return any(re.search(pattern, name) for pattern in patterns)


def _require_scaled_mm() -> None:
    if not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("torch._scaled_mm is required for SANA-WM quantization.")


def _require_fp8_dtype() -> None:
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError(
            "torch.float8_e4m3fn is required for the SANA-WM FP8 backend."
        )


def _require_fp4_dtype() -> None:
    if not hasattr(torch, "float4_e2m1fn_x2"):
        raise RuntimeError(
            "torch.float4_e2m1fn_x2 is required for the SANA-WM FP4 backend."
        )
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError(
            "torch.float8_e4m3fn scales are required for the SANA-WM FP4 backend."
        )


__all__ = [
    "TorchScaledMMFP4Linear",
    "TorchScaledMMFP4Recipe",
    "TorchScaledMMFP8Linear",
    "TorchScaledMMFP8Recipe",
    "apply_rht16",
    "nvfp4_global_scale",
    "quantize_nvfp4_swizzled",
    "replace_linear_with_quant",
    "replace_linear_with_torch_fp4",
    "replace_linear_with_torch_fp8",
]
