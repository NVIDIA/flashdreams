"""Torch fallback for official QVG residual packing on non-E4M3 Triton targets."""

from __future__ import annotations

import torch


def _pack_int4(y: torch.Tensor) -> torch.Tensor:
    y = y.to(torch.int16) + 7
    y = y.reshape(*y.shape[:-1], y.shape[-1] // 2, 2)
    return ((y[..., 0] << 4) | y[..., 1]).to(torch.uint8)


def _pack_int2(y: torch.Tensor) -> torch.Tensor:
    y = y.to(torch.int16) + 1
    y = y.reshape(*y.shape[:-1], y.shape[-1] // 2, 2)
    y13 = y[..., 0]
    y24 = y[..., 1]
    y13 = y13.reshape(*y13.shape[:-1], y13.shape[-1] // 2, 2)
    y24 = y24.reshape(*y24.shape[:-1], y24.shape[-1] // 2, 2)
    y1 = y13[..., 0]
    y3 = y13[..., 1]
    y2 = y24[..., 0]
    y4 = y24[..., 1]
    return ((y1 << 6) | (y2 << 4) | (y3 << 2) | y4).to(torch.uint8)


def quant_pack(
    x: torch.Tensor,
    block_size: int,
    num_bits: int,
    scale_precision: torch.dtype,
    pack_output_int8: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert num_bits in (2, 3, 4, 8), "num_bits must be 2, 3, 4, or 8"
    assert scale_precision in (
        torch.bfloat16,
        torch.float8_e4m3fn,
    ), "scale_precision must be bfloat16 or float8_e4m3fn"
    if pack_output_int8:
        assert num_bits in (2, 4), "num_bits must be 2 or 4 when pack_output_int8 is True"

    if scale_precision == torch.float8_e4m3fn:
        scale_precision = torch.bfloat16

    batch, heads, seq, dim = x.shape
    assert dim % block_size == 0, "last dimension must be divisible by block_size"
    max_int_value = 2 ** (num_bits - 1) - 1

    x_blocks = x.to(torch.float32).reshape(batch, heads, seq, dim // block_size, block_size)
    scales = x_blocks.abs().amax(dim=-1).div(max_int_value).clamp_min(1e-10)
    y = torch.round(x_blocks / scales.unsqueeze(-1))
    y = y.clamp(min=-max_int_value, max=max_int_value).to(torch.int8)
    y = y.reshape(batch, heads, seq, dim)

    if pack_output_int8:
        if num_bits == 4:
            y_out = _pack_int4(y)
        elif num_bits == 2:
            y_out = _pack_int2(y)
        else:
            raise AssertionError("unreachable")
    else:
        y_out = y

    return y_out.contiguous(), scales.to(scale_precision).contiguous()
