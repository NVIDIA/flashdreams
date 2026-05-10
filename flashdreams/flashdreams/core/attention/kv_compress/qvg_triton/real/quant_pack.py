# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Ported from Quant-VideoGen:
# https://github.com/svg-project/Quant-VideoGen
# Source file: quant_videogen/real/quant_pack.py
# Source commit: 11ff61a4f744d125a1d27fa13e2b9ecb89e94e07

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

def get_configs():
    configs = [triton.Config({"BLOCK_D": 64}, num_stages=1, num_warps=1)]
    for BLOCK_D in [128, 256, 512]:
            for num_warps in [1, 2, 4]:
                for num_stages in [3, 4, 5]:
                    configs.append(
                        triton.Config({'BLOCK_D': BLOCK_D}, num_stages=num_stages, num_warps=num_warps)
                    )
    return configs

@triton.autotune(
    configs=get_configs(),
    key=["N", "D"],
)
@triton.heuristics(
    values={
        'SCALE_BLOCK_D': lambda args: triton.next_power_of_2(args['BLOCK_D'] // args['Q_BLOCK_SIZE']),
        'BLOCK_D_AFTER_PACK': lambda args: args['BLOCK_D'] // (8 // args['n_bits']),
    }
)
@triton.jit
def _quant_pack_kernel(
    X_ptr,
    Y_ptr,
    SY_ptr,
    D: tl.constexpr,
    SCALE_D: tl.constexpr,
    D_AFTER_PACK: tl.constexpr,
    n_bits: tl.constexpr,
    SCALE_IS_E4M3: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr, # quantization block size
    PACK_OUTPUT_INT8: tl.constexpr,
    # Autotune
    BLOCK_D: tl.constexpr,
    SCALE_BLOCK_D: tl.constexpr,
    BLOCK_D_AFTER_PACK: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    offset_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offset_d < D
    
    if PACK_OUTPUT_INT8:
        offset_d_for_output = pid_d * BLOCK_D_AFTER_PACK + tl.arange(0, BLOCK_D_AFTER_PACK)
        mask_for_output = offset_d_for_output < D_AFTER_PACK
    else:
        offset_d_for_output = offset_d
        mask_for_output = mask
    
    offset_scale_d = pid_d * SCALE_BLOCK_D + tl.arange(0, SCALE_BLOCK_D)
    mask_scale_d = offset_scale_d < SCALE_D

    # Load input
    x_ptr = X_ptr + pid_s * D + offset_d
    x = tl.load(x_ptr, mask=mask, other=0.0)
    

    # Integer Quantization
    x = x.to(tl.float32)
    x = tl.reshape(x, (SCALE_BLOCK_D, Q_BLOCK_SIZE))
    abs_x = tl.abs(x)
    max_x = tl.max(abs_x, axis=1, keep_dims=True)

    max_int_value = 2**(n_bits - 1) - 1
    scale_x = max_x / max_int_value
    scale_x = tl.maximum(scale_x, 1e-10)
    
    # If the scaling factor is also being quantized to E4M3
    if SCALE_IS_E4M3:
        scale_x = scale_x.to(tl.float8e4nv)
    
    # Float Quantization
    y = x / scale_x
    y = libdevice.round(y)
    y = tl.clamp(y, min=-max_int_value, max=max_int_value)
    
    # Deal with output shape
    y = tl.reshape(y, (BLOCK_D))
    y = y.to(tl.int32)
    scale_x = tl.reshape(scale_x, (SCALE_BLOCK_D))
    scale_x = scale_x.to(SY_ptr.dtype.element_ty)
    
    # Pack output into int8 if needed
    if PACK_OUTPUT_INT8:
        if n_bits == 4:
            y = y + max_int_value
            y = tl.reshape(y, (BLOCK_D // 2, 2))
            y1, y2 = tl.split(y)
            y_new = y1 << 4 | y2
            y = y_new
        elif n_bits == 2:
            y = y + max_int_value
            y = tl.reshape(y, (BLOCK_D // 2, 2))
            y13, y24 = tl.split(y)
            
            y13 = tl.reshape(y13, (BLOCK_D // 4, 2))
            y24 = tl.reshape(y24, (BLOCK_D // 4, 2))
            
            y1, y3 = tl.split(y13)
            y2, y4 = tl.split(y24)
            y_new = y1 << 6 | y2 << 4 | y3 << 2 | y4
            y = y_new
            
    y = y.to(Y_ptr.dtype.element_ty)
    
    # tl.device_print("HD After Pack: ", HD_AFTER_PACK)
        
    # Store output
    y_ptr = Y_ptr + pid_s * D_AFTER_PACK + offset_d_for_output
    tl.store(y_ptr, y, mask=mask_for_output)
    
    scale_y_ptr = SY_ptr + pid_s * SCALE_D + offset_scale_d
    tl.store(scale_y_ptr, scale_x, mask=mask_scale_d)
    

def quant_pack(
    x: torch.Tensor,
    block_size: int,
    num_bits: int,
    scale_precision: torch.dtype,
    pack_output_int8: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize input tensor with block-wise scaling.

    Performs block-wise quantization on the input tensor. Each block of size
    `block_size` along the last dimension is quantized with a shared scale factor.

    Args:
        x: Tensor of shape (B, H, S, D), the input tensor to quantize.
           D should be divisible by block_size.
        block_size: The size of each block for quantization.
        num_bits: The number of bits for quantization.
        scale_precision: The dtype for storing scale factors. Must be bfloat16 or float8_e4m3fn.
        pack_output_int8: If True, pack the output into int8. In this case, num_bits must be 2 or 4.
        
    Returns:
        x_quant: Quantized tensor.
        scales: Scale factors with dtype `scale_precision`, shape (..., D // block_size).
    """
    assert num_bits in (2, 3, 4, 8), "num_bits must be 2, 3, 4, or 8"
    assert scale_precision in (torch.bfloat16, torch.float8_e4m3fn), "scale_precision must be bfloat16 or float8_e4m3fn"
    if pack_output_int8:
        assert num_bits in (2, 4), "num_bits must be 2 or 4 when pack_output_int8 is True"
    
    B, H, S, D = x.shape
    SCALE_D = D // block_size
    
    HD = H * D
    HSCALE_D = H * SCALE_D
    
    SCALE_IS_E4M3 = scale_precision == torch.float8_e4m3fn
    PACK_OUTPUT_INT8 = pack_output_int8 and num_bits in (2, 4)
    
    # First, reshape to (B * S, H * D) while keeping the original shape
    x = x.permute(0, 2, 1, 3).reshape(B * S, H * D).contiguous()
    
    # Define Outputs
    if PACK_OUTPUT_INT8:
        elem_per_int = 8 // num_bits
        HD_AFTER_PACK = HD // elem_per_int
        y = torch.zeros(B * S, HD // elem_per_int, device=x.device, dtype=torch.uint8)
    else:
        y = torch.zeros(B * S, HD, device=x.device, dtype=torch.int8)
        HD_AFTER_PACK = HD
    scales = torch.zeros(B * S, HSCALE_D, device=x.device, dtype=scale_precision)

    grid = lambda meta: (B * S, triton.cdiv(HD, meta["BLOCK_D"]))
    _quant_pack_kernel[grid](
        x, y, scales, HD, HSCALE_D, HD_AFTER_PACK, num_bits, SCALE_IS_E4M3, block_size, PACK_OUTPUT_INT8)
    
    # Reshape to original shape
    if PACK_OUTPUT_INT8:
        y = y.reshape(B, S, H, D // elem_per_int).permute(0, 2, 1, 3)
    else:
        y = y.reshape(B, S, H, D).permute(0, 2, 1, 3)
    scales = scales.reshape(B, S, H, SCALE_D).permute(0, 2, 1, 3)
    
    return y, scales


if __name__ == "__main__":
    x = torch.randn(1, 4, 1, 128, device="cuda").to(torch.bfloat16)
    y, scales = quant_pack(x, block_size=16, num_bits=4, scale_precision=torch.float8_e4m3fn, pack_output_int8=True)
    print(y)
    print(scales)
