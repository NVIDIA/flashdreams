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

"""Numerical parity for the fused bicubic + pixel-shuffle CUDA kernel.

Compares both kernel outputs against the eager PyTorch path for every
``(T_raw, scale)`` pair in
``FLASHVSR_CHUNK_FRAME_TARGETS``: the projector's post-pixel-shuffle
``proj_input`` and the un-padded ``last_upres``. Bit-exact match isn't
expected because PyTorch's ``F.interpolate(mode="bicubic")`` internally
accumulates fp32 in a specific order and our kernel computes the same
math but with potentially different rounding -- the tolerance is set at
~1 ULP of bf16, tight enough to catch real index-math regressions while
absorbing the fp32-rounding noise.

Marker convention (see
``agentic/skills/flashdreams-recipe-architecture/SKILL.md`` section on
``pytest-manual-marker``): ``@pytest.mark.manual`` would unconditionally
xfail the test even with ``-m manual``, so we use only
``skipif(not torch.cuda.is_available())``. Skipped automatically when
CUDA is missing or the extension fails to load.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange

from flashdreams.recipes.flashvsr import encoder as encoder_module
from flashdreams.recipes.flashvsr.constants import FLASHVSR_CHUNK_FRAME_TARGETS

_GPU_REASON = "fused bicubic + pixel-shuffle kernel requires CUDA"


def _eager_reference(
    input: torch.Tensor,
    target_T: int,
    target_H: int,
    target_W: int,
    n_left_padding: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference: eager F.pad + F.interpolate + PixelShuffle3d.

    Mirrors the eager fallback in
    :meth:`flashdreams.recipes.flashvsr.encoder.FlashVSREncoder.forward`
    plus the projector's ``self.pixel_shuffle(video)`` call. Returns
    ``(proj_input, last_upres)``.
    """
    B, _3, T_raw, H, W = input.shape
    if n_left_padding > 0:
        padded = F.pad(input, (0, 0, 0, 0, n_left_padding, 0), mode="replicate")
    else:
        padded = input
    upres = (
        F.interpolate(
            padded.permute(0, 2, 1, 3, 4).reshape(B * target_T, 3, H, W),
            size=(target_H, target_W),
            mode="bicubic",
            align_corners=False,
        )
        .view(B, target_T, 3, target_H, target_W)
        .permute(0, 2, 1, 3, 4)
    )
    last_upres = upres[:, :, n_left_padding:, :, :].contiguous()
    proj_input = rearrange(
        upres,
        "b c (f ff) (h hh) (w ww) -> b (c ff hh ww) f h w",
        ff=1,
        hh=16,
        ww=16,
    ).contiguous()
    return proj_input, last_upres


# (T_raw, scale, dtype, atol, rtol) rows. Tolerances:
#   - bf16: 1 ULP of bf16 around |x|<=1 is 2^-7 ~ 7.8e-3. The accumulation
#     order between our kernel (single fp32 separable 4x4 pass) and
#     PyTorch's may differ, plus the final fp32->bf16 rounding can land
#     on different ties. atol=1/128, rtol=1/64 covers it without
#     hiding index-math bugs.
#   - fp32: kernel does the same separable cubic_convolution math as
#     ATen, but ordering can still differ at the FMA level. 1e-5 / 1e-5
#     is the standard "fp32 reordered accumulation" tolerance used
#     elsewhere in this repo.
_PARITY_CASES = [
    pytest.param(t, s, dtype, atol, rtol, id=f"T{t}-scale{s}-{name}")
    for (dtype, atol, rtol, name) in [
        (torch.bfloat16, 1.0 / 128.0, 1.0 / 64.0, "bf16"),
        (torch.float32, 1e-5, 1e-5, "fp32"),
    ]
    for t in sorted(FLASHVSR_CHUNK_FRAME_TARGETS)
    for s in (2, 4)
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
@pytest.mark.parametrize(("t_raw", "scale", "dtype", "atol", "rtol"), _PARITY_CASES)
@torch.no_grad()
def test_fused_kernel_matches_eager(
    t_raw: int,
    scale: int,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
) -> None:
    """Kernel ``proj_input`` and ``last_upres`` match the eager reference.

    Smallest valid input dims for each scale (target divisible by 128
    via the encoder's setup assertion): scale=2 picks (64, 64) ->
    (128, 128); scale=4 picks (32, 32) -> (128, 128). 128 also divides
    by the projector's 16x spatial pixel-shuffle, so the kernel and the
    eager reference both produce 8x8 ``proj_input`` spatial dims.
    """
    if dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("GPU does not support bfloat16")

    ext = encoder_module._load_bicubic_pixelshuffle_extension()
    if ext is None:
        pytest.skip(
            f"fused bicubic + pixel-shuffle extension unavailable: "
            f"{encoder_module._BICUBIC_PIXELSHUFFLE_LOAD_ERROR}"
        )

    target_T = FLASHVSR_CHUNK_FRAME_TARGETS[t_raw]
    n_left_padding = target_T - t_raw

    if scale == 2:
        input_H, input_W = 64, 64
    else:
        input_H, input_W = 32, 32
    target_H = input_H * scale
    target_W = input_W * scale
    assert target_H % 128 == 0 and target_W % 128 == 0
    assert target_H % 16 == 0 and target_W % 16 == 0

    torch.manual_seed(0xF1A57)
    B = 1
    input = (
        torch.rand(B, 3, t_raw, input_H, input_W, device="cuda", dtype=dtype) * 2.0
        - 1.0
    ).contiguous()

    proj_kernel, last_upres_kernel = ext.bicubic_pixelshuffle_forward(
        input, target_T, target_H, target_W, n_left_padding
    )
    proj_eager, last_upres_eager = _eager_reference(
        input, target_T, target_H, target_W, n_left_padding
    )

    assert proj_kernel.shape == proj_eager.shape, (
        f"proj_input shape mismatch: kernel {tuple(proj_kernel.shape)} "
        f"vs eager {tuple(proj_eager.shape)}"
    )
    assert last_upres_kernel.shape == last_upres_eager.shape, (
        f"last_upres shape mismatch: kernel {tuple(last_upres_kernel.shape)} "
        f"vs eager {tuple(last_upres_eager.shape)}"
    )
    assert proj_kernel.dtype == proj_eager.dtype == dtype
    assert last_upres_kernel.dtype == last_upres_eager.dtype == dtype

    # Compute diffs in fp32 so the bf16 rows aren't dominated by their
    # own rounding error during the comparison.
    proj_diff = (proj_kernel.float() - proj_eager.float()).abs()
    lu_diff = (last_upres_kernel.float() - last_upres_eager.float()).abs()
    proj_max = proj_diff.max().item()
    lu_max = lu_diff.max().item()

    print(
        f"T_raw={t_raw} scale={scale} dtype={dtype} "
        f"proj_max_abs={proj_max:.6g} last_upres_max_abs={lu_max:.6g}"
    )

    torch.testing.assert_close(
        proj_kernel,
        proj_eager,
        atol=atol,
        rtol=rtol,
        msg=lambda s: f"proj_input parity failed: {s}",
    )
    torch.testing.assert_close(
        last_upres_kernel,
        last_upres_eager,
        atol=atol,
        rtol=rtol,
        msg=lambda s: f"last_upres parity failed: {s}",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason=_GPU_REASON)
def test_extension_loads() -> None:
    """The fused kernel extension JIT-builds + exposes the expected symbol."""
    ext = encoder_module._load_bicubic_pixelshuffle_extension()
    if ext is None:
        pytest.skip(
            f"fused bicubic + pixel-shuffle extension unavailable: "
            f"{encoder_module._BICUBIC_PIXELSHUFFLE_LOAD_ERROR}"
        )
    assert hasattr(ext, "bicubic_pixelshuffle_forward")
