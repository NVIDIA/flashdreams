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

"""CUDA tests for SANA-WM Stage-1 kernels."""

from __future__ import annotations

import pytest
import sana_wm.stage1_model as stage1
import torch

pytestmark = pytest.mark.ci_gpu


def _require_stage1_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for Stage-1 kernel tests")
    if stage1.triton is None:
        pytest.skip("Triton is required for the Stage-1 temporal conv CUDA kernel")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bidirectional_temporal_conv_fast_path_matches_eager(
    dtype: torch.dtype,
) -> None:
    """Compare the CUDA temporal-conv fast path against the eager conv path."""
    _require_stage1_cuda()
    torch.manual_seed(1234)
    batch, frames, height, width, channels = 2, 5, 3, 4, 16
    x = torch.randn(
        batch,
        frames * height * width,
        channels,
        device="cuda",
        dtype=dtype,
    )
    conv = torch.nn.Conv1d(
        channels,
        channels,
        kernel_size=4,
        groups=channels,
        bias=False,
        device="cuda",
        dtype=dtype,
    )

    reference = stage1._apply_bidirectional_temporal_conv_eager(
        x,
        conv,
        (frames, height, width),
    )
    with torch.inference_mode():
        actual = stage1._apply_bidirectional_temporal_conv(
            x,
            conv,
            (frames, height, width),
        )

    assert actual.shape == x.shape
    assert actual.dtype == dtype
    torch.testing.assert_close(
        actual.float(),
        reference.float(),
        rtol=2e-2,
        atol=8e-2 if dtype is torch.bfloat16 else 1e-5,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("inverse_rope", [False, True])
@pytest.mark.parametrize("matrix_batch", [1, 2])
def test_ucpe_transform_apply_fast_path_matches_eager(
    dtype: torch.dtype,
    inverse_rope: bool,
    matrix_batch: int,
) -> None:
    """Compare the CUDA UCPE transform fast path against the eager path."""
    _require_stage1_cuda()
    torch.manual_seed(4321)
    batch, frames, height, width, heads, dim = 2, 3, 2, 3, 2, 16
    tokens = frames * height * width
    x = torch.randn(
        batch,
        tokens,
        heads,
        dim,
        device="cuda",
        dtype=dtype,
    )
    matrix = torch.randn(
        matrix_batch,
        tokens,
        4,
        4,
        device="cuda",
        dtype=dtype,
    )
    rope = stage1._slice_rope_for_camera(
        stage1._wan_rope_complex(dim, frames, height, width, x.device),
        dim,
    )

    reference = stage1._ucpe_transform_apply_eager(
        x,
        matrix,
        rope,
        inverse_rope=inverse_rope,
    )
    with torch.inference_mode():
        actual = stage1._ucpe_transform_apply(
            x,
            matrix,
            rope,
            inverse_rope=inverse_rope,
        )

    assert actual.shape == x.shape
    assert actual.dtype is torch.float32
    torch.testing.assert_close(
        actual,
        reference,
        rtol=2e-2 if dtype is torch.bfloat16 else 1e-5,
        atol=8e-2 if dtype is torch.bfloat16 else 1e-5,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("inverse_rope", [False, True])
@pytest.mark.parametrize("matrix_batch", [1, 2])
def test_ucpe_transform_fast_path_matches_eager_norms(
    dtype: torch.dtype,
    inverse_rope: bool,
    matrix_batch: int,
) -> None:
    """Compare the UCPE fast path including GDN norm buffers."""
    _require_stage1_cuda()
    torch.manual_seed(2468)
    batch, frames, height, width, heads, dim = 2, 3, 2, 3, 2, 16
    tokens = frames * height * width
    x = torch.randn(
        batch,
        tokens,
        heads,
        dim,
        device="cuda",
        dtype=dtype,
    )
    matrix = torch.randn(
        matrix_batch,
        tokens,
        4,
        4,
        device="cuda",
        dtype=dtype,
    )
    rope = stage1._slice_rope_for_camera(
        stage1._wan_rope_complex(dim, frames, height, width, x.device),
        dim,
    )

    reference = stage1._ucpe_transform_apply_eager(
        x,
        matrix,
        rope,
        inverse_rope=inverse_rope,
    )
    reference_pre = stage1._ucpe_transform_norms_eager(x)
    reference_post = stage1._ucpe_transform_norms_eager(reference)
    with torch.inference_mode():
        actual, actual_pre, actual_post = stage1._ucpe_transform(
            x,
            matrix,
            rope,
            inverse_rope=inverse_rope,
        )

    assert actual.shape == x.shape
    assert actual_pre.shape == (batch, heads, tokens)
    assert actual_post.shape == (batch, heads, tokens)
    torch.testing.assert_close(
        actual,
        reference,
        rtol=2e-2 if dtype is torch.bfloat16 else 1e-5,
        atol=8e-2 if dtype is torch.bfloat16 else 1e-5,
    )
    torch.testing.assert_close(
        actual_pre,
        reference_pre,
        rtol=2e-2 if dtype is torch.bfloat16 else 1e-5,
        atol=8e-2 if dtype is torch.bfloat16 else 1e-5,
    )
    torch.testing.assert_close(
        actual_post,
        reference_post,
        rtol=2e-2 if dtype is torch.bfloat16 else 1e-5,
        atol=8e-2 if dtype is torch.bfloat16 else 1e-5,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_silu_multiply_fast_path_matches_eager(dtype: torch.dtype) -> None:
    """Compare the fused GLU elementwise fast path against PyTorch eager."""
    _require_stage1_cuda()
    torch.manual_seed(1357)
    value = torch.randn(
        4,
        17,
        5,
        7,
        device="cuda",
        dtype=dtype,
    ).contiguous(memory_format=torch.channels_last)
    gate = torch.randn_like(value).contiguous(memory_format=torch.channels_last)

    reference = value * torch.nn.functional.silu(gate)
    with torch.inference_mode():
        actual = stage1._silu_multiply(value, gate, inplace=False)

    assert actual.shape == value.shape
    assert actual.dtype == value.dtype
    assert actual.is_contiguous(memory_format=torch.channels_last)
    torch.testing.assert_close(
        actual.float(),
        reference.float(),
        rtol=2e-2 if dtype is torch.bfloat16 else 1e-5,
        atol=8e-2 if dtype is torch.bfloat16 else 1e-5,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("scale", [1.0, 0.0375])
def test_rmsnorm_relu_heads_fast_path_matches_eager(
    dtype: torch.dtype,
    scale: float,
) -> None:
    """Compare fused GDN RMSNorm/ReLU preparation against eager math."""
    _require_stage1_cuda()
    torch.manual_seed(9753)
    batch, tokens, heads, dim = 2, 13, 4, 8
    x = torch.randn(
        batch,
        tokens,
        heads,
        dim,
        device="cuda",
        dtype=dtype,
    )
    weight = torch.randn(heads * dim, device="cuda", dtype=dtype)
    eps = 1e-6
    inv_rms = stage1._inv_rms(x, eps)
    reference = torch.nn.functional.relu(
        x.float() * inv_rms[:, :, None, None] * weight.float().view(heads, dim)
    )
    if scale != 1.0:
        reference = reference * scale

    with torch.inference_mode():
        actual = stage1._rmsnorm_relu_heads(x, weight, eps, scale=scale)

    assert actual.shape == x.shape
    assert actual.dtype is torch.float32
    torch.testing.assert_close(
        actual,
        reference,
        rtol=2e-2 if dtype is torch.bfloat16 else 1e-5,
        atol=8e-2 if dtype is torch.bfloat16 else 1e-5,
    )
