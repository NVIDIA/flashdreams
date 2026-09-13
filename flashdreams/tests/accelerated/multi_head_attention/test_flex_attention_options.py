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

"""CPU tests for shared FlexAttention mask and kernel configuration."""

from __future__ import annotations

import pytest

from flashdreams.accelerated.multi_head_attention import (
    FlexAttentionOptions as PublicFlexAttentionOptions,
)
from flashdreams.accelerated.multi_head_attention.flex import FlexAttentionOptions
from flashdreams.accelerated.multi_head_attention.functional import (
    FlexAttentionOptions as FunctionalFlexAttentionOptions,
)
from flashdreams.accelerated.multi_head_attention.optimized import (
    FlexAttentionOptions as ModuleFlexAttentionOptions,
)

pytestmark = pytest.mark.ci_cpu


def test_functional_and_module_apis_export_one_configuration_type() -> None:
    assert PublicFlexAttentionOptions is FlexAttentionOptions
    assert FunctionalFlexAttentionOptions is FlexAttentionOptions
    assert ModuleFlexAttentionOptions is FlexAttentionOptions


def test_mask_geometry_defaults_to_a_square() -> None:
    assert FlexAttentionOptions(block_size=64).mask_block_size == 64


def test_mask_geometry_is_independent_of_kernel_tiles() -> None:
    options = FlexAttentionOptions(
        block_size=128,
        mask_block_m=64,
        mask_block_n=32,
        block_m=32,
        block_n=64,
        num_warps=4,
        num_stages=2,
        prescale_qk=True,
        use_tma=False,
        backend="TRITON",
        rows_guaranteed_safe=True,
    )

    assert options.mask_block_size == (64, 32)
    assert options.kernel_options == {
        "BLOCK_M": 32,
        "BLOCK_N": 64,
        "num_warps": 4,
        "num_stages": 2,
        "PRESCALE_QK": True,
        "USE_TMA": False,
        "BACKEND": "TRITON",
        "ROWS_GUARANTEED_SAFE": True,
    }


def test_each_mask_dimension_can_override_the_square_independently() -> None:
    assert FlexAttentionOptions(mask_block_m=64).mask_block_size == (64, 128)
    assert FlexAttentionOptions(mask_block_n=64).mask_block_size == (128, 64)


@pytest.mark.parametrize(
    "field", ["block_size", "mask_block_m", "mask_block_n", "block_m", "block_n"]
)
def test_invalid_block_dimensions_are_refused(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        FlexAttentionOptions(**{field: 48})  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("field", ["compile_dynamic", "prescale_qk", "use_tma"])
def test_tri_state_options_do_not_accept_integer_booleans(field: str) -> None:
    with pytest.raises(TypeError, match="bool or None"):
        FlexAttentionOptions(**{field: 1})  # ty: ignore[invalid-argument-type]


def test_kernel_backend_is_validated_before_compilation() -> None:
    with pytest.raises(ValueError, match="TRITON, FLASH"):
        FlexAttentionOptions(backend="fastest")
