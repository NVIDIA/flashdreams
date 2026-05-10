# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for QVG config parity with Self-Forcing BF16."""

from flashdreams.recipes.wan.config.causal_wan21 import (
    build_self_forcing,
    build_self_forcing_qvg_int2,
    build_self_forcing_qvg_int4,
)
from flashdreams.core.attention.kv_compress import QVGQuantConfig


def test_qvg_quant_config_defaults_to_official_triton() -> None:
    assert QVGQuantConfig().kernel_impl == "official_triton"


def test_qvg_int2_uses_self_forcing_scheduler_shift() -> None:
    bf16 = build_self_forcing(compile_network=False)
    qvg_int2 = build_self_forcing_qvg_int2(compile_network=False)
    qvg_int4 = build_self_forcing_qvg_int4(compile_network=False)

    expected_shift = bf16.diffusion_model.scheduler.shift
    assert expected_shift == 8.0
    assert qvg_int2.diffusion_model.scheduler.shift == expected_shift
    assert qvg_int4.diffusion_model.scheduler.shift == expected_shift


def test_qvg_int2_uses_self_forcing_noise_layout() -> None:
    bf16 = build_self_forcing(compile_network=False)
    qvg_int2 = build_self_forcing_qvg_int2(compile_network=False)

    assert qvg_int2.diffusion_model._noise_in_unpatchified_shape == (
        bf16.diffusion_model._noise_in_unpatchified_shape
    )
