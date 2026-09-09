# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint adaptation for Wan 2.2 TI2V-5B."""

import torch

from flashdreams.recipes.wan import wan_dit_state_dict_from_diffusers


def wan22_ti2v_5b_dit_state_dict_transform(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remap a diffusers Wan 2.2 TI2V-5B checkpoint to WanDiTNetwork keys."""
    return wan_dit_state_dict_from_diffusers(state_dict)
