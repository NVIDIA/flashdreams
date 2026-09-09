# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint adaptation for FastVideo Causal Wan 2.2."""

import torch

from flashdreams.recipes.wan import wan_dit_state_dict_from_diffusers


def state_dict_transform(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remap an HF diffusers Wan 2.2 state-dict to the WanDiTNetwork layout."""
    return wan_dit_state_dict_from_diffusers(state_dict)
