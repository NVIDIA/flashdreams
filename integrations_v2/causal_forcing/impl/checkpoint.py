# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint adaptation for Causal Forcing."""

from typing import Any

from torch import Tensor


def state_dict_transform(state_dict: dict[str, Any]) -> dict[str, Tensor]:
    """Strip Causal-Forcing wrapper prefixes from the checkpoint state-dict."""
    if "generator_ema" in state_dict:
        state_dict = state_dict["generator_ema"]
    elif "generator" in state_dict:
        state_dict = state_dict["generator"]

    out: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            key = key[len("model.") :]
        elif key.startswith("net."):
            key = key[len("net.") :]
        if key.startswith("_fsdp_wrapped_module."):
            key = key[len("_fsdp_wrapped_module.") :]
        out[key] = value
    return out
