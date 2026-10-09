# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch module helpers shared by OmniDreams runtime features."""

from __future__ import annotations

import torch.nn as nn


def unwrap_compiled_module(module: nn.Module) -> nn.Module:
    """Return the original module stored by a ``torch.compile`` wrapper.

    Raises:
        TypeError: ``_orig_mod`` exists but is not an ``nn.Module``.
    """
    if not hasattr(module, "_orig_mod"):
        return module
    original = module._orig_mod
    if not isinstance(original, nn.Module):
        raise TypeError(
            f"{type(module).__name__}._orig_mod must be an nn.Module, "
            f"got {type(original).__name__}"
        )
    return original


__all__ = ["unwrap_compiled_module"]
