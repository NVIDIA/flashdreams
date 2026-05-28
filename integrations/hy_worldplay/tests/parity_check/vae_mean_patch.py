# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monkey-patch vendor VAE sampling to return the mean (parity diagnostic).

Vendor's ``DiagonalGaussianDistribution.sample()`` draws
``mean + std * randn(...)``; flashdreams' :class:`WanVAE` returns the
deterministic mean. Setting ``HY_VENDOR_VAE_MEAN=1`` and installing
this patch forces vendor onto the mean-only path so vendor-vs-native
diffs are apples-to-apples (no VAE sample noise floor).
"""

from __future__ import annotations

import os
from typing import Any


def enabled() -> bool:
    return os.environ.get("HY_VENDOR_VAE_MEAN", "") == "1"


def install_vae_mean_patch() -> None:
    if not enabled():
        return
    try:
        from diffusers.models.autoencoders.vae import (
            DiagonalGaussianDistribution,
        )
    except ImportError as exc:
        raise RuntimeError(
            "HY_VENDOR_VAE_MEAN=1 but diffusers is not importable; "
            "the patch targets diffusers' DiagonalGaussianDistribution."
        ) from exc

    def _mean_only_sample(self: Any, generator: Any = None) -> Any:
        return self.mean

    DiagonalGaussianDistribution.sample = _mean_only_sample  # type: ignore[method-assign]
    print(
        "[vae_mean_patch] DiagonalGaussianDistribution.sample -> mean (no std*randn)",
        flush=True,
    )
