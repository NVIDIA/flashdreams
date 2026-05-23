

"""Phase 2b.6.2 VAE-sample-vs-mean parity probe.

Monkey-patches vendor's ``DiagonalGaussianDistribution.sample`` to
return ``self.mean`` instead of ``mean + std * randn(...)``. The
vendor pipeline calls ``self.vae.encode(first_image, return_dict=False)[0].sample()``
at AR-step setup; the stochastic ``.sample()`` draws ``randn_like(mean)``
from the global RNG and adds it scaled by ``std`` to the mean. The
distilled WAN-5B encoder's posterior is reasonably tight so the
noise term is small, but at the 2-chunk 704x1280 reference config it
contributes a ~0.008 abs_mean shift to the image latent that the
diffusion network amplifies into ~12-13 / 255 mean |\u0394| at the
chunk-0 video frames (and, via the reconstituted-context memory
prefill, ~5 / 255 of additional drift at chunk-1+).

Flashdreams' :class:`~flashdreams.recipes.wan.autoencoder.vae.WanVAE`
encoder returns the deterministic mean directly (no ``.sample()``);
this patch forces vendor to do the same so the diff is apples-to-apples.

Set ``HY_VENDOR_VAE_MEAN=1`` together with this patch installed to
get a vendor baseline that uses the mean-only path. Running the dump
harness with this on + native HY-WorldPlay gives an upper bound on
how much of the residual chunk-0 / chunk-1 drift is sample-noise vs.
genuine numerical divergence in the rest of the stack.
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

    def _mean_only_sample(
        self: Any, generator: Any = None
    ) -> Any:
        return self.mean

    DiagonalGaussianDistribution.sample = _mean_only_sample  # type: ignore[method-assign]
    print(
        "[vae_mean_patch] DiagonalGaussianDistribution.sample -> mean (no std*randn)",
        flush=True,
    )
