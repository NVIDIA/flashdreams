# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional latent-codec seam for the pixel path.

Lets a codec sit between the DiT and the pixel decoder so the decoder sees exactly the
latents a remote client would have reconstructed, rather than the pristine ones. That is
what makes an offline quality comparison possible without a browser: the token-streaming
path needs WebGPU on the client, which is unavailable here.

BUFFER CONTRACT
---------------
``roundtrip`` MUST return a tensor whose storage is disjoint from its input. The input
buffer is shared -- ``cache.clean_latent`` is the same tensor object, and it is what the
token stream emits -- so writing through it would corrupt the token path and double-apply
quantization. ``pipeline/base.py`` asserts disjointness on every call rather than trusting
this comment; note that comparing ``data_ptr()`` is NOT sufficient, since two tensors can
share one storage at different offsets.

The DiT's own feedback tensor (``final_state.clean_latent``) lives in separate storage and
is unaffected either way, so the generation trajectory is identical with a codec active.
That makes this an open-loop comparison: reconstruction error stays bounded per frame
instead of compounding through the autoregressive loop.

Select with the ``FD_LATENT_CODEC`` environment variable: ``identity`` or ``sas``.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch import Tensor


class LatentCodec(Protocol):
    """A codec that round-trips a latent through compression and back."""

    name: str

    def roundtrip(self, latent: Tensor) -> Tensor:
        """Return a NEW tensor: the latent as a decoder would reconstruct it."""
        ...


class IdentityLatentCodec:
    """Copies the latent and changes nothing.

    The control for the seam itself. With this active the decoded video must be
    bit-identical to a run with no codec at all -- any difference means the plumbing is
    wrong, and that is worth knowing before a lossy codec makes small differences
    expected. It also exercises the disjoint-storage assertion on every step.
    """

    name = "identity"

    def roundtrip(self, latent: Tensor) -> Tensor:
        return latent.clone()


def get_latent_codec(name: str | None) -> LatentCodec | None:
    """Resolve a codec by name. ``None``/empty disables the seam entirely."""
    if not name:
        return None
    key = name.lower()
    if key == "identity":
        return IdentityLatentCodec()
    if key == "sas":
        # Imported lazily: it pulls in the vendored SAS package and Triton, which
        # should not be a hard dependency of the pipeline.
        from flashdreams.infra.pipeline.latent_sas import SASLatentCodec

        return SASLatentCodec()
    raise ValueError(f"unknown latent codec {name!r}; expected one of: identity, sas")
