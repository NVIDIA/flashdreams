# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SAS (Scale-Adaptive Scalar / PRQ) token codec � pure-PyTorch, no Triton.

Compresses each generated latent frame with the same progressive-residual
quantizer used offline (``flashdreams.infra.sas._compress_latents``): per-frame
k-means over ``num_stages`` stages plus a block-wise bit-packed integer residual.
The browser (``codec/sas.js``) reverses it byte-for-byte and feeds the
reconstructed latent to the LightTAE ONNX decoder exactly like the raw codec.

Sits beside the raw float16 reference codec on the token-streaming path; the
client selects it per session via ``?codec=sas-int4-2s`` / ``sas-int4-4s``.

WIRE FORMAT (per frame, little-endian; mirrored in codec/sas.js)
---------------------------------------------------------------
Shapes for the OmniDreams latent (C=16, H=88, W=160): C_padded=max(C,16)=16,
S=H*W=14080, block_size=min(C_padded,16)=16.

    [ num_stages x centroids ]  each  k * C_padded  float32   (k = num_clusters)
    [ num_stages x cluster_ids] each  S             uint8
    [ residual_quant          ]        S * C_padded * num_bits / 8   uint8
    [ scales                  ]        S * (C_padded / block_size)   float16

At int4-2s that is 201,728 B/frame (403,456 B/2-frame chunk); int4-4s is
262,656 B/frame (525,312 B/chunk) � a ~2.2x / ~1.7x shrink vs raw float16.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from flashdreams.infra.sas import _compress_latents
from flashdreams.serving.token_stream.codec.base import (
    TokenCodec,
    TokenCodecConfig,
    TokenCodecEncodeResult,
)


@dataclass(kw_only=True)
class SASTokenCodecConfig(TokenCodecConfig):
    """Config for the SAS token codec.

    ``num_bits`` is the residual bit width (2/4/8), ``num_stages`` the number of
    k-means residual stages, ``num_clusters`` the k-means codebook size (<=256 so
    cluster ids fit in a uint8).
    """

    num_bits: int = 4
    num_stages: int = 2
    num_clusters: int = 256
    _target: type[TokenCodec] = field(default_factory=lambda: SASTokenCodec)


class SASTokenCodec(TokenCodec[SASTokenCodecConfig]):
    """Encode latent frames with the pure-PyTorch SAS/PRQ quantizer."""

    @property
    def codec_id(self) -> str:
        """Identifier advertised in the session header (client registry key)."""
        return "sas"

    @property
    def static_params(self) -> dict:
        """Constant params the client needs to size and parse the wire payload.

        C_padded / block_size / S are derived on the client from ``latent_shape``
        in the session header; only the quantizer settings travel here.
        """
        return {
            "num_bits": self.config.num_bits,
            "num_stages": self.config.num_stages,
            "num_clusters": self.config.num_clusters,
        }

    def encode_frame(self, latent: torch.Tensor) -> TokenCodecEncodeResult:
        """Compress one ``[C, H, W]`` latent frame into the SAS wire payload."""
        c, h, w = latent.shape
        state = _compress_latents(
            latent.reshape(1, c, h, w),
            num_bits=self.config.num_bits,
            num_stages=self.config.num_stages,
            num_clusters=self.config.num_clusters,
        )
        c_padded = state["_meta"]["C_padded"]

        parts: list[bytes] = []
        # Stage centroids: k x C_padded float32, one block per stage.
        for centroids in state["centroids_list"]:
            parts.append(
                centroids.reshape(-1, c_padded)
                .to(torch.float32)
                .contiguous()
                .cpu()
                .numpy()
                .tobytes()
            )
        # Cluster ids: S uint8 per stage.
        for ids in state["cluster_ids_list"]:
            parts.append(
                ids.reshape(-1).to(torch.uint8).contiguous().cpu().numpy().tobytes()
            )
        # Packed integer residual (uint8) then per-block scales (float16).
        parts.append(
            state["residual_quant"]
            .reshape(-1)
            .to(torch.uint8)
            .contiguous()
            .cpu()
            .numpy()
            .tobytes()
        )
        parts.append(
            state["scales"]
            .reshape(-1)
            .to(torch.float16)
            .contiguous()
            .cpu()
            .numpy()
            .tobytes()
        )
        return TokenCodecEncodeResult(payload=b"".join(parts), frame_params=b"")
