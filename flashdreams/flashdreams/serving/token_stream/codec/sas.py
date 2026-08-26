# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SAS token codec: cascaded PRQ over the Cosmos DiT latents.

Replaces the raw float16 reference codec on the token-streaming path. The server encodes
each generated latent frame with SAS; the browser decodes it (``codec/sas.js``) and feeds
the result to the LightTAE ONNX decoder exactly as before.

USES THE REFERENCE IMPLEMENTATION UNMODIFIED
--------------------------------------------
All quantization is delegated to ``sol_media_compression.sas.prq``, the Triton
implementation already validated across the Cosmos CI8x8 and Wan studies. Nothing in that
package is edited. We call ``prq_quant`` / ``prq_dequant`` directly rather than the
``compress_latents`` / ``decompress_latents`` wrappers, because those hardcode
``PACK_OUTPUT_INT8=True`` / ``PACK_INPUT_INT8=True`` and both assert ``num_bits in (2, 4)``
(quant_pack.py:136, accumulate.py:136). Packing exists to stuff sub-byte codes into bytes;
at 8 bits a value already IS a byte, so packing off is the only meaningful setting and the
kernels accept it (both allow ``num_bits in (..., 8)``).

Consequence: this codec supports 2, 4 and 8-bit residuals. 16-bit is NOT reachable through
the reference at all -- both asserts cap at 8 -- so the int16-8s idea would need a separate
implementation. int8-8s is the highest-fidelity preset the tested code can express.

Measured on this Blackwell for one [16, 88, 160] frame (1280x704), int8-8s at the
default max_iters=100 / tol=1e-4:

    encode 78.1 ms, decode 0.1 ms   (first call pays ~14 s of Triton JIT, once)
    err max 7.8e-3, mean 4.0e-4, rel 0.184%
    431,616 B vs raw fp16 450,560 B -> 0.958x

Encode runs once per latent frame, so 2x per 266.7 ms chunk (2 latent frames = 8 video
frames @30fps) -> ~156 ms, about 59% of the budget, and it is serial with the DiT step.
The looser max_iters=50 / tol=1e-2 used in the offline sweeps costs 26.8 ms (~20% of
budget) for err mean 4.055e-4 vs 4.036e-4 -- i.e. the tighter setting is ~2.9x the encode
time for a 0.5% mean-error improvement. Both are exposed on the config; if the chunk
budget gets tight, this is the first knob to loosen, since decode is unaffected either
way (no k-means on the client).

WIRE FORMAT (little-endian; mirrored byte-for-byte in codec/sas.js)
------------------------------------------------------------------
``static_params`` carries the knobs in the session header. The per-frame payload is four
concatenated blocks whose sizes follow from those knobs plus the per-frame ``[C, H, W]``::

    centroids    n_stages * n_clusters * C   bf16     (raw 16 bits)
    cluster_ids  n_stages * N                uint8    (n_clusters <= 256)
    scales       N * (C // block_size)       bf16
    residual     N * C                       int8     (num_bits=8, unpacked)

with ``N = H * W``. Decode is a lookup, a sum and a multiply -- no k-means -- which is why
it is cheap enough to run in a browser::

    x = sum(centroids[s][ids[s]] for s in stages) + residual * scale

bf16 is sent as its raw bits and widened on the client with ``f32_bits = u16 << 16``, since
bf16 is exactly the top half of an fp32. No half-precision support needed in JS.
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from flashdreams.serving.token_stream.codec.base import (
    TokenCodec,
    TokenCodecConfig,
    TokenCodecEncodeResult,
)

# The reference SAS package is not pip-installed; it is vendored alongside the pipeline.
# Overridable so the path is not baked into the source tree.
_SAS_ROOT = Path(
    __import__("os").environ.get("SAS_ROOT", Path.home() / "integrated_pipeline")
)


def _load_prq():
    """Import prq_quant / prq_dequant from the vendored reference package."""
    if str(_SAS_ROOT) not in sys.path:
        sys.path.insert(0, str(_SAS_ROOT))
    from sol_media_compression.sas.prq import prq_dequant, prq_quant

    return prq_quant, prq_dequant


def _bf16_bytes(t: torch.Tensor) -> bytes:
    """Raw bf16 bits; torch has no numpy bf16, so reinterpret through int16."""
    return t.to(torch.bfloat16).view(torch.int16).cpu().numpy().tobytes()


@dataclass(kw_only=True)
class SASTokenCodecConfig(TokenCodecConfig):
    """Config for the SAS token codec. Defaults are the ``int8-8s`` preset."""

    _target: type[TokenCodec] = field(default_factory=lambda: SASTokenCodec)
    num_bits: int = 8
    """Residual bits. 2, 4 or 8 -- 8 is the reference implementation's ceiling."""
    n_stages: int = 8
    """PRQ cascade depth."""
    n_clusters: int = 256
    """Centroids per stage; <= 256 keeps cluster ids in one byte."""
    block_size: int = 16
    """Channels sharing one residual scale. 16 = one scale per token, matching
    compress_latents, which uses min(C_padded, 16)."""
    max_iters: int = 100
    """k-means iteration cap per stage."""
    tol: float = 1e-4
    """k-means early-stop tolerance: stop when the fraction of tokens changing cluster
    assignment drops below this. Not a centroid-movement threshold (kmeans_euclid.py:67)."""
    debug_stats: bool = True
    """Decode server-side and log per-frame reconstruction error against the true latent."""


class SASTokenCodec(TokenCodec[SASTokenCodecConfig]):
    """SAS PRQ codec for the video token stream."""

    def __init__(self, config: SASTokenCodecConfig) -> None:
        super().__init__(config)
        if config.num_bits not in (2, 4, 8):
            raise ValueError(
                f"num_bits must be 2, 4 or 8 (reference kernels cap at 8); "
                f"got {config.num_bits}"
            )
        if config.n_clusters > 256:
            raise ValueError("n_clusters must be <= 256 (cluster ids are uint8)")
        self._prq_quant, self._prq_dequant = _load_prq()
        self._frames = 0
        self._err_min = float("inf")
        self._err_max = 0.0
        self._err_sum = 0.0
        self._sas_bytes = 0
        self._raw_bytes = 0

    @property
    def codec_id(self) -> str:
        c = self.config
        return f"sas-int{c.num_bits}-{c.n_stages}s"

    @property
    def static_params(self) -> dict[str, Any]:
        c = self.config
        return {
            "num_bits": c.num_bits,
            "n_stages": c.n_stages,
            "n_clusters": c.n_clusters,
            "block_size": c.block_size,
            "centroid_dtype": "bf16",
            "scale_dtype": "bf16",
            "residual_dtype": "int8",
        }

    def encode_frame(self, latent: torch.Tensor) -> TokenCodecEncodeResult:
        c = self.config
        C, H, W = latent.shape
        n = H * W
        # (B, H, S, D) with B=H=1: one C-dim token per spatial position, channel-last.
        # Same layout compress_latents builds via permute(0, 2, 3, 1).
        x = latent.permute(1, 2, 0).reshape(1, 1, n, C).contiguous().to(torch.bfloat16)

        cents, ids, res_q, scales = self._prq_quant(
            x, n_stages=c.n_stages, n_clusters=c.n_clusters, block_size=c.block_size,
            num_bits=c.num_bits, scale_precision=torch.bfloat16,
            max_iters=c.max_iters, tol=c.tol,
            PACK_OUTPUT_INT8=False,   # 8-bit needs no packing; see module docstring
            CLUSTER_ID_INT8=True,
        )

        payload = b"".join((
            _bf16_bytes(torch.stack([t.reshape(-1) for t in cents])),
            torch.stack([t.reshape(-1) for t in ids]).to(torch.uint8).cpu().numpy().tobytes(),
            _bf16_bytes(scales.reshape(-1)),
            res_q.reshape(-1).to(torch.int8).cpu().numpy().tobytes(),
        ))

        if c.debug_stats:
            self._record(latent, x, cents, ids, res_q, scales, len(payload))

        # Shape travels per frame so the client sizes buffers from the frame itself
        # rather than trusting the session header to hold for every frame.
        return TokenCodecEncodeResult(
            payload=payload, frame_params=struct.pack("<HHH", C, H, W)
        )

    def _record(self, latent, x, cents, ids, res_q, scales, nbytes: int) -> None:
        """Server-side decode + error stats. This is exactly what the client rebuilds."""
        from loguru import logger

        c = self.config
        rec = self._prq_dequant(
            centroids_list=cents, cluster_ids_list=ids, residual_quant=res_q,
            scales=scales, block_size=c.block_size, num_bits=c.num_bits,
            PACK_INPUT_INT8=False, CLUSTER_ID_INT8=True, output_dtype=torch.bfloat16,
        )
        # Compare against the ORIGINAL latent, not the bf16 copy handed to prq_quant.
        # If the DiT ever emits fp32, measuring against x would hide the fp32->bf16 cast
        # error inside the encoder and report a flatteringly small number; the client
        # only ever sees `rec`, so the honest reference is what came in.
        C, H, W = latent.shape
        ref = latent.permute(1, 2, 0).reshape(1, 1, H * W, C).float()
        err = (rec.float() - ref).abs()
        mn, mx, mean = err.min().item(), err.max().item(), err.mean().item()
        rel = (err.pow(2).mean().sqrt() / ref.std().clamp_min(1e-12)).item()

        self._frames += 1
        self._err_min = min(self._err_min, mn)
        self._err_max = max(self._err_max, mx)
        self._err_sum += mean
        self._sas_bytes += nbytes
        self._raw_bytes += latent.numel() * 2      # raw fp16 reference codec

        logger.info(
            f"[SAS {self.codec_id}] frame {self._frames}  "
            f"err min={mn:.3e} max={mx:.3e} mean={mean:.3e} rel={rel:.4%}  |  "
            f"{nbytes:,} B vs fp16 {latent.numel() * 2:,} B = "
            f"{nbytes / (latent.numel() * 2):.3f}x  |  "
            f"session: err mean={self._err_sum / self._frames:.3e} "
            f"min={self._err_min:.3e} max={self._err_max:.3e} "
            f"ratio={self._sas_bytes / max(self._raw_bytes, 1):.3f}x"
        )
