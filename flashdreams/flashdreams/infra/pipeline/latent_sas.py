# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SAS (cascaded PRQ) latent codec for the pixel path.

Round-trips a DiT latent through quantization and back so the comparison decoder sees
what a remote client would have reconstructed. Same algorithm and same settings as the
token-streaming codec in ``serving/token_stream/codec/sas.py``, so the offline pixel
comparison characterises the codec that actually runs on the wire.

USES THE REFERENCE IMPLEMENTATION UNMODIFIED
--------------------------------------------
Quantization is delegated to ``sol_media_compression.sas.prq``, the Triton implementation
already validated across the Cosmos CI8x8 and Wan studies and, more recently, over 366
live frames of token streaming. Nothing in that package is edited.

``prq_quant`` / ``prq_dequant`` are called directly rather than the ``compress_latents``
/ ``decompress_latents`` wrappers: those hardcode ``PACK_OUTPUT_INT8=True`` and assert
``num_bits in (2, 4)`` (quant_pack.py:136, accumulate.py:136). Packing exists to stuff
sub-byte codes into bytes; at 8 bits a value already IS a byte, so packing off is the
only meaningful setting and the kernels accept it.

Consequently 2, 4 and 8-bit residuals are reachable; 16-bit is not, at all.

SHAPE
-----
The pixel path carries ``[B, V, T, Cl, Hl, Wl]`` -- for this pipeline
``[1, 1, 2, 16, 88, 160]``. PRQ works on ``(B, H, S, D)`` token-major data, so each
latent frame is transposed to ``[1, 1, Hl*Wl, Cl]`` and quantized independently, exactly
as the token codec does per frame. Quantizing per frame rather than per chunk keeps the
two paths comparable, and matches how a streaming client receives data.

BUFFER CONTRACT
---------------
``roundtrip`` allocates via ``empty_like`` and never writes through its input. The caller
(``pipeline/base.py``) asserts storage disjointness on every step regardless.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from loguru import logger
from torch import Tensor

# The reference SAS package is vendored rather than pip-installed.
_SAS_ROOT = Path(os.environ.get("SAS_ROOT", Path.home() / "integrated_pipeline"))


def _load_prq():
    if str(_SAS_ROOT) not in sys.path:
        sys.path.insert(0, str(_SAS_ROOT))
    from sol_media_compression.sas.prq import prq_dequant, prq_quant

    return prq_quant, prq_dequant


class SASLatentCodec:
    """int8-8s cascaded PRQ: 8 stages, 256 centroids/stage, 8-bit residual."""

    name = "sas-int8-8s"

    def __init__(
        self,
        num_bits: int = 8,
        n_stages: int = 8,
        n_clusters: int = 256,
        block_size: int = 16,
        max_iters: int = 100,
        tol: float = 1e-4,
    ) -> None:
        if num_bits not in (2, 4, 8):
            raise ValueError(
                f"num_bits must be 2, 4 or 8 (reference kernels cap at 8); got {num_bits}"
            )
        if n_clusters > 256:
            raise ValueError("n_clusters must be <= 256 (cluster ids are uint8)")
        self.num_bits = num_bits
        self.n_stages = n_stages
        self.n_clusters = n_clusters
        self.block_size = block_size
        self.max_iters = max_iters
        self.tol = tol
        self.name = f"sas-int{num_bits}-{n_stages}s"
        self._prq_quant, self._prq_dequant = _load_prq()

        self._frames = 0
        self._err_min = float("inf")
        self._err_max = 0.0
        self._err_sum = 0.0
        self._sas_bytes = 0
        self._raw_bytes = 0

    def _frame_bytes(self, n_tokens: int, channels: int) -> int:
        """Wire size for one quantized frame, matching the token codec's layout.

        The residual is counted at its true bit width, not at the int8 container the
        kernels hand back with ``PACK_OUTPUT_INT8=False``. At 4 bits two values share a
        byte and at 2 bits four do; counting the container instead would overstate the
        payload by 2x or 4x and make the low-bit presets look far worse than they are.
        """
        n_scale = max(1, channels // self.block_size)
        return (
            self.n_stages * self.n_clusters * channels * 2   # centroids, bf16
            + self.n_stages * n_tokens                       # cluster ids, uint8
            + n_tokens * n_scale * 2                         # scales, bf16
            + (n_tokens * channels * self.num_bits + 7) // 8  # residual, packed
        )

    def _roundtrip_frame(self, frame: Tensor) -> Tensor:
        """Quantize + dequantize one ``[Cl, Hl, Wl]`` latent frame."""
        c, h, w = frame.shape
        n = h * w
        # (B, H, S, D) with B=H=1: one C-dim token per spatial position, channel-last.
        # permute makes this non-contiguous, so contiguous() copies -- the returned
        # tensor never shares storage with `frame`.
        x = frame.permute(1, 2, 0).reshape(1, 1, n, c).contiguous().to(torch.bfloat16)

        cents, ids, res_q, scales = self._prq_quant(
            x,
            n_stages=self.n_stages,
            n_clusters=self.n_clusters,
            block_size=self.block_size,
            num_bits=self.num_bits,
            scale_precision=torch.bfloat16,
            max_iters=self.max_iters,
            tol=self.tol,
            PACK_OUTPUT_INT8=False,  # 8-bit needs no packing; see module docstring
            CLUSTER_ID_INT8=True,
        )
        rec = self._prq_dequant(
            centroids_list=cents,
            cluster_ids_list=ids,
            residual_quant=res_q,
            scales=scales,
            block_size=self.block_size,
            num_bits=self.num_bits,
            PACK_INPUT_INT8=False,
            CLUSTER_ID_INT8=True,
            output_dtype=torch.bfloat16,
        )

        err = (rec.float() - x.float()).abs()
        mn, mx, mean = err.min().item(), err.max().item(), err.mean().item()
        rel = (err.pow(2).mean().sqrt() / x.float().std().clamp_min(1e-12)).item()
        nbytes = self._frame_bytes(n, c)
        raw = c * h * w * 2

        self._frames += 1
        self._err_min = min(self._err_min, mn)
        self._err_max = max(self._err_max, mx)
        self._err_sum += mean
        self._sas_bytes += nbytes
        self._raw_bytes += raw
        logger.info(
            "[SAS-PIXEL {}] frame {}  err min={:.3e} max={:.3e} mean={:.3e} rel={:.4%}"
            "  |  {:,} B vs fp16 {:,} B = {:.3f}x  |  session: err mean={:.3e} "
            "min={:.3e} max={:.3e} ratio={:.3f}x",
            self.name, self._frames, mn, mx, mean, rel, nbytes, raw, nbytes / raw,
            self._err_sum / self._frames, self._err_min, self._err_max,
            self._sas_bytes / max(self._raw_bytes, 1),
        )

        # Undo the transpose: [1,1,n,C] -> [n,C] -> [C,n] -> [C,H,W].
        return rec.reshape(n, c).permute(1, 0).reshape(c, h, w)

    def roundtrip(self, latent: Tensor) -> Tensor:
        """Round-trip ``[B, V, T, Cl, Hl, Wl]``, returning a NEW tensor of the same shape."""
        if latent.ndim != 6:
            raise ValueError(
                f"expected [B, V, T, Cl, Hl, Wl]; got {tuple(latent.shape)}"
            )
        out = torch.empty_like(latent)  # fresh storage; the seam asserts disjointness
        b_n, v_n, t_n = latent.shape[:3]
        for b in range(b_n):
            for v in range(v_n):
                for t in range(t_n):
                    out[b, v, t] = self._roundtrip_frame(latent[b, v, t]).to(
                        latent.dtype
                    )
        return out
