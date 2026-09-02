# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SAS (Scale-Adaptive Scalar / PRQ) latent quantizer for the OmniDreams pipeline.

Single-decode production knob. This module is fully self-contained: the PRQ
algorithm (multi-stage Euclidean k-means + block-wise residual quantization with
bit-packing) is implemented below in pure PyTorch, with no dependency on
``sol_media_compression`` or Triton. It runs on CUDA or CPU.

* :class:`SASQuantizer` / :class:`SASQuantizerConfig` � the quantizer. ``compress``
  serializes the clean latent to a compact payload; ``decompress`` reconstructs it.
* :class:`SASMetrics` / :func:`measure_sas` � perf / compression / quality for one
  chunk.
* :class:`SASHooks` / :func:`maybe_build_hooks` � the ``OMNI_SAS=1`` knob. It
  round-trips each chunk's clean latent (compress -> decompress) BEFORE the single
  VAE decode, so the decoder sees the reconstructed latent � the real client/server
  path (only the compressed payload would cross the network). It logs per-chunk
  compression and, at exit, writes ``prod_metrics.csv`` (+ ``prod_frame_metrics.csv``
  when ``OMNI_SAS_FRAME_STATS=1``). Off by default -> zero production overhead, and
  there is no second decode / no cache duplication.

Shape contract: works on any ``[..., C, H, W]`` latent. The Omnidreams clean
latent is ``[B, V, T, C, H, W]``; the leading dims are flattened into the PRQ
batch dim and restored on decompress.
"""

from __future__ import annotations

import atexit
import io
import math
import os
import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from flashdreams.infra.config import InstantiateConfig

try:
    from loguru import logger
except Exception:  # pragma: no cover - loguru always present in flashdreams
    import logging

    logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core PRQ � self-contained pure-PyTorch implementation
# ---------------------------------------------------------------------------
# PRQ (Progressive Residual Quantization) = N stages of Euclidean k-means, each
# quantizing the residual the previous stage left, followed by a block-wise
# symmetric integer quantization (with bit-packing) of the final residual. This is
# a direct pure-torch port of ``sol_media_compression.sas`` (prq / kmeans /
# quant_pack / accumulate). It produces the same serialized state layout � float32
# centroids + uint8 cluster ids + packed-uint8 residual + bf16 scales � so the
# compression ratios match the original kernels. Runs on CUDA or CPU.

# Team A pads the channel dim to >= 16 (their Triton ``tl.dot`` needs K >= 16); we
# keep the same padding so the token/byte layout is identical.
_MIN_DIM = 16


@torch.no_grad()
def _kmeans(
    x: torch.Tensor, k: int, max_iters: int, tol: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched Euclidean k-means (Lloyd's algorithm).

    Args:
        x: ``[BH, S, D]`` tokens.
        k: number of clusters (clamped to S).
        max_iters: iteration cap.
        tol: early-stop when the fraction of tokens changing assignment < tol.

    Returns ``(cluster_ids [BH, S] long, centroids [BH, k, D] float32)``, with the
    ids consistent with the returned centroids (a final assignment is done).
    """
    bh, s, d = x.shape
    k = min(k, s)
    # init: sample k random tokens per batch as centroids
    idx = torch.randint(0, s, (bh, k), device=x.device)
    centroids = torch.gather(x, 1, idx[..., None].expand(-1, -1, d)).clone().float()
    prev = None
    for _ in range(max_iters):
        ids = torch.cdist(x, centroids).argmin(dim=-1)         # assign [BH, S]
        onehot = F.one_hot(ids, k).to(x.dtype)                 # [BH, S, k]
        counts = onehot.sum(dim=1)                             # [BH, k]
        sums = torch.einsum("bsk,bsd->bkd", onehot, x)         # [BH, k, D]
        # empty clusters keep their previous centroid
        centroids = torch.where(
            counts[..., None] > 0, sums / counts[..., None].clamp(min=1), centroids
        )
        if prev is not None and (ids != prev).float().mean() < tol:
            break
        prev = ids
    ids = torch.cdist(x, centroids).argmin(dim=-1)             # final consistent assign
    return ids, centroids


@torch.no_grad()
def _quant_pack(
    residual: torch.Tensor, block_size: int, num_bits: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-wise symmetric integer quantization of ``[B, H, S, D]`` residual.

    Each contiguous ``block_size`` slice along D shares one scale
    (``max|x| / (2**(bits-1) - 1)``). Values are rounded/clamped to the signed
    range, shifted to unsigned, and packed into uint8 (2 values/byte at 4-bit,
    4/byte at 2-bit; 8-bit is one byte each). Returns ``(packed_uint8, scales_bf16)``.
    """
    b, h, s, d = residual.shape
    scale_d = d // block_size
    max_int = 2 ** (num_bits - 1) - 1
    x = residual.float().reshape(b, h, s, scale_d, block_size)
    scale = (x.abs().amax(dim=-1, keepdim=True) / max_int).clamp(min=1e-10)
    y = torch.round(x / scale).clamp(-max_int, max_int).to(torch.int32).reshape(b, h, s, d)
    scales = scale.squeeze(-1).to(torch.bfloat16)             # [B, H, S, scale_d]
    if num_bits == 4:
        u = (y + max_int).to(torch.uint8).reshape(b, h, s, d // 2, 2)
        packed = (u[..., 0] << 4) | u[..., 1]
    elif num_bits == 2:
        u = (y + max_int).to(torch.uint8).reshape(b, h, s, d // 4, 4)
        packed = (u[..., 0] << 6) | (u[..., 1] << 4) | (u[..., 2] << 2) | u[..., 3]
    else:  # 8-bit: one byte per value
        packed = (y + max_int).to(torch.uint8)
    return packed.contiguous(), scales.contiguous()


@torch.no_grad()
def _unpack_dequant(
    residual_quant: torch.Tensor, scales: torch.Tensor,
    b: int, h: int, s: int, d: int, block_size: int, num_bits: int,
) -> torch.Tensor:
    """Inverse of :func:`_quant_pack` -> dequantized residual ``[B, H, S, D]`` float."""
    max_int = 2 ** (num_bits - 1) - 1
    if num_bits == 4:
        p = residual_quant.to(torch.int32)
        hi = (p >> 4) & 0xF
        lo = p & 0xF
        y = torch.stack([hi, lo], dim=-1).reshape(b, h, s, d) - max_int
    elif num_bits == 2:
        p = residual_quant.to(torch.int32)
        v1 = (p >> 6) & 0x3
        v2 = (p >> 4) & 0x3
        v3 = (p >> 2) & 0x3
        v4 = p & 0x3
        y = torch.stack([v1, v2, v3, v4], dim=-1).reshape(b, h, s, d) - max_int
    else:
        y = residual_quant.to(torch.int32) - max_int
    scale_d = d // block_size
    resid = y.float().reshape(b, h, s, scale_d, block_size) * scales.float().reshape(
        b, h, s, scale_d, 1
    )
    return resid.reshape(b, h, s, d)


@torch.no_grad()
def _compress_latents(
    latents: torch.Tensor,
    num_bits: int = 4,
    num_stages: int = 2,
    num_clusters: int = 256,
    block_size: int | None = None,
    max_iters: int = 30,
    tol: float = 0.1,
) -> dict:
    """Compress ``[B, C, H, W]`` latents via PRQ -> serializable state dict.

    Reshapes to ``(B, 1, H*W, C_padded)`` tokens, runs ``num_stages`` k-means
    stages (subtracting each stage's centroids), then quantizes the final residual.
    """
    b, c, hp, wp = latents.shape
    c_padded = max(c, _MIN_DIM)
    if c_padded != c:
        latents = F.pad(latents, (0, 0, 0, 0, 0, c_padded - c))
    if block_size is None:
        block_size = min(c_padded, 16)

    x = latents.permute(0, 2, 3, 1).reshape(b, 1, hp * wp, c_padded).contiguous().float()
    s, d = hp * wp, c_padded

    centroids_list: list[torch.Tensor] = []
    cluster_ids_list: list[torch.Tensor] = []
    residual = x
    for _ in range(num_stages):
        ids, centroids = _kmeans(residual.reshape(b, s, d), num_clusters, max_iters, tol)
        k = centroids.shape[1]
        ids = ids.reshape(b, 1, s)
        centroids = centroids.reshape(b, 1, k, d)
        assert int(ids.max()) < 256, "cluster id overflow (num_clusters must be <= 256)"
        centroids_list.append(centroids.float())               # float32 (matches layout)
        cluster_ids_list.append(ids.to(torch.uint8))
        gathered = torch.gather(
            centroids.float(), 2, ids.long()[..., None].expand(-1, -1, -1, d)
        )
        residual = residual - gathered

    residual_quant, scales = _quant_pack(residual, block_size, num_bits)
    return {
        "centroids_list": centroids_list,
        "cluster_ids_list": cluster_ids_list,
        "residual_quant": residual_quant,
        "scales": scales,
        "_meta": {
            "B": b, "C": c, "C_padded": c_padded, "Hp": hp, "Wp": wp,
            "num_bits": num_bits, "block_size": block_size,
        },
    }


@torch.no_grad()
def _decompress_latents(state: dict) -> torch.Tensor:
    """Reconstruct ``[B, C, H, W]`` latents from a :func:`_compress_latents` state."""
    m = state["_meta"]
    b, c, hp, wp = m["B"], m["C"], m["Hp"], m["Wp"]
    c_padded = m.get("C_padded", c)
    s, d = hp * wp, c_padded
    accum = _unpack_dequant(
        state["residual_quant"], state["scales"], b, 1, s, d, m["block_size"], m["num_bits"]
    )
    for centroids, ids in zip(state["centroids_list"], state["cluster_ids_list"]):
        gathered = torch.gather(
            centroids.float(), 2, ids.long()[..., None].expand(-1, -1, -1, d)
        )
        accum = accum + gathered
    latents = accum.reshape(b, hp, wp, c_padded).permute(0, 3, 1, 2).contiguous()
    if c_padded != c:
        latents = latents[:, :c]
    return latents


# ---------------------------------------------------------------------------
# Quantizer
# ---------------------------------------------------------------------------
class SASQuantizer:
    """SAS/PRQ quantizer for diffusion latents.

    ``compress`` returns opaque bytes (a serialized PRQ state); ``decompress``
    reconstructs the original-shaped latent; ``roundtrip`` does both in place �
    the single-process path used by the pipeline for latent-space quantization
    and quality measurement. For a real client/server split, call ``compress``
    on the server and ``decompress`` on the client instead.

    Instantiated via :meth:`SASQuantizerConfig.setup`, i.e. ``SASQuantizer(config)``.
    """

    def __init__(self, config: "SASQuantizerConfig") -> None:
        self.config = config

    @torch.no_grad()
    def compress(self, latent: torch.Tensor) -> bytes:
        """``[..., C, H, W]`` latent -> serialized PRQ bytes."""
        assert latent.dim() >= 3, (
            f"expected a [..., C, H, W] latent, got shape {tuple(latent.shape)}"
        )
        # Two token layouts:
        #  per-frame (default): each frame -> its own PRQ codebook (frames folded into batch).
        #  share_frames       : ONE shared codebook across the T frames of a [B,V,T,C,H,W]
        #                       latent (fold T into the token axis) -> temporally consistent
        #                       (less flicker) and halves the fixed codebook cost.
        if getattr(self.config, "share_frames", False) and latent.dim() == 6:
            b, v, t, c, h, w = latent.shape
            x = latent.permute(0, 1, 3, 2, 4, 5).reshape(b * v, c, t * h, w).float()
            meta = {"mode": "share", "shape": tuple(latent.shape)}
        else:
            lead = tuple(latent.shape[:-3])
            c, h, w = latent.shape[-3:]
            x = latent.reshape(-1, c, h, w).float()
            meta = {"mode": "perframe", "lead": lead}
        if torch.cuda.is_available() and not x.is_cuda:
            x = x.cuda()
        state = _compress_latents(
            x,
            num_bits=self.config.num_bits,
            num_stages=self.config.num_stages,
            max_iters=self.config.max_iters,
            tol=self.config.tol,
        )
        meta["dtype"] = str(latent.dtype).replace("torch.", "")
        state["_flashdreams_meta"] = meta
        buf = io.BytesIO()
        torch.save(state, buf)
        return buf.getvalue()

    @torch.no_grad()
    def decompress(self, payload: bytes) -> torch.Tensor:
        """Serialized PRQ bytes -> ``[..., C, H, W]`` latent (original shape/dtype)."""
        buf = io.BytesIO(payload)
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
        state = torch.load(buf, map_location=map_location, weights_only=False)
        meta = state.get("_flashdreams_meta", {})
        recon = _decompress_latents(state)  # [-1, C, H, W]  (H is T*H in share mode)
        if meta.get("mode") == "share":
            b, v, t, c, h, w = meta["shape"]
            out = recon.reshape(b, v, c, t, h, w).permute(0, 1, 3, 2, 4, 5).contiguous()
        else:
            c, h, w = recon.shape[-3:]
            lead = tuple(meta.get("lead", ()))
            out = recon.reshape(*lead, c, h, w)
        dtype = getattr(torch, meta.get("dtype", "float32"), torch.float32)
        return out.to(dtype=dtype)

    @torch.no_grad()
    def roundtrip(self, latent: torch.Tensor) -> torch.Tensor:
        """compress -> decompress, returning the SAS-reconstructed latent."""
        return self.decompress(self.compress(latent))


@dataclass(kw_only=True)
class SASQuantizerConfig(InstantiateConfig):
    """Config for :class:`SASQuantizer`.

    Used by the ``OMNI_SAS`` single-decode knob (see :class:`SASHooks`), e.g.
    ``SASQuantizerConfig(num_bits=4, num_stages=2)``.
    """

    _target: type = field(default_factory=lambda: SASQuantizer)

    num_bits: int = 4
    """PRQ residual quantization bits per stage. Only 2 or 4 are supported by the
    residual bit-packer (which packs 2 or 4 values per byte)."""

    num_stages: int = 2
    """Number of progressive residual (k-means) stages. Each stage restores the
    residual the previous stage missed at dequantization, so more stages = higher
    fidelity, lower compression. This is the main residual/fidelity dial."""

    max_iters: int = 30
    """k-means iteration cap per stage (converges well before this)."""

    tol: float = 0.1
    """k-means early-stop threshold (fraction of tokens changing assignment)."""

    share_frames: bool = False
    """If True and the latent is [B,V,T,C,H,W], fit ONE codebook across the T frames
    (fold T into the token axis) instead of one per frame. More temporally consistent
    (less flicker) and halves the fixed codebook cost, at the price of a larger k-means."""


# ---------------------------------------------------------------------------
# Presets  (e.g. "int4-2s", "int4-4s", "int4-2s+share")
# ---------------------------------------------------------------------------
def parse_preset(name: str) -> SASQuantizerConfig:
    """``"int4-2s"`` / ``"int4-4s+share"`` -> :class:`SASQuantizerConfig`.

    Format: ``int<bits>-<stages>s`` with an optional ``+share`` suffix.
    ``bits`` must be 2 or 4 (``quant_pack`` limitation).
    """
    raw = name.strip().lower()
    share = raw.endswith("+share")
    if share:
        raw = raw[: -len("+share")]
    import re

    m = re.fullmatch(r"int(\d+)-(\d+)s", raw)
    if not m:
        raise ValueError(
            f"bad SAS preset {name!r}; expected 'int<bits>-<stages>s' "
            f"(e.g. 'int4-2s', 'int4-4s', 'int4-2s+share')"
        )
    bits, stages = int(m.group(1)), int(m.group(2))
    if bits not in (2, 4):
        raise ValueError(
            f"SAS preset {name!r}: num_bits={bits} unsupported; "
            f"the residual bit-packer only allows 2 or 4."
        )
    return SASQuantizerConfig(num_bits=bits, num_stages=stages, share_frames=share)


def preset_name(cfg: SASQuantizerConfig) -> str:
    """Canonical ``int<bits>-<stages>s[+share]`` label for a config."""
    base = f"int{cfg.num_bits}-{cfg.num_stages}s"
    return base + ("+share" if cfg.share_frames else "")


def build_preset_quantizer(name: str) -> "SASQuantizer":
    """Build a :class:`SASQuantizer` from a preset string, e.g. ``"int4-4s"``."""
    return SASQuantizer(parse_preset(name))


# ---------------------------------------------------------------------------
# Metrics: perf / compression / quality
# ---------------------------------------------------------------------------
@dataclass
class SASMetrics:
    """One chunk's SAS measurement."""

    # --- PERFORMANCE ---
    latency_ms: float      # SAS compress+decompress time (CUDA-synced wall clock)

    # --- COMPRESSION ---
    fp16_bytes: int        # original latent size (fp16 = 2 bytes / value)
    sas_bytes: int         # SAS payload size
    ratio: float           # fp16_bytes / sas_bytes  (e.g. 2.22x)
    saved_pct: float       # 100 * (1 - sas_bytes / fp16_bytes)

    # --- QUALITY (latent space) ---
    rel_error: float       # ||recon - latent|| / ||latent||
    latent_snr_db: float   # -20 * log10(rel_error)

    def summary(self) -> str:
        return (
            f"perf {self.latency_ms:.2f} ms | "
            f"compression {self.ratio:.2f}x ({self.saved_pct:.0f}% saved) | "
            f"quality {self.latent_snr_db:.1f} dB SNR ({100 * self.rel_error:.2f}% err)"
        )


@torch.no_grad()
def measure_sas(quantizer, latent: torch.Tensor) -> tuple[torch.Tensor, SASMetrics]:
    """Run SAS on one latent chunk and measure perf / compression / quality.

    Returns ``(reconstructed_latent, metrics)``.
    """
    # PERFORMANCE � time the compress + decompress round-trip (GPU-synced).
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    payload = quantizer.compress(latent)      # latent tensor -> compact bytes
    recon = quantizer.decompress(payload)     # bytes -> latent tensor
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - t0) * 1e3

    # COMPRESSION � original bytes vs payload bytes.
    fp16_bytes = latent.numel() * 2           # fp16 stores 2 bytes per value
    sas_bytes = len(payload)
    ratio = fp16_bytes / max(1, sas_bytes)

    # QUALITY (latent) � relative reconstruction error, expressed as an SNR.
    err = (recon.float() - latent.float()).norm()
    rel_error = (err / latent.float().norm().clamp(min=1e-8)).item()
    latent_snr_db = -20.0 * math.log10(rel_error) if rel_error > 0 else float("inf")

    return recon, SASMetrics(
        latency_ms=latency_ms,
        fp16_bytes=fp16_bytes,
        sas_bytes=sas_bytes,
        ratio=ratio,
        saved_pct=100.0 * (1.0 - sas_bytes / fp16_bytes),
        rel_error=rel_error,
        latent_snr_db=latent_snr_db,
    )


def pixel_psnr(reference: torch.Tensor, test: torch.Tensor, max_val: float = 255.0) -> float:
    """QUALITY (pixel space) � PSNR of ``test`` frames against ``reference`` frames.

    Use for generated vs ground-truth frames.
    """
    mse = ((reference.float() - test.float()) ** 2).mean()
    return (10.0 * torch.log10((max_val ** 2) / mse.clamp(min=1e-8))).item()


# ---------------------------------------------------------------------------
# Single-decode SAS knob (OMNI_SAS=1)
# ---------------------------------------------------------------------------
class SASHooks:
    """Production single-decode SAS knob.

    Enabled by ``OMNI_SAS=1``. Applies the SAS quantizer (compress -> decompress)
    to each chunk's clean latent BEFORE the one VAE decode, so the decoder sees the
    reconstructed latent � the real client/server path (only the compressed payload
    would cross the network). No second decode, no decoder-cache duplication.

    Logs per-chunk compression and, at exit, writes a compression report:
    ``prod_metrics.csv`` (per chunk) and, when ``OMNI_SAS_FRAME_STATS=1``,
    ``prod_frame_metrics.csv`` (per latent frame � SAS fits one codebook per latent
    frame, so each T-slice is measured on its own).

    Env:
      OMNI_SAS=1                enable
      OMNI_SAS_PRESET=int4-4s   preset (default int4-4s; also reads OMNI_SAS_PRESETS)
      OMNI_SAS_OUTDIR=~         directory for the CSVs (default ~)
      OMNI_SAS_FRAME_STATS=1    also dump per-latent-frame stats
    """

    def __init__(self) -> None:
        name = (
            os.environ.get("OMNI_SAS_PRESET")
            or os.environ.get("OMNI_SAS_PRESETS", "").split(",")[0].strip()
            or "int4-4s"
        )
        self.preset = name
        self.quantizer = build_preset_quantizer(name)
        self.outdir = os.path.expanduser(os.environ.get("OMNI_SAS_OUTDIR", "~"))
        self.frame_stats = os.environ.get("OMNI_SAS_FRAME_STATS") == "1"
        self._rows: list[str] = []
        self._frame_rows: list[str] = []
        atexit.register(self._write_metrics)
        logger.info(
            f"[SAS] single-decode knob ENABLED | preset={name} | "
            f"outdir={self.outdir} | frame_stats={self.frame_stats}"
        )

    @torch.no_grad()
    def before_decode(self, clean_latent: torch.Tensor, ar_index: int) -> torch.Tensor:
        """Quantize the chunk latent and return the reconstruction for the single
        decode. Records per-chunk (and optionally per-latent-frame) compression."""
        recon, m = measure_sas(self.quantizer, clean_latent)
        self._rows.append(
            f"{ar_index},{m.fp16_bytes},{m.sas_bytes},{m.ratio:.4f},"
            f"{m.saved_pct:.2f},{100 * m.rel_error:.4f},{m.latent_snr_db:.2f},"
            f"{m.latency_ms:.3f}"
        )
        if self.frame_stats:
            self._record_frame_stats(clean_latent, ar_index)
        logger.info(f"[SAS] AR {ar_index} | {self.preset} | {m.summary()}")
        return recon

    @torch.no_grad()
    def _record_frame_stats(self, clean_latent: torch.Tensor, ar_index: int) -> None:
        """Per-latent-frame breakdown. SAS fits one codebook per latent frame, so
        measure each temporal slice on its own. ``clean_latent`` is [B,V,T,C,H,W]."""
        if clean_latent.dim() != 6:
            return
        t_dim = clean_latent.shape[2]
        for t in range(t_dim):
            frame = clean_latent[:, :, t : t + 1]
            _, fm = measure_sas(self.quantizer, frame)
            self._frame_rows.append(
                f"{ar_index},{t},{fm.fp16_bytes},{fm.sas_bytes},"
                f"{fm.ratio:.4f},{100 * fm.rel_error:.4f},{fm.latency_ms:.3f}"
            )

    def _write_metrics(self) -> None:
        """Write prod_metrics.csv (+ prod_frame_metrics.csv) and log a summary."""
        if not self._rows:
            return
        path = os.path.join(self.outdir, "prod_metrics.csv")
        with open(path, "w") as fh:
            fh.write(
                "ar_index,fp16_bytes,sas_bytes,ratio,saved_pct,rel_err_pct,"
                "latent_snr_db,quantize_ms\n"
            )
            fh.write("\n".join(self._rows) + "\n")

        col = lambda i: [float(r.split(",")[i]) for r in self._rows]
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
        lat = sorted(col(7))
        med_ms = lat[len(lat) // 2] if lat else 0.0
        logger.info(
            f"[SAS-PROD] wrote {path} | {len(self._rows)} chunks | {self.preset} | "
            f"COMPRESSION {mean(col(1)) / 1024:.0f} KB -> {mean(col(2)) / 1024:.0f} KB "
            f"({mean(col(3)):.3f}x, saved {mean(col(4)):.1f}%) | "
            f"QUALITY {mean(col(6)):.1f} dB SNR ({mean(col(5)):.2f}% err) | "
            f"PERF {med_ms:.1f} ms/chunk (median)"
        )

        if self._frame_rows:
            fpath = os.path.join(self.outdir, "prod_frame_metrics.csv")
            with open(fpath, "w") as fh:
                fh.write(
                    "ar_index,latent_frame,fp16_bytes,sas_bytes,ratio,"
                    "rel_err_pct,quantize_ms\n"
                )
                fh.write("\n".join(self._frame_rows) + "\n")
            logger.info(
                f"[SAS-PROD] wrote {fpath} | {len(self._frame_rows)} latent-frame rows"
            )


def maybe_build_hooks() -> "SASHooks | None":
    """Return :class:`SASHooks` iff ``OMNI_SAS=1``; else None.

    When None, ``base.py`` leaves the production path completely untouched.
    """
    if os.environ.get("OMNI_SAS") == "1":
        return SASHooks()
    return None
