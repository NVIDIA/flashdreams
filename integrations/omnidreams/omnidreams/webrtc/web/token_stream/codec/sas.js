// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Client decoder for the SAS token codec. Reverses codec/sas.py byte-for-byte:
// parse the per-frame payload (stage centroids f32, cluster ids u8, packed
// integer residual u8, per-block scales f16), then reconstruct the latent by
// dequantizing the residual and adding each stage's centroid (gathered by
// cluster id) -- the inverse of infra/sas.py._decompress_latents. The result is
// a [C, H, W] float32 latent in C-major order, exactly what the raw codec emits,
// so the WebGPU VAE decoder consumes it unchanged.

const HAS_F16 = typeof globalThis.Float16Array === "function"

/** IEEE-754 half (u16) -> float32, for browsers without native Float16Array. */
function halfToFloat(h) {
  const sign = (h & 0x8000) >> 15
  const exp = (h & 0x7c00) >> 10
  const frac = h & 0x03ff
  if (exp === 0) return (sign ? -1 : 1) * Math.pow(2, -14) * (frac / 1024)
  if (exp === 0x1f) return frac ? NaN : (sign ? -Infinity : Infinity)
  return (sign ? -1 : 1) * Math.pow(2, exp - 15) * (1 + frac / 1024)
}

export class SASDecoder {
  /**
   * Configure from the session header. ``staticParams`` carries the quantizer
   * settings; ``latentShape`` is the per-frame [C, H, W] tail. Everything else
   * (padded channels, block size, token count, section sizes) is derived here so
   * decode() is a tight loop.
   *
   * @param {{num_bits:number,num_stages:number,num_clusters:number}} staticParams
   * @param {number[]} latentShape [C, H, W]
   */
  configure(staticParams, latentShape) {
    this._nb = staticParams?.num_bits ?? 4
    this._ns = staticParams?.num_stages ?? 2
    this._k = staticParams?.num_clusters ?? 256

    const [c, h, w] = latentShape
    this._C = c
    this._S = h * w // tokens per frame (one per H*W position)
    this._cp = Math.max(c, 16) // C_padded (matches infra/sas.py _MIN_DIM=16)
    this._bs = Math.min(this._cp, 16) // block_size
    this._scaleD = Math.floor(this._cp / this._bs)
    this._maxInt = Math.pow(2, this._nb - 1) - 1

    // Wire section byte sizes, in the order codec/sas.py writes them.
    this._centFloats = this._k * this._cp // per stage
    this._centBytes = this._ns * this._centFloats * 4
    this._idsBytes = this._ns * this._S
    this._residPerTok = (this._cp * this._nb) / 8 // packed bytes per token
    this._residBytes = this._S * this._residPerTok
    this._scaleCount = this._S * this._scaleD
    this._scaleBytes = this._scaleCount * 2
  }

  /**
   * Decode one frame payload into a [C, H, W] float32 latent (C-major).
   *
   * @param {ArrayBuffer} payload the SAS frame bytes
   * @returns {Float32Array} length C * H * W
   */
  decode(payload) {
    const buf = payload instanceof ArrayBuffer ? payload : payload.buffer
    const ns = this._ns
    const S = this._S
    const cp = this._cp
    const C = this._C
    const nb = this._nb
    const bs = this._bs
    const scaleD = this._scaleD
    const maxInt = this._maxInt

    let off = 0
    // Stage centroids: ns x [k*cp] float32 (slice -> aligned copy).
    const centroids = new Array(ns)
    for (let st = 0; st < ns; st += 1) {
      centroids[st] = new Float32Array(buf.slice(off, off + this._centFloats * 4))
      off += this._centFloats * 4
    }
    // Stage cluster ids: ns x [S] uint8.
    const ids = new Array(ns)
    for (let st = 0; st < ns; st += 1) {
      ids[st] = new Uint8Array(buf, off, S)
      off += S
    }
    // Packed integer residual: [S * residPerTok] uint8.
    const resid = new Uint8Array(buf, off, this._residBytes)
    off += this._residBytes
    // Per-block scales: [scaleCount] float16.
    let scales
    if (HAS_F16) {
      scales = new globalThis.Float16Array(buf.slice(off, off + this._scaleBytes))
    } else {
      const dv = new DataView(buf, off, this._scaleBytes)
      scales = new Float32Array(this._scaleCount)
      for (let i = 0; i < this._scaleCount; i += 1) {
        scales[i] = halfToFloat(dv.getUint16(i * 2, true))
      }
    }

    const out = new Float32Array(C * S)
    const perTok = this._residPerTok
    for (let t = 0; t < S; t += 1) {
      const residBase = t * perTok
      const scaleBase = t * scaleD
      for (let dd = 0; dd < cp; dd += 1) {
        // Unpack the signed quantized residual for this dim.
        let q
        if (nb === 4) {
          const byte = resid[residBase + (dd >> 1)]
          q = (dd & 1) === 0 ? (byte >> 4) & 0xf : byte & 0xf
        } else if (nb === 2) {
          const byte = resid[residBase + (dd >> 2)]
          const shift = 6 - (dd & 3) * 2
          q = (byte >> shift) & 0x3
        } else {
          q = resid[residBase + dd]
        }
        let val = (q - maxInt) * scales[scaleBase + ((dd / bs) | 0)]
        // Add each stage's centroid (gathered by this token's cluster id).
        for (let st = 0; st < ns; st += 1) {
          val += centroids[st][ids[st][t] * cp + dd]
        }
        // Drop padded channels (cp may exceed C); write C-major [C, H, W].
        if (dd < C) out[dd * S + t] = val
      }
    }
    return out
  }
}

export default SASDecoder
