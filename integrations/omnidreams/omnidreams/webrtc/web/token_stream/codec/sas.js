// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Decoder for the "sas-int<bits>-<stages>s" token codecs. Mirrors
// flashdreams/serving/token_stream/codec/sas.py byte-for-byte.
//
// Decoding cascaded PRQ needs no k-means -- that cost is entirely on the encoder. A frame
// rebuilds as a lookup, a sum and a multiply:
//
//     x[t][c] = SUM_over_stages centroids[s][ids[s][t]][c] + residual[t][c] * scale[t]
//
// which is what makes it affordable per frame in a browser on top of the ONNX decode.
//
// PAYLOAD LAYOUT (little-endian). Sizes come from the session header params plus the
// per-frame [C, H, W], with N = H * W:
//
//     centroids    n_stages * n_clusters * C   bf16 raw bits
//     cluster_ids  n_stages * N                uint8
//     scales       N * (C / block_size)        bf16 raw bits
//     residual     N * C                       int8
//
// bf16 IS the top half of an fp32, so widening is a shift. Unlike raw_f16 -- which needs
// a real half-float path or a manual mantissa/exponent unpack -- this needs neither.

/**
 * Widen packed bf16 bits into a Float32Array. Exact: bf16 occupies the high 16 bits of
 * the fp32 and the remaining mantissa bits are zero.
 */
function bf16ToFloat32(view, byteOffset, count) {
  const out = new Float32Array(count)
  const scratch = new ArrayBuffer(4)
  const asU32 = new Uint32Array(scratch)
  const asF32 = new Float32Array(scratch)
  for (let i = 0; i < count; i += 1) {
    asU32[0] = view.getUint16(byteOffset + i * 2, true) << 16
    out[i] = asF32[0]
  }
  return out
}

export class SASDecoder {
  /**
   * @param {string} codecId the wire id this instance serves
   */
  constructor(codecId) {
    this.codecId = codecId
    this.params = null
  }

  /**
   * Accept the session-header static params: n_stages, n_clusters, block_size,
   * num_bits and the dtype names. Unlike raw_f16 this codec is stateful -- without
   * these the payload cannot be segmented.
   */
  configure(staticParams) {
    const p = staticParams || {}
    for (const key of ["n_stages", "n_clusters", "block_size", "num_bits"]) {
      if (typeof p[key] !== "number") {
        throw new Error(`${this.codecId}: session header missing "${key}"`)
      }
    }
    if (p.residual_dtype !== "int8") {
      // The server emits int8 only: the reference kernels cap at 8 bits and packing is
      // off. Any other value means the two sides have drifted apart.
      throw new Error(`${this.codecId}: unsupported residual dtype "${p.residual_dtype}"`)
    }
    this.params = p
  }

  /**
   * Decode one latent frame.
   *
   * @param {ArrayBuffer} payloadArrayBuffer concatenated centroids/ids/scales/residual
   * @param {ArrayBuffer} frameParamsArrayBuffer packed <HHH: C, H, W
   * @returns {Float32Array} latent values, channel-first [C * H * W]
   */
  decode(payloadArrayBuffer, frameParamsArrayBuffer) {
    if (!this.params) {
      throw new Error(`${this.codecId}: decode() before configure()`)
    }
    if (!frameParamsArrayBuffer || frameParamsArrayBuffer.byteLength < 6) {
      throw new Error(`${this.codecId}: frame params must carry C, H, W`)
    }
    const fp = new DataView(frameParamsArrayBuffer)
    const C = fp.getUint16(0, true)
    const H = fp.getUint16(2, true)
    const W = fp.getUint16(4, true)
    const N = H * W

    const S = this.params.n_stages
    const K = this.params.n_clusters
    const blockSize = this.params.block_size
    const nScale = Math.max(1, Math.floor(C / blockSize))

    const view = new DataView(payloadArrayBuffer)
    let off = 0

    const centroids = bf16ToFloat32(view, off, S * K * C)
    off += S * K * C * 2

    const ids = new Uint8Array(payloadArrayBuffer, off, S * N)
    off += S * N

    const scales = bf16ToFloat32(view, off, N * nScale)
    off += N * nScale * 2

    const residual = new Int8Array(payloadArrayBuffer, off, N * C)
    off += N * C

    if (off !== payloadArrayBuffer.byteLength) {
      // A length mismatch means client and server disagree about the layout. Failing
      // loudly beats decoding plausible-looking garbage into the video.
      throw new Error(
        `${this.codecId}: consumed ${off} of ${payloadArrayBuffer.byteLength} bytes`
      )
    }

    // Sum the PRQ cascade and add the scaled residual. The encoder works in [N, C]
    // token-major order; the ONNX decoder wants channel-first, so transpose on write.
    const out = new Float32Array(C * N)
    for (let s = 0; s < S; s += 1) {
      const idBase = s * N
      const centStageBase = s * K * C
      for (let t = 0; t < N; t += 1) {
        const centBase = centStageBase + ids[idBase + t] * C
        for (let c = 0; c < C; c += 1) {
          out[c * N + t] += centroids[centBase + c]
        }
      }
    }
    for (let t = 0; t < N; t += 1) {
      const rBase = t * C
      const sBase = t * nScale
      for (let c = 0; c < C; c += 1) {
        out[c * N + t] += residual[rBase + c] * scales[sBase + ((c / blockSize) | 0)]
      }
    }
    return out
  }
}

export default SASDecoder
