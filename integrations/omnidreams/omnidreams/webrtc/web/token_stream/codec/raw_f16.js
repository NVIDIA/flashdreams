// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Decoder for the "raw_f16" token codec. The server encodes each latent frame
// as a contiguous little-endian IEEE-754 half-precision (float16) byte buffer
// with no per-frame codec params. This decoder reconstructs a typed array of
// the latent values.
//
// Precision: values are stored as float16 on the wire, so the decoded output
// carries at most float16 precision even when widened to a Float32Array in the
// fallback path.

/**
 * Whether the runtime exposes a native Float16Array. When present it is used
 * directly; otherwise a manual half-to-float widening produces a Float32Array.
 */
const HAS_FLOAT16 = typeof globalThis.Float16Array === "function"

export class RawFloat16Decoder {
  constructor() {
    this.codecId = "raw_f16"
  }

  /**
   * Accept the session-header static params. The raw codec is stateless, so
   * this is a no-op kept for interface parity with other decoders.
   */
  configure(_staticParams) {}

  /**
   * Decode one latent frame payload.
   *
   * @param {ArrayBuffer} payloadArrayBuffer raw float16 bytes, little-endian
   * @param {ArrayBuffer} [_frameParamsArrayBuffer] unused for this codec
   * @returns {Float16Array | Float32Array} decoded latent values
   */
  decode(payloadArrayBuffer, _frameParamsArrayBuffer) {
    if (payloadArrayBuffer.byteLength % 2 !== 0) {
      throw new Error(
        `raw_f16 payload must be an even byte length, got ${payloadArrayBuffer.byteLength}`
      )
    }

    if (HAS_FLOAT16) {
      // Native path: zero-copy view over the payload bytes.
      return new globalThis.Float16Array(payloadArrayBuffer)
    }

    // Fallback path: widen each half-precision value to float32 manually.
    const source = new DataView(payloadArrayBuffer)
    const count = payloadArrayBuffer.byteLength / 2
    const out = new Float32Array(count)
    for (let i = 0; i < count; i += 1) {
      out[i] = halfToFloat(source.getUint16(i * 2, true))
    }
    return out
  }
}

/**
 * Convert a 16-bit IEEE-754 half-precision bit pattern to a JS number.
 */
function halfToFloat(bits) {
  const sign = (bits & 0x8000) >> 15
  const exponent = (bits & 0x7c00) >> 10
  const fraction = bits & 0x03ff
  const signMul = sign ? -1 : 1

  if (exponent === 0) {
    // Subnormal or zero.
    return signMul * Math.pow(2, -14) * (fraction / 1024)
  }
  if (exponent === 0x1f) {
    // Inf / NaN.
    return fraction ? NaN : signMul * Infinity
  }
  return signMul * Math.pow(2, exponent - 15) * (1 + fraction / 1024)
}
