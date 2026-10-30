// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Display-side pixel transform for the token path. The exported decoder graph
// already bakes in latent de-normalization and the decode, emitting RGB in the
// [-1, 1] range with a planar [T, 3, H, W] layout. Presenting it requires only
// the final display remap: [-1, 1] -> [0, 255] with clamping, reordered to the
// interleaved RGBA an ImageData/canvas expects.
//
// This is the required display op the token path keeps; optional pixel
// post-processing (e.g. upscaling) is intentionally not applied here and
// remains exclusive to the pixel-streaming path.

/**
 * Convert a decoded RGB tensor into one ImageData per frame.
 *
 * @param {object} rgbResult
 * @param {ArrayLike<number>} rgbResult.data planar [T, C, H, W] RGB in [-1, 1]
 *   (a Float32Array, or a Float16Array whose elements read back as floats)
 * @param {number} rgbResult.frames number of frames T
 * @param {number} rgbResult.channels channels C (expected 3)
 * @param {number} rgbResult.height frame height H
 * @param {number} rgbResult.width frame width W
 * @returns {ImageData[]} one RGBA ImageData per frame, display-ready
 */
export function applyDisplayTransform({ data, frames, channels, height, width }) {
  const hw = height * width
  const frameStride = channels * hw
  const out = new Array(frames)

  for (let t = 0; t < frames; t += 1) {
    const base = t * frameStride
    const rBase = base
    const gBase = base + hw
    const bBase = base + 2 * hw
    // Uint8ClampedArray clamps out-of-range values to [0, 255] on assignment.
    const rgba = new Uint8ClampedArray(hw * 4)
    for (let p = 0; p < hw; p += 1) {
      const o = p * 4
      rgba[o] = (data[rBase + p] + 1) * 127.5
      rgba[o + 1] = (data[gBase + p] + 1) * 127.5
      rgba[o + 2] = (data[bBase + p] + 1) * 127.5
      rgba[o + 3] = 255
    }
    out[t] = new ImageData(rgba, width, height)
  }
  return out
}
