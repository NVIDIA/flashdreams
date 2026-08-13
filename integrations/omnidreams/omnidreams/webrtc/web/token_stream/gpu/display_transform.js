// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Seam for the display-side pixel transform. VAE output is typically in a
// normalized range (for example [-1, 1]); presenting it requires de-normalizing
// to [0, 1] (or [0, 255]) and clamping out-of-range values before the blit.
//
// The concrete transform (likely a small WebGPU compute/fragment step) is
// filled in by the follow-up alongside the VAE decoder. It is kept separate so
// the normalization convention lives in one place.

/**
 * Apply the de-normalize + clamp step to a decoded RGB result.
 *
 * @param {object} rgbResult decoder output (texture/buffer + metadata)
 * @returns {object} display-ready RGB result
 */
export function applyDisplayTransform(rgbResult) {
  void rgbResult
  throw new Error("display transform not implemented in this scaffold")
}
