// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Decoder-agnostic seam for turning a chunk of decoded latent frames into
// displayable RGB. The concrete WebGPU VAE implementation is intentionally not
// part of this scaffold; a follow-up fills these methods in.
//
// Intended contract:
//   - init(sessionHeader, device): prepare pipelines/weights for the latent
//     shape and codec described by the session header, bound to the supplied
//     WebGPU device.
//   - decode(latentFrames): take one chunk's worth of latent frames (an array
//     of typed arrays shaped [T][Cl*Hl*Wl], matching latent_shape from the
//     session header) and return an RGB result the render loop can present,
//     e.g. a GPUTexture or a GPUBuffer plus width/height metadata.
//
// Both methods are async so the implementation may await device work.

export class VaeDecoder {
  constructor() {
    this._sessionHeader = null
    this._device = null
  }

  /**
   * Prepare the decoder for a session.
   *
   * @param {object} sessionHeader parsed session-header JSON
   * @param {GPUDevice} device WebGPU device from the capability probe
   */
  async init(sessionHeader, device) {
    // Placeholder: retain inputs so a follow-up can build pipelines from the
    // latent shape / codec static params without changing the call site.
    this._sessionHeader = sessionHeader
    this._device = device
    throw new Error("VAE decode not implemented in this scaffold")
  }

  /**
   * Decode a chunk of latent frames into an RGB texture/buffer.
   *
   * @param {Array<Float16Array | Float32Array>} latentFrames chunk latents
   * @returns {Promise<object>} RGB result (texture/buffer + metadata)
   */
  async decode(latentFrames) {
    void latentFrames
    throw new Error("VAE decode not implemented in this scaffold")
  }
}
