// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// WebGPU VAE decoder for the token path. Runs the exported cache-as-IO TAEHV
// decoder (onnxruntime-web, WebGPU EP) to turn a chunk of latent frames into
// displayable RGB, threading the temporal cache from one chunk to the next.
//
// The graph exposes the decoder's per-block temporal cache as explicit
// `cache_in_*` inputs and `cache_out_*` outputs (see the `.spec.json` the
// session header carries). The initial cache is all zeros, which is exactly the
// decoder's native cold-start state, so chunk 0 needs no special handling; each
// run's `cache_out_*` become the next run's `cache_in_*`.
//
// Precision: fp32 is the portable default. fp16 (half the model size) needs the
// WebGPU `shader-f16` feature and a native `Float16Array`; it is used only when
// the session advertises an fp16 artifact and the client supports both,
// otherwise the decoder transparently falls back to fp32.

import { applyDisplayTransform } from "./display_transform.js"
import { loadOrt } from "./ort_loader.js"
import { runExclusive } from "./ort_lock.js"

export class VaeDecoder {
  constructor() {
    this._ort = null
    this._session = null
    this._cache = null // ort.Tensor[] threaded across chunks (cache_in/cache_out)
    this._precision = null
    // Serializes decode() calls: ORT-web run() is not reentrant, and the
    // temporal cache must thread strictly in submission order.
    this._chain = Promise.resolve()
  }

  /**
   * Prepare the decoder for a session from its `vae_model` descriptor.
   *
   * @param {object} sessionHeader parsed session-header JSON
   * @param {GPUDevice} device WebGPU device from the capability probe
   */
  async init(sessionHeader, device) {
    const descriptor = sessionHeader?.vae_model
    if (!descriptor) {
      throw new Error("session header carries no vae_model descriptor")
    }
    this._device = device
    this._ort = await loadOrt()

    this._precision = await this._selectPrecision(descriptor)
    this._dtype = this._precision === "fp16" ? "float16" : "float32"
    this._ArrayCtor =
      this._precision === "fp16" ? globalThis.Float16Array : Float32Array

    const modelUrl = new URL(descriptor.precisions[this._precision], location.href)
      .href
    this._session = await this._ort.InferenceSession.create(modelUrl, {
      executionProviders: ["webgpu"],
      graphOptimizationLevel: "all",
    })

    // I/O names: [0] is the latent / rgb, the rest are the cache slots in order.
    this._latentName = descriptor.input_names[0]
    this._rgbName = descriptor.output_names[0]
    this._cacheInNames = descriptor.input_names.slice(1)
    this._cacheOutNames = descriptor.output_names.slice(1)

    // ONNX latent input is [B=1, V=1, T, Cl, Hl, Wl]; descriptor.latent_shape is
    // the [T, Cl, Hl, Wl] tail.
    this._latentShape = [1, 1, ...descriptor.latent_shape]
    this._latentLen = this._latentShape.reduce((a, b) => a * b, 1)

    const [frames, channels, height, width] = descriptor.output_shape
    this._out = { frames, channels, height, width }

    this._cache = this._zeroCache(descriptor.cache)
    await this._warmup()
  }

  /**
   * Decode one chunk of latent frames into per-frame display-ready RGB.
   *
   * @param {Array<Float16Array | Float32Array>} latentFrames chunk latents,
   *   each frame flattened to [Cl * Hl * Wl]
   * @returns {Promise<{frames: ImageData[], width: number, height: number}>}
   */
  async decode(latentFrames) {
    // Queue behind any in-flight decode so runs never overlap and the cache
    // threads in order. A rejected decode must not wedge the queue, so the
    // chain swallows errors while the returned promise still surfaces them.
    const result = this._chain.then(() => this._decodeOne(latentFrames))
    this._chain = result.catch(() => {})
    return result
  }

  async _decodeOne(latentFrames) {
    const rgb = await this._run(this._assembleLatent(latentFrames))
    const frames = applyDisplayTransform({
      data: rgb.data,
      frames: this._out.frames,
      channels: this._out.channels,
      height: this._out.height,
      width: this._out.width,
    })
    return { frames, width: this._out.width, height: this._out.height }
  }

  async _selectPrecision(descriptor) {
    const preferred = descriptor.default_precision || "fp32"
    // Opt into fp16 only when the session offers it and the client can run the
    // half-precision shaders (native Float16Array + WebGPU shader-f16).
    if (descriptor.precisions.fp16 && typeof globalThis.Float16Array === "function") {
      try {
        const adapter = await navigator.gpu.requestAdapter()
        if (adapter && adapter.features.has("shader-f16")) {
          const device = await adapter.requestDevice({
            requiredFeatures: ["shader-f16"],
          })
          this._ort.env.webgpu.device = device
          return "fp16"
        }
      } catch {
        // fall through to fp32
      }
    }
    return descriptor.precisions[preferred] ? preferred : "fp32"
  }

  _zeroCache(cacheSpec) {
    return cacheSpec.map((slot) => {
      const count = slot.shape.reduce((a, b) => a * b, 1)
      return new this._ort.Tensor(this._dtype, new this._ArrayCtor(count), slot.shape)
    })
  }

  _assembleLatent(latentFrames) {
    const buffer = new this._ArrayCtor(this._latentLen)
    let offset = 0
    for (const frame of latentFrames) {
      buffer.set(frame, offset) // widens/narrows to the tensor dtype element-wise
      offset += frame.length
    }
    return new this._ort.Tensor(this._dtype, buffer, this._latentShape)
  }

  async _run(latentTensor) {
    const feeds = { [this._latentName]: latentTensor }
    this._cacheInNames.forEach((name, i) => {
      feeds[name] = this._cache[i]
    })
    const results = await runExclusive(() => this._session.run(feeds))
    // Thread the temporal cache forward for the next chunk.
    this._cache = this._cacheOutNames.map((name) => results[name])
    return results[this._rgbName]
  }

  async _warmup() {
    // One decode on a zero latent compiles the WebGPU shaders so the first real
    // chunk is not stalled. The cache is left at its zero cold-start state:
    // _run() reassigns this._cache from the warmup outputs, so restore it.
    const cold = this._cache
    await this._run(new this._ort.Tensor(
      this._dtype,
      new this._ArrayCtor(this._latentLen),
      this._latentShape,
    ))
    this._cache = cold
  }
}
