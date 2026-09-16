// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Loads onnxruntime-web (WebGPU build) once and returns the global `ort`
// namespace. The runtime is fetched from a CDN so the page carries no bundled
// binary; the URL and version live here alone so a later revision can point at
// a vendored copy served from the same origin without touching the decoder.

const ORT_VERSION = "1.20.1"
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`

let _ortPromise = null

/**
 * Load onnxruntime-web and configure its asset paths. Idempotent: the script is
 * injected at most once and subsequent calls resolve to the same namespace.
 *
 * @returns {Promise<object>} the global `ort` namespace
 */
export function loadOrt() {
  if (_ortPromise) {
    return _ortPromise
  }
  _ortPromise = new Promise((resolve, reject) => {
    if (globalThis.ort) {
      resolve(_configure(globalThis.ort))
      return
    }
    const script = document.createElement("script")
    script.src = `${ORT_BASE}ort.webgpu.min.js`
    script.onload = () => {
      if (!globalThis.ort) {
        reject(new Error("onnxruntime-web loaded but the global 'ort' is missing"))
        return
      }
      resolve(_configure(globalThis.ort))
    }
    script.onerror = () =>
      reject(new Error(`failed to load onnxruntime-web from ${script.src}`))
    document.head.appendChild(script)
  })
  return _ortPromise
}

async function _configure(ort) {
  // WebGPU EP still fetches its wasm (jsep) glue from the dist directory.
  ort.env.wasm.wasmPaths = ORT_BASE
  ort.env.logLevel = "warning"
  await _ensureWebGpuDevice(ort)
  return ort
}

// The cache-as-IO VAE decoder materializes full-resolution intermediates larger
// than WebGPU's default 1 GiB maxBufferSize / 128 MiB storage-binding limits, so
// CreateBuffer fails (and, since ORT-web shares one WebGPU device, that error
// also breaks the caption session). Create a single device up front requesting
// the adapter's maximum buffer limits (and shader-f16 when available) and hand
// it to ORT-web, so both sessions run on a device that can allocate them.
async function _ensureWebGpuDevice(ort) {
  if (ort.env.webgpu.device || !globalThis.navigator?.gpu) {
    return
  }
  try {
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) {
      return
    }
    ort.env.webgpu.device = await adapter.requestDevice({
      requiredFeatures: adapter.features.has("shader-f16") ? ["shader-f16"] : [],
      requiredLimits: {
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      },
    })
  } catch {
    // Leave ORT to create its own default device; large decodes may still fail.
  }
}

export { ORT_VERSION }
