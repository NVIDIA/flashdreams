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

function _configure(ort) {
  // WebGPU EP still fetches its wasm (jsep) glue from the dist directory.
  ort.env.wasm.wasmPaths = ORT_BASE
  ort.env.logLevel = "warning"
  return ort
}

export { ORT_VERSION }
