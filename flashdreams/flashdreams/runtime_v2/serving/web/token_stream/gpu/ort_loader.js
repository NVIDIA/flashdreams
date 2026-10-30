// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Loads onnxruntime-web (WebGPU build) once and returns the global `ort`
// namespace. The runtime is served SAME-ORIGIN from the token-stream asset
// route: the server prefetches the pinned dist into its cache and advertises
// the loader + wasm URLs in the session header's `ort_runtime` descriptor, so
// the page carries no bundled binary and no cross-origin CDN dependency (which
// also sidesteps the COEP/CORP friction WebGPU + SharedArrayBuffer impose).

// Pinned reference version; the `ort_runtime` descriptor the server sends is the
// authoritative source of the actual URLs and version served.
export const ORT_VERSION = "1.20.1"

let _ortPromise = null

/**
 * Load onnxruntime-web and configure its asset paths from the session header's
 * `ort_runtime` descriptor. Idempotent: the script is injected at most once and
 * subsequent calls resolve to the same namespace.
 *
 * @param {object} ortDescriptor the session header `ort_runtime`:
 *   {version, wasm_base, loader}
 * @returns {Promise<object>} the global `ort` namespace
 */
export function loadOrt(ortDescriptor) {
  if (_ortPromise) {
    return _ortPromise
  }
  if (!ortDescriptor || !ortDescriptor.loader || !ortDescriptor.wasm_base) {
    return Promise.reject(
      new Error("session header carries no ort_runtime descriptor")
    )
  }
  _ortPromise = new Promise((resolve, reject) => {
    if (globalThis.ort) {
      resolve(_configure(globalThis.ort, ortDescriptor))
      return
    }
    const script = document.createElement("script")
    script.src = new URL(ortDescriptor.loader, location.href).href
    script.onload = () => {
      if (!globalThis.ort) {
        reject(new Error("onnxruntime-web loaded but the global 'ort' is missing"))
        return
      }
      resolve(_configure(globalThis.ort, ortDescriptor))
    }
    script.onerror = () =>
      reject(new Error(`failed to load onnxruntime-web from ${script.src}`))
    document.head.appendChild(script)
  })
  return _ortPromise
}

function _configure(ort, ortDescriptor) {
  // The WebGPU EP fetches its wasm (jsep) glue from wasmPaths; point it at the
  // same-origin dist directory the server serves.
  ort.env.wasm.wasmPaths = new URL(ortDescriptor.wasm_base, location.href).href
  ort.env.logLevel = "warning"
  return ort
}
