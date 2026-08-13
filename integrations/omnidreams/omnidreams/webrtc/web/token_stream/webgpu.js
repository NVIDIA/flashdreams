// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// WebGPU capability probe used by the client to decide whether the token path
// is viable. Returns null when WebGPU is unavailable so callers can fall back
// to the pixel path.

/**
 * Detect WebGPU support and acquire an adapter + device.
 *
 * @returns {Promise<{adapter: GPUAdapter, device: GPUDevice, info: object} | null>}
 *   the adapter, device, and adapter info, or null when unavailable.
 */
export async function detectWebGPU() {
  if (!navigator.gpu) {
    return null
  }

  let adapter = null
  try {
    adapter = await navigator.gpu.requestAdapter()
  } catch {
    return null
  }
  if (!adapter) {
    return null
  }

  let device = null
  try {
    device = await adapter.requestDevice()
  } catch {
    return null
  }
  if (!device) {
    return null
  }

  // adapter.info is the current API; requestAdapterInfo() is the legacy path.
  let info = {}
  if (adapter.info) {
    info = adapter.info
  } else if (typeof adapter.requestAdapterInfo === "function") {
    try {
      info = await adapter.requestAdapterInfo()
    } catch {
      info = {}
    }
  }

  return { adapter, device, info }
}
