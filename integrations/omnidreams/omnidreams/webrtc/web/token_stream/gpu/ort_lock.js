// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Global serialization of onnxruntime-web runs. ORT-web's WebGPU (JSEP) backend
// executes one InferenceSession.run() at a time across ALL sessions sharing the
// backend; overlapping runs (e.g. the VAE decoder and the caption model running
// in parallel) corrupt its shared state and throw "Session already started" /
// "Session mismatch". Route every run through this queue so runs never overlap.
// Callers that also need their own ordering (e.g. the VAE cache) keep it too.

let _tail = Promise.resolve()

/**
 * Run `fn` (returning a Promise) exclusively — after any previously queued run
 * settles. A rejected run does not wedge the queue; the returned promise still
 * surfaces the error to its caller.
 * @template T
 * @param {() => Promise<T>} fn
 * @returns {Promise<T>}
 */
export function runExclusive(fn) {
  const result = _tail.then(fn)
  _tail = result.catch(() => {})
  return result
}
