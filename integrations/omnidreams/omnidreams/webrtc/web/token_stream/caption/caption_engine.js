// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Live caption engine for the token path. It consumes the SAME assembled latent
// chunks the VAE decoder uses — WITHOUT converting to pixels — and emits a short
// scene caption on a throttled cadence, running in parallel to decode + display.
//
// Two classifiers:
//   - Real model: when the session header carries a `caption_model` descriptor,
//     the exported latent->caption-bank ONNX is fetched and run via
//     onnxruntime-web on WebGPU over a rolling latent window; argmax indexes the
//     caption bank. (Works even with an untrained/random model — it just yields
//     meaningless captions, useful for validating the end-to-end path.)
//   - Stub: a model-free latent-activity heuristic used when no model is served
//     or if the model fails to load, so the overlay always works.
//
// Callbacks: onCaption(text), onState("waiting"|"generating"), onModel(label)
// so the UI can show the caption, a Waiting/Generating dot, and the model in use.

import { loadOrt } from "../gpu/ort_loader.js"
import { runExclusive } from "../gpu/ort_lock.js"

const noop = () => {}

// Caption bank for the stub path, indexed by normalized latent activity.
const STUB_CAPTIONS = [
  "Scene mostly static",
  "Slowly cruising forward",
  "Driving forward through the scene",
  "Moving quickly — steering through the scene",
  "Sharp maneuver — turning or braking",
]

export class CaptionEngine {
  constructor({
    onCaption = noop,
    onState = noop,
    onModel = noop,
    log = noop,
    intervalMs = 1200,
    windowChunks = 6,
    descriptor = null,
    device = null,
  } = {}) {
    this._onCaption = onCaption
    this._onState = onState
    this._onModel = onModel
    this._log = log
    this._intervalMs = intervalMs
    this._descriptor = descriptor
    this._device = device

    // The model's temporal window (in chunks); prefer the descriptor's value.
    this._windowChunks = descriptor?.input_window_chunks ?? windowChunks
    this._captionBank = descriptor?.caption_bank ?? null
    this._modelVersion = descriptor?.version ?? null

    this._window = [] // recent chunks' latent frames
    this._prevFrame = null
    this._activityEma = 0
    this._actMin = Infinity
    this._actMax = -Infinity
    this._lastEmit = 0
    this._running = false

    this._ort = null
    this._session = null
    this._inputName = null
    this._outputName = null
    this._modelReady = false

    this._onState("waiting")
    if (descriptor) {
      this._onModel(`${this._modelVersion} · loading`)
      this._log(
        `caption model advertised (${descriptor.version ?? "?"}); loading…`,
        { source: "client" }
      )
      void this._initModel()
    } else {
      this._onModel("stub")
      this._log("no caption model served; using stub latent-activity classifier", {
        source: "client",
      })
    }
  }

  async _initModel() {
    try {
      this._ort = await loadOrt()
      const precision = this._descriptor.default_precision || "fp32"
      const url = new URL(this._descriptor.precisions[precision], location.href).href
      this._session = await this._ort.InferenceSession.create(url, {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: "all",
      })
      this._inputName = this._session.inputNames[0]
      this._outputName = this._session.outputNames[0]
      this._modelReady = true
      this._onModel(this._modelVersion)
      this._log(
        `caption model ready (${precision}, ${this._captionBank?.length ?? "?"} captions)`,
        { source: "client" }
      )
    } catch (error) {
      this._modelReady = false
      this._onModel("stub")
      this._log(`caption model unavailable (${error.message}); using stub`, {
        source: "client",
      })
    }
  }

  /**
   * Feed one chunk's assembled latent frames (typed arrays, each [Cl*Hl*Wl]).
   * Updates the rolling window + activity signal and emits a caption on cadence.
   */
  pushChunk(latentFrames) {
    if (!latentFrames || latentFrames.length === 0) {
      return
    }
    this._window.push(latentFrames)
    if (this._window.length > this._windowChunks) {
      this._window.shift()
    }

    // Latent activity (for the stub) = mean |Δ| vs the previous chunk's frame.
    const last = latentFrames[latentFrames.length - 1]
    if (this._prevFrame && this._prevFrame.length === last.length) {
      let acc = 0
      const n = last.length
      for (let i = 0; i < n; i += 1) {
        acc += Math.abs(last[i] - this._prevFrame[i])
      }
      const activity = acc / n
      this._activityEma =
        this._activityEma === 0 ? activity : 0.6 * this._activityEma + 0.4 * activity
      this._actMin = Math.min(this._actMin, this._activityEma)
      this._actMax = Math.max(this._actMax, this._activityEma)
    }
    this._prevFrame = last

    const now = performance.now()
    if (now - this._lastEmit >= this._intervalMs && !this._running) {
      this._lastEmit = now
      void this._emit()
    }
  }

  async _emit() {
    this._running = true
    this._onState("generating")
    try {
      const useModel = this._modelReady && this._window.length >= this._windowChunks
      const text = useModel ? await this._classifyModel() : this._classifyStub()
      this._onModel(useModel ? this._modelVersion : "stub")
      if (text) {
        this._onCaption(text)
      }
    } catch (error) {
      this._log(`caption inference failed (${error.message}); using stub`, {
        source: "client",
      })
      this._onModel("stub")
      this._onCaption(this._classifyStub())
    } finally {
      this._onState("waiting")
      this._running = false
    }
  }

  async _classifyModel() {
    // Flatten the last _windowChunks chunks' frames into [1, Twin, Cl, Hl, Wl].
    const chunks = this._window.slice(-this._windowChunks)
    const frames = []
    for (const chunk of chunks) {
      for (const frame of chunk) {
        frames.push(frame)
      }
    }
    const frameLen = frames[0].length
    const buffer = new Float32Array(frames.length * frameLen)
    let offset = 0
    for (const frame of frames) {
      buffer.set(frame, offset) // widens fp16 -> fp32 element-wise
      offset += frameLen
    }
    const [cl, hl, wl] = this._descriptor.latent_shape
    const shape = [1, frames.length, cl, hl, wl]
    const feeds = { [this._inputName]: new this._ort.Tensor("float32", buffer, shape) }
    const results = await runExclusive(() => this._session.run(feeds))
    const logits = results[this._outputName].data
    let best = 0
    for (let i = 1; i < logits.length; i += 1) {
      if (logits[i] > logits[best]) {
        best = i
      }
    }
    return this._captionBank?.[best] ?? `caption ${best}`
  }

  _classifyStub() {
    const range = this._actMax - this._actMin
    const norm = range > 1e-9 ? (this._activityEma - this._actMin) / range : 0.5
    const idx = Math.min(
      STUB_CAPTIONS.length - 1,
      Math.max(0, Math.floor(norm * STUB_CAPTIONS.length))
    )
    return STUB_CAPTIONS[idx]
  }

  stop() {
    this._window = []
    this._prevFrame = null
    this._session = null
    this._modelReady = false
  }
}
