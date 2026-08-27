// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Live caption engine for the token path. It consumes the SAME assembled latent
// chunks the VAE decoder uses — WITHOUT converting to pixels — and emits a short
// scene caption on a throttled cadence, running in parallel to decode + display.
//
// The classifier is a seam. A real trained latent->caption model (a small
// single-forward classifier exported to ONNX, run via onnxruntime-web on
// WebGPU, keyed by the session header's `caption_model` descriptor and its label
// schema / caption bank) will drop into `_classify`. Until that model is trained
// and served, a lightweight, model-free latent-activity heuristic stands in so
// the overlay and the end-to-end wiring are testable now.

const noop = () => {}

// Caption bank indexed by normalized latent activity (0 = still, 1 = high motion).
const STUB_CAPTIONS = [
  "Scene mostly static",
  "Slowly cruising forward",
  "Driving forward through the scene",
  "Moving quickly — steering through the scene",
  "Sharp maneuver — turning or braking",
]

export class CaptionEngine {
  /**
   * @param {object} options
   * @param {(text: string) => void} [options.onCaption] receives each caption
   * @param {(message: string, meta?: object) => void} [options.log] logger
   * @param {number} [options.intervalMs] min ms between emitted captions
   * @param {number} [options.windowChunks] latent chunks kept for the model seam
   * @param {object|null} [options.descriptor] session-header `caption_model`
   * @param {GPUDevice|null} [options.device] WebGPU device (for the real model)
   */
  constructor({
    onCaption = noop,
    log = noop,
    intervalMs = 1200,
    windowChunks = 6,
    descriptor = null,
    device = null,
  } = {}) {
    this._onCaption = onCaption
    this._log = log
    this._intervalMs = intervalMs
    this._windowChunks = windowChunks
    this._descriptor = descriptor
    this._device = device

    this._window = [] // recent chunks' latent frames, for the real-model seam
    this._prevFrame = null
    this._activityEma = 0
    this._actMin = Infinity
    this._actMax = -Infinity
    this._lastEmit = 0
    this._model = null // future: onnxruntime-web WebGPU session

    if (descriptor) {
      // Seam: a caption_model is advertised — a real model could be loaded here
      // (fetch ONNX, build the WebGPU session). Left unimplemented in the
      // scaffold, so the stub classifier runs.
      this._log(
        `caption model advertised (${descriptor.version ?? "?"}); stub classifier active for now`,
        { source: "client" }
      )
    } else {
      this._log("no caption model served; using stub latent-activity classifier", {
        source: "client",
      })
    }
  }

  /**
   * Feed one chunk's assembled latent frames (typed arrays, each [Cl*Hl*Wl]).
   * Updates the activity signal and emits a caption on cadence. Cheap: one pass
   * over the latest frame; the emit is throttled.
   */
  pushChunk(latentFrames) {
    if (!latentFrames || latentFrames.length === 0) {
      return
    }
    this._window.push(latentFrames)
    if (this._window.length > this._windowChunks) {
      this._window.shift()
    }

    // Latent activity = mean |Δ| between this chunk's last frame and the prior.
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
    if (now - this._lastEmit >= this._intervalMs) {
      this._lastEmit = now
      const text = this._classify()
      if (text) {
        this._onCaption(text)
      }
    }
  }

  _classify() {
    // Real-model seam: if a WebGPU caption model were loaded, run it on the
    // latent window (this._window) here and map its logits to a caption via the
    // descriptor's label schema / caption bank.
    //
    // Stub: map the smoothed latent activity, normalized within the observed
    // range (so it is scale-free w.r.t. the latent magnitude), to a caption.
    const range = this._actMax - this._actMin
    const norm =
      range > 1e-9 ? (this._activityEma - this._actMin) / range : 0.5
    const idx = Math.min(
      STUB_CAPTIONS.length - 1,
      Math.max(0, Math.floor(norm * STUB_CAPTIONS.length))
    )
    return STUB_CAPTIONS[idx]
  }

  stop() {
    this._window = []
    this._prevFrame = null
    this._model = null
  }
}
