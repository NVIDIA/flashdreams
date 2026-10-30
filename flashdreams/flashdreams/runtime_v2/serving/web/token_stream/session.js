// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Orchestrator for the token-streaming path. Given a token-stream URL and a
// WebGPU device, it opens the TokenStreamSocket, reads the session header,
// selects the codec from the registry, assembles per-chunk latent frames
// (buffered by chunk id until the last-in-chunk flag), decodes them with the
// WebGPU VAE decoder, and drives the render loop.
//
// If the VAE decoder cannot start (no WebGPU, an unsupported client, or a
// mismatched artifact), the failure is surfaced on-screen and the session keeps
// acking chunks so flow control does not stall — it never falls back to pixel.

import { getDecoder } from "./codec/registry.js"
import { TokenStreamSocket } from "./network/token_socket.js"
import { RenderLoop } from "./render/render_loop.js"
import { VaeDecoder } from "./gpu/vae_decoder.js"

const noop = () => {}

export class TokenStreamSession {
  /**
   * @param {object} options
   * @param {string} options.url token-stream endpoint URL
   * @param {GPUDevice} [options.device] WebGPU device from the capability probe
   * @param {HTMLCanvasElement} [options.canvas] target canvas for presentation
   * @param {RTCDataChannel} [options.controlChannel] control channel for
   *   capability signaling (reserved for negotiation; not required here)
   * @param {(message: string, meta?: object) => void} [options.log] logger
   */
  constructor({ url, device = null, canvas = null, controlChannel = null, log = noop }) {
    this._url = url
    this._device = device
    this._canvas = canvas
    this._controlChannel = controlChannel
    this._log = log

    this._socket = null
    this._decoder = null
    this._vaeDecoder = null
    this._renderLoop = null
    this._sessionHeader = null
    this._vaeReady = false
    this._started = false

    // chunkId -> ordered array of decoded latent frames awaiting completion.
    this._pendingChunks = new Map()
  }

  /** Open the socket and begin consuming token frames. Idempotent. */
  start() {
    if (this._started) {
      return
    }
    this._started = true

    this._socket = new TokenStreamSocket(this._url, {
      onOpen: () => this._log("token-stream socket open", { source: "client" }),
      onSessionHeader: (header) => this._onSessionHeader(header),
      onFrame: (frame) => this._onFrame(frame),
      onClose: () => this._log("token-stream socket closed", { source: "client" }),
      onError: (error) =>
        this._log(`token-stream error: ${error.message}`, {
          source: "client",
          level: "error",
        }),
    })
    this._socket.open()
  }

  /** Tear down the socket, render loop, and buffered state. */
  stop() {
    this._started = false
    if (this._renderLoop) {
      this._renderLoop.stop()
      this._renderLoop = null
    }
    if (this._socket) {
      this._socket.close()
      this._socket = null
    }
    this._pendingChunks.clear()
    this._vaeReady = false
  }

  _onSessionHeader(header) {
    this._sessionHeader = header
    const codecId = header?.codec?.id
    this._log(
      `token session header: codec=${codecId} shape=${JSON.stringify(header?.latent_shape)} T=${header?.frames_per_chunk} fps=${header?.fps}`,
      { source: "client" }
    )

    try {
      this._decoder = getDecoder(codecId)
      this._decoder.configure(header?.codec?.static_params ?? {})
    } catch (error) {
      this._log(`token codec unavailable: ${error.message}`, {
        source: "client",
        level: "error",
      })
      return
    }

    this._renderLoop = new RenderLoop({
      canvas: this._canvas,
      onChunkPresented: (chunkId) => this._socket?.sendAck(chunkId),
    })
    this._renderLoop.start()

    // Start the VAE decoder. On failure, _initVaeDecoder surfaces it on-screen
    // and the session keeps acking chunks without a present step (never falling
    // back to pixel), so flow control does not stall.
    void this._initVaeDecoder(header)
  }

  async _initVaeDecoder(header) {
    if (!this._device) {
      this._log("no WebGPU device; token frames will assemble without decode", {
        source: "client",
      })
      this._renderLoop?.showMessage("Token decode unavailable: no WebGPU device")
      return
    }
    this._vaeDecoder = new VaeDecoder()
    try {
      await this._vaeDecoder.init(header, this._device, this._log)
      this._vaeReady = true
      const cacheMode = this._vaeDecoder.cacheMode
      if (cacheMode === "cpu") {
        // GPU-resident cache unavailable on this client → decode is ~2x slower;
        // surface it on-screen so a slow stream isn't mistaken for "just slow".
        this._renderLoop?.setBadge("SLOW: CPU decode (~2x)")
      }
      this._log(`VAE decoder ready (precision=${this._vaeDecoder.precision})`, {
        source: "client",
      })
    } catch (error) {
      this._vaeReady = false
      this._log(`VAE decoder not available: ${error.message}`, {
        source: "client",
      })
      // Fail loud on-screen; token mode stays selected (never falls back to pixel).
      this._renderLoop?.showMessage(`Token decode unavailable: ${error.message}`)
    }
  }

  _onFrame(frame) {
    if (!this._decoder) {
      // No usable codec; ack so the server does not stall on flow control.
      if (frame.isLastInChunk) {
        this._socket?.sendAck(frame.chunkId)
      }
      return
    }

    let latent
    try {
      latent = this._decoder.decode(frame.payload, frame.codecParams)
    } catch (error) {
      this._log(`token frame decode failed: ${error.message}`, {
        source: "client",
        level: "error",
      })
      if (frame.isLastInChunk) {
        this._socket?.sendAck(frame.chunkId)
      }
      return
    }

    let frames = this._pendingChunks.get(frame.chunkId)
    if (!frames) {
      frames = []
      this._pendingChunks.set(frame.chunkId, frames)
    }
    frames[frame.frameIdx] = latent

    if (frame.isLastInChunk) {
      this._pendingChunks.delete(frame.chunkId)
      void this._completeChunk(frame.chunkId, frames)
    }
  }

  async _completeChunk(chunkId, latentFrames) {
    const codecId = this._sessionHeader?.codec?.id
    let decodedFloats = 0
    for (const latent of latentFrames) {
      decodedFloats += latent?.length ?? 0
    }
    this._log(
      `token chunk ${chunkId}: ${latentFrames.length} frames, ${codecId} decoded floats=${decodedFloats} (VAE decode pending)`,
      { source: "client" }
    )

    if (!this._vaeReady || !this._vaeDecoder || !this._renderLoop) {
      // Decoder unavailable (failed to start / no WebGPU): ack directly so flow
      // control keeps advancing; the failure is already shown on-screen.
      this._socket?.sendAck(chunkId)
      return
    }

    try {
      const rgb = await this._vaeDecoder.decode(latentFrames)
      this._renderLoop.enqueue({ chunkId, frames: rgb.frames ?? [rgb] })
    } catch (error) {
      this._log(`token chunk ${chunkId} decode failed: ${error.message}`, {
        source: "client",
        level: "error",
      })
      // Ack anyway so a decode failure does not deadlock the stream.
      this._socket?.sendAck(chunkId)
    }
  }
}
