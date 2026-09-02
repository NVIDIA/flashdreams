// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Orchestrator for the token-streaming path. Given a token-stream URL and a
// WebGPU device, it opens the TokenStreamSocket, reads the session header,
// selects the decoder from the registry, assembles per-chunk frames (buffered
// by chunk id until the last-in-chunk flag), decodes them, and drives the
// render loop.
//
// The VAE decoder is a seam in this scaffold: session assembly, per-frame codec
// decoding, and ack wiring are live, and the VAE/render stage gracefully no-ops
// (logs and acks the chunk) until the WebGPU decoder is implemented.
//
// ---------------------------------------------------------------------------
// Changes on top of the upstream video_token_streaming branch (client side):
//   * SAS token codec decode: codec/sas.js + registry, wired via ?codec= and
//     the "Video Tokens (SAS int4s2/4s4)" stream modes (smaller latents on the
//     wire; byte-exact vs the server-side quantizer).
//   * fp16/fp32 VAE fallback: vae_decoder.js picks fp16 only when the client
//     supports shader-f16, else fp32 (no hard fail on GPUs lacking it).
//   * WebGPU-unavailable message: request_session.js alerts instead of
//     silently falling back to pixel/H.264.
//   * Display-fit: render_loop.js aspect-fits the frame to the canvas.
//   * Fix #1 (below): buffer the newest chunk during VAE warmup instead of
//     dropping it, so token modes render from the first drive input.
//   * Fix #2 (server: emitter/manager): the session header is now sent eagerly
//     at WS attach, so this decoder starts compiling at Connect -- before the
//     user drives; see _onSessionHeader / _initVaeDecoder.
// ---------------------------------------------------------------------------

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
  constructor({ url, codec = null, device = null, canvas = null, controlChannel = null, log = noop, onMetrics = noop }) {
    // SAS token codec: append the selected codec id so the server picks it for
    // this session (?codec=sas-int4-2s / -4s4); default keeps the raw codec.
    this._url = codec
      ? url + (url.includes("?") ? "&" : "?") + "codec=" + encodeURIComponent(codec)
      : url
    this._device = device
    this._canvas = canvas
    this._controlChannel = controlChannel
    this._log = log
    this._onMetrics = onMetrics

    this._socket = null
    this._decoder = null
    this._vaeDecoder = null
    this._renderLoop = null
    this._sessionHeader = null
    this._vaeReady = false
    // Fix #1 (buffer-instead-of-drop): newest chunk that arrived while the VAE
    // decoder was still compiling. Rendered once the decoder is ready so
    // playback starts immediately instead of only after warmup. A single slot
    // (not a backlog) avoids replaying stale driving footage on a live stream.
    this._pendingLatest = null
    this._started = false

    // chunkId -> ordered array of decoded latent frames awaiting completion.
    this._pendingChunks = new Map()
    // chunkId -> wire payload bytes accumulated for the chunk (telemetry).
    this._chunkBytes = new Map()
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
      this._decoder.configure(header?.codec?.static_params ?? {}, header?.latent_shape)
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

    // Prepare the VAE decoder seam. It is expected to be unimplemented in this
    // scaffold, so failure is logged and the session continues assembling and
    // acking chunks without a GPU present step.
    void this._initVaeDecoder(header)
  }

  async _initVaeDecoder(header) {
    if (!this._device) {
      this._log("no WebGPU device; token frames will assemble without decode", {
        source: "client",
      })
      return
    }
    this._vaeDecoder = new VaeDecoder()
    try {
      await this._vaeDecoder.init(header, this._device)
      this._vaeReady = true
      this._log("VAE decoder ready", { source: "client" })
      // Fix #1: render the last chunk buffered during warmup, then go live.
      const pending = this._pendingLatest
      this._pendingLatest = null
      if (pending && this._renderLoop) {
        await this._renderChunk(pending.chunkId, pending.latentFrames, pending.payloadBits)
      }
    } catch (error) {
      this._vaeReady = false
      this._log(`VAE decoder not available: ${error.message}`, {
        source: "client",
      })
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
    this._chunkBytes.set(
      frame.chunkId,
      (this._chunkBytes.get(frame.chunkId) || 0) + (frame.payload?.byteLength ?? 0)
    )

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

    const payloadBits = (this._chunkBytes.get(chunkId) || 0) * 8
    this._chunkBytes.delete(chunkId)

    if (!this._vaeReady || !this._vaeDecoder || !this._renderLoop) {
      // Fix #1 (buffer-instead-of-drop): the WebGPU VAE decoder is still warming
      // up (model download + shader compile). The original code dropped these
      // chunks, so playback only went live once the decoder was ready (the
      // "first ~N chunks are lost" symptom). Instead keep the newest one and
      // render it the moment the decoder is ready. Ack now so the server's flow
      // window keeps advancing exactly as before. One slot (vs a backlog) avoids
      // replaying stale footage on a live stream and cannot leak if the decoder
      // never becomes ready (no WebGPU / init failed).
      this._pendingLatest = { chunkId, latentFrames, payloadBits }
      this._socket?.sendAck(chunkId)
      return
    }

    await this._renderChunk(chunkId, latentFrames, payloadBits)
  }

  /** Decode a chunk's latents, present them, and report client-side telemetry. */
  async _renderChunk(chunkId, latentFrames, payloadBits = null) {
    try {
      const t0 = performance.now()
      const rgb = await this._vaeDecoder.decode(latentFrames)
      const decodeMs = performance.now() - t0
      this._renderLoop.enqueue({ chunkId, frames: rgb.frames ?? [rgb] })
      // Client GPU load (telemetry panel): RGB frames produced per second of
      // decode time, the per-chunk decode latency, and the compressed wire bytes
      // for this chunk. decodeMs is ~pure decode (the decoder's internal queue is
      // empty at steady state) and excludes network receive and the canvas blit.
      const outFrames = rgb.frames?.length ?? 1
      this._onMetrics({
        clientFps: decodeMs > 0 ? (outFrames * 1000) / decodeMs : null,
        clientLatencyMs: decodeMs,
        payloadBits,
      })
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
