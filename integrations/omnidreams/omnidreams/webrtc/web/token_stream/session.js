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

// Debug/telemetry gate (see flashdreams/serving/debug.py for the server side).
// Add ?debug to the URL to enable latent capture-for-download and the extra
// per-chunk instrumentation. Off by default so production pays no memory cost.
const FD_DEBUG = new URLSearchParams(location.search).has("debug")

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
    this._lastPresentT = null // wall time of the last presented chunk (client FPS)
    // chunkId -> accumulated pure SAS-unpack (dequantize) time for the chunk (ms).
    this._chunkUnpackMs = new Map()

    // --- Latent capture (download-for-offline) --------------------------------
    // Buffer every frame's latent as it streams in, so the user can download all
    // latents generated so far. For the raw_f16 codec we keep the EXACT fp16 wire
    // bytes (byte-identical, .npy dtype '<f2'); for other codecs we fall back to
    // the decoded latent values as float32 ('<f4', lossy for SAS). Capped so a
    // long session can't exhaust memory.
    this._cap = { on: FD_DEBUG, frames: [], dtype: null, frameShape: null, truncated: false }
    this._capMaxFrames = 20000 // ~2500 chunks safety cap
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
    const unpackT0 = performance.now()
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

    // Pure primitive: accumulate the SAS-unpack (dequantize) time for this chunk.
    if (FD_DEBUG) {
      this._chunkUnpackMs.set(
        frame.chunkId,
        (this._chunkUnpackMs.get(frame.chunkId) || 0) + (performance.now() - unpackT0)
      )
    }

    // Latent capture for offline download (buffers as frames arrive).
    if (this._cap.on) this._recordLatent(frame, latent)

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
      `token chunk ${chunkId}: ${latentFrames.length} frames, ${codecId}, ${decodedFloats} latent floats -> VAE decode`,
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
      // Pre-existing telemetry: wall time around decode(), and decode-capacity FPS.
      let decodeMs = performance.now() - t0
      this._renderLoop.enqueue({ chunkId, frames: rgb.frames ?? [rgb] })
      const outFrames = rgb.frames?.length ?? 1
      let clientFps = decodeMs > 0 ? (outFrames * 1000) / decodeMs : null
      const _extra = {}
      if (FD_DEBUG) {
        // --- added instrumentation only (see flashdreams/serving/debug.py) ---
        // Prefer the decoder's PURE GPU decode time; the wall time above also
        // includes the _chain queue wait under pileup, which inflates the panel.
        if (Number.isFinite(rgb.decodeMs)) decodeMs = rgb.decodeMs
        // Real display FPS = frames presented per wall-clock second across chunk
        // arrivals (gen-bound), NOT frames/decode-time (= decode capacity).
        const nowT = performance.now()
        clientFps = null
        if (this._lastPresentT != null) {
          const dWall = (nowT - this._lastPresentT) / 1000
          if (dWall > 0) clientFps = outFrames / dWall
        }
        this._lastPresentT = nowT
        // Per-chunk SAS-unpack (dequantize) time (0 for raw).
        _extra.sasUnpackMs = this._chunkUnpackMs.get(chunkId) || 0
        this._chunkUnpackMs.delete(chunkId)
      }
      this._onMetrics({
        clientFps,
        clientLatencyMs: decodeMs,
        ..._extra,
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

  // --- Latent capture / download -------------------------------------------

  /** Buffer one frame's latent for later download. */
  _recordLatent(frame, latent) {
    const cap = this._cap
    if (cap.frames.length >= this._capMaxFrames) {
      cap.truncated = true
      return
    }
    const codecId = this._sessionHeader?.codec?.id
    if (codecId === "raw_f16" && frame.payload != null) {
      // Exact fp16 wire bytes (little-endian half). .npy dtype '<f2'.
      if (!cap.dtype) cap.dtype = "<f2"
      const u8 =
        frame.payload instanceof Uint8Array
          ? frame.payload
          : new Uint8Array(frame.payload)
      cap.frames.push(u8.slice()) // copy so the socket buffer can be reused
    } else {
      // Non-raw codec (e.g. SAS): keep decoded latent values as float32 ('<f4').
      if (!cap.dtype) cap.dtype = "<f4"
      cap.frames.push(Float32Array.from(latent))
    }
    if (!cap.frameShape && Array.isArray(this._sessionHeader?.latent_shape)) {
      cap.frameShape = this._sessionHeader.latent_shape.slice()
    }
  }

  /** Frames buffered so far + byte size (for the UI). */
  latentCaptureStats() {
    let bytes = 0
    for (const f of this._cap.frames) bytes += f.byteLength
    return { frames: this._cap.frames.length, bytes, truncated: this._cap.truncated }
  }

  /**
   * Serialize all captured latents into a single NumPy .npy file (loadable with
   * ``np.load``). Shape is ``[frames, ...latent_shape]``; dtype is ``<f2`` for
   * the exact-fp16 raw codec, else ``<f4``. Returns ``null`` if nothing buffered.
   */
  exportLatentsNpy() {
    const cap = this._cap
    if (!cap.frames.length) return null
    const dtype = cap.dtype || "<f4"
    const elemBytes = dtype === "<f2" ? 2 : 4
    const nframes = cap.frames.length
    let dataBytes = 0
    for (const f of cap.frames) dataBytes += f.byteLength
    const perFrameElems = dataBytes / elemBytes / nframes
    let shape
    if (cap.frameShape && cap.frameShape.reduce((a, b) => a * b, 1) === perFrameElems) {
      shape = [nframes, ...cap.frameShape]
    } else {
      shape = [nframes, Math.round(perFrameElems)]
    }
    const data = new Uint8Array(dataBytes)
    let off = 0
    for (const f of cap.frames) {
      const u8 =
        f instanceof Uint8Array ? f : new Uint8Array(f.buffer, f.byteOffset, f.byteLength)
      data.set(u8, off)
      off += u8.byteLength
    }
    const header = this._npyHeader(dtype, shape)
    const blob = new Blob([header, data], { type: "application/octet-stream" })
    const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)
    return {
      blob,
      filename: `latents_${ts}.npy`,
      frames: nframes,
      bytes: header.byteLength + dataBytes,
      dtype,
      shape,
      truncated: cap.truncated,
    }
  }

  /** Build a NumPy v1.0 .npy header (64-byte aligned) for dtype + shape. */
  _npyHeader(dtype, shape) {
    const shapeStr = shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(", ")})`
    let dict = `{'descr': '${dtype}', 'fortran_order': False, 'shape': ${shapeStr}, }`
    const unpadded = 10 + dict.length + 1 // 8 magic+ver, 2 len, +1 trailing \n
    const pad = (64 - (unpadded % 64)) % 64
    dict = dict + " ".repeat(pad) + "\n"
    const dictBytes = new TextEncoder().encode(dict)
    const out = new Uint8Array(10 + dictBytes.byteLength)
    out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 0x01, 0x00], 0) // \x93NUMPY v1.0
    out[8] = dictBytes.byteLength & 0xff
    out[9] = (dictBytes.byteLength >> 8) & 0xff
    out.set(dictBytes, 10)
    return out
  }
}
