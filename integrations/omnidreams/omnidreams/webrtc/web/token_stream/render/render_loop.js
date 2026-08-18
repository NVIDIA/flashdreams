// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Presentation loop skeleton for the token path. A requestAnimationFrame loop
// drains decoded chunks from a queue, blits each frame's RGB result to the
// canvas, and acks the chunk once its last frame has been presented so the
// server can release a flow-control slot.
//
// The GPU blit itself is a stub until the WebGPU VAE decoder exists; the queue,
// pacing, and ack wiring are in place so filling in the blit is a local change.

/**
 * @typedef {Object} DecodedChunk
 * @property {number} chunkId
 * @property {Array<object>} frames per-frame RGB results (texture/buffer)
 */
export class RenderLoop {
  /**
   * @param {object} options
   * @param {HTMLCanvasElement} options.canvas target canvas
   * @param {(chunkId: number) => void} options.onChunkPresented called after a
   *   chunk's last frame is on screen; wired to the socket ack
   * @param {(now: number) => void} [options.onFramePresented] optional per-frame
   *   hook, e.g. for FPS accounting
   */
  constructor({ canvas, onChunkPresented, onFramePresented }) {
    this._canvas = canvas
    this._onChunkPresented = onChunkPresented
    this._onFramePresented = onFramePresented
    this._queue = []
    this._running = false
    this._rafId = null
    this._tick = this._tick.bind(this)
    this._ctx = null
    this._scratch = null
    this._scratchCtx = null
    this._current = null
    this._frameCursor = 0
  }

  /** Begin the animation-frame loop. Idempotent. */
  start() {
    if (this._running) {
      return
    }
    this._running = true
    this._rafId = window.requestAnimationFrame(this._tick)
  }

  /** Stop the loop and drop any queued chunks. */
  stop() {
    this._running = false
    if (this._rafId !== null) {
      window.cancelAnimationFrame(this._rafId)
      this._rafId = null
    }
    this._queue.length = 0
    this._current = null
    this._frameCursor = 0
  }

  /**
   * Enqueue a decoded chunk for presentation.
   * @param {DecodedChunk} decodedChunk
   */
  enqueue(decodedChunk) {
    this._queue.push(decodedChunk)
  }

  _tick(now) {
    if (!this._running) {
      return
    }

    // Present one frame per animation frame so every decoded frame is shown
    // (blitting a whole chunk in a single frame would leave only its last frame
    // visible). The chunk is acked once its last frame is on screen, which
    // paces the server's flow-control window to the display rate.
    if (!this._current) {
      this._current = this._queue.shift() ?? null
      this._frameCursor = 0
    }
    if (this._current) {
      const frames = this._current.frames
      this._blit(frames[this._frameCursor])
      this._onFramePresented?.(now)
      this._frameCursor += 1
      if (this._frameCursor >= frames.length) {
        this._onChunkPresented?.(this._current.chunkId)
        this._current = null
      }
    }

    this._rafId = window.requestAnimationFrame(this._tick)
  }

  _blit(frame) {
    const ctx = this._ctx ?? (this._ctx = this._canvas?.getContext("2d"))
    if (!ctx) {
      return
    }
    // `frame` is an ImageData at the decoder's native resolution. Stage it on a
    // matching scratch canvas, then scale-blit onto the (dpr-sized) display
    // canvas, which cannot take a scaled putImageData directly.
    if (
      !this._scratchCtx ||
      this._scratch.width !== frame.width ||
      this._scratch.height !== frame.height
    ) {
      this._scratch = document.createElement("canvas")
      this._scratch.width = frame.width
      this._scratch.height = frame.height
      this._scratchCtx = this._scratch.getContext("2d")
    }
    this._scratchCtx.putImageData(frame, 0, 0)
    ctx.drawImage(this._scratch, 0, 0, this._canvas.width, this._canvas.height)
  }
}
