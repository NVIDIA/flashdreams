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

    // Present at most one chunk per animation frame to pace playback.
    const chunk = this._queue.shift()
    if (chunk) {
      for (const frame of chunk.frames) {
        this._blit(frame)
        this._onFramePresented?.(now)
      }
      // The chunk is fully on screen; release the server flow-control slot.
      this._onChunkPresented?.(chunk.chunkId)
    }

    this._rafId = window.requestAnimationFrame(this._tick)
  }

  _blit(frame) {
    // TODO: copy the decoded RGB texture/buffer to the canvas via WebGPU once
    // the VAE decoder produces real output. Kept as a stub so the loop can run
    // without a decoder.
    void frame
    void this._canvas
  }
}
