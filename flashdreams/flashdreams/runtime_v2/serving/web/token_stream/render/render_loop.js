// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Presentation loop for the token path. A requestAnimationFrame loop drains
// decoded chunks from a queue, blits each frame's RGB result to the canvas (one
// frame per animation frame), and acks the chunk once its last frame is on
// screen so the server can release a flow-control slot.
//
// It can also overlay a corner badge or a full-canvas message (the slow
// CPU-decode fallback, or a "token decode unavailable" notice) so a degraded or
// failed decode is visible rather than a silent blank.

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
  constructor({
    canvas,
    onChunkPresented,
    onFramePresented,
    onChunkBuffered = () => {},
    paced = false,
    fps = 30,
    preRoll = 8,
  }) {
    this._canvas = canvas
    this._onChunkPresented = onChunkPresented
    this._onFramePresented = onFramePresented
    this._onChunkBuffered = onChunkBuffered
    this._queue = []
    this._running = false
    this._rafId = null
    this._tick = this._tick.bind(this)
    this._ctx = null
    this._scratch = null
    this._scratchCtx = null
    this._current = null
    this._frameCursor = 0
    this._badge = null // persistent corner label drawn over each presented frame
    // Paced playout (the default; session can disable it): present at a fixed
    // cadence (the advertised fps) through a small jitter buffer, instead of one
    // frame per rAF on arrival, to smooth the chunk-boundary gaps.
    // See docs/token-frame-pacing-plan.md.
    this._paced = paced
    this._interval = 1000 / (fps || 30)
    this._preRoll = Math.max(1, preRoll)
    this._frameQueue = [] // flat {chunkId, frame, isLast} awaiting paced playout
    this._paceStarted = false
    this._nextAt = 0
    this._starved = false
  }

  /** Set (or clear with null) a small corner badge drawn over every frame. */
  setBadge(text) {
    this._badge = text || null
  }

  /**
   * Paint a full-canvas status message (e.g. "token decode unavailable: ...").
   * Used when there is no video to show, so the failure is visible rather than a
   * silent blank — and never a fall back to the pixel path.
   */
  showMessage(text) {
    const ctx = this._ctx ?? (this._ctx = this._canvas?.getContext("2d"))
    if (!ctx) {
      return
    }
    const { width, height } = this._canvas
    ctx.fillStyle = "#101010"
    ctx.fillRect(0, 0, width, height)
    const fontPx = Math.max(13, Math.round(height / 36))
    ctx.fillStyle = "#f0f0f0"
    ctx.font = `${fontPx}px sans-serif`
    ctx.textAlign = "center"
    ctx.textBaseline = "middle"
    // Word-wrap to ~90% width so a long, actionable message stays readable.
    const lines = []
    let line = ""
    for (const word of String(text).split(" ")) {
      const candidate = line ? `${line} ${word}` : word
      if (line && ctx.measureText(candidate).width > width * 0.9) {
        lines.push(line)
        line = word
      } else {
        line = candidate
      }
    }
    if (line) {
      lines.push(line)
    }
    const lineH = fontPx * 1.3
    const top = height / 2 - ((lines.length - 1) * lineH) / 2
    lines.forEach((ln, i) => ctx.fillText(ln, width / 2, top + i * lineH))
  }

  _drawBadge() {
    if (!this._badge || !this._ctx) {
      return
    }
    const ctx = this._ctx
    ctx.save()
    ctx.font = "14px sans-serif"
    ctx.textAlign = "left"
    ctx.textBaseline = "middle"
    const padding = 6
    const boxW = ctx.measureText(this._badge).width + padding * 2
    ctx.fillStyle = "rgba(176,42,42,0.85)"
    ctx.fillRect(8, 8, boxW, 22)
    ctx.fillStyle = "#ffffff"
    ctx.fillText(this._badge, 8 + padding, 8 + 11)
    ctx.restore()
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
    this._frameQueue.length = 0
    this._paceStarted = false
    this._starved = false
  }

  /**
   * Enqueue a decoded chunk for presentation.
   * @param {DecodedChunk} decodedChunk
   */
  enqueue(decodedChunk) {
    if (this._paced) {
      const { chunkId, frames } = decodedChunk
      for (let i = 0; i < frames.length; i += 1) {
        this._frameQueue.push({
          chunkId,
          frame: frames[i],
          isLast: i === frames.length - 1,
        })
      }
      // Ack on buffer, not on present: paced presentation delays frames, and
      // acking on present would throttle the server's flow-control window and
      // starve the buffer (see docs/token-frame-pacing-plan.md §4).
      this._onChunkBuffered(chunkId)
      return
    }
    this._queue.push(decodedChunk)
  }

  _tick(now) {
    if (!this._running) {
      return
    }
    if (this._paced) {
      this._tickPaced(now)
    } else {
      this._tickOnArrival(now)
    }
    this._rafId = window.requestAnimationFrame(this._tick)
  }

  // Default: present one frame per animation frame, draining chunks as they
  // arrive (blitting a whole chunk in one frame would leave only its last frame
  // visible). The chunk is acked once its last frame is on screen, which paces
  // the server's flow-control window to the display rate.
  _tickOnArrival(now) {
    if (!this._current) {
      this._current = this._queue.shift() ?? null
      this._frameCursor = 0
    }
    if (this._current) {
      const frames = this._current.frames
      this._blit(frames[this._frameCursor])
      this._drawBadge()
      this._onFramePresented?.(now)
      this._frameCursor += 1
      if (this._frameCursor >= frames.length) {
        this._onChunkPresented?.(this._current.chunkId)
        this._current = null
      }
    }
  }

  // Paced playout: hold a small pre-roll, then present one buffered frame per
  // ``interval`` (the advertised fps) so the chunk-boundary bursts are smoothed.
  // On starvation, hold the last frame and re-anchor on resume so playback
  // continues at cadence rather than bursting through the backlog.
  _tickPaced(now) {
    if (!this._paceStarted) {
      if (this._frameQueue.length < this._preRoll) {
        return // building the pre-roll buffer
      }
      this._paceStarted = true
      this._nextAt = now
    }
    if (this._frameQueue.length === 0) {
      this._starved = true // hold the last frame; re-anchor when frames resume
      return
    }
    if (now < this._nextAt) {
      return // not yet time for the next frame
    }
    if (this._starved) {
      this._nextAt = now // resume at cadence after a stall, no catch-up burst
      this._starved = false
    }
    const item = this._frameQueue.shift()
    this._blit(item.frame)
    this._drawBadge()
    this._onFramePresented?.(now)
    if (item.isLast) {
      this._onChunkPresented?.(item.chunkId)
    }
    this._nextAt += this._interval
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
