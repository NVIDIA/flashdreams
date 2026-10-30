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

import { PresentationClock } from "./presentation_clock.js"

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
  }) {
    this._canvas = canvas
    this._onChunkPresented = onChunkPresented
    this._onFramePresented = onFramePresented
    this._onChunkBuffered = onChunkBuffered
    this._renderer = null // GPU-direct renderer; null => CPU blit path
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
    // Paced playout (the default; session can disable it with ?pace=off):
    // present through an adaptive presentation clock instead of one frame per rAF
    // on arrival, so the chunk-boundary gaps become an even cadence.
    this._paced = paced
    // The clock is a parity port of the pixel path's server-side
    // _PresentationClock (runtime_v2/presentation_manager.py): it paces at the
    // observed chunk-arrival rate (x0.98 drain), not a fixed fps, so an even
    // cadence replaces the 30 fps bursts + deficit freezes. See
    // docs/token-frame-pacing-plan.md.
    this._clock = new PresentationClock(fps || 30)
    this._frameQueue = [] // flat {chunkId, frame, isLast} awaiting paced playout
    this._chunksBuffered = 0 // complete chunks with frames still queued (backlog)
    this._paceStarted = false
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
    // GPU-direct: dispose any held chunk buffers still queued (return them to
    // ORT's pool instead of leaking).
    this._disposeQueuedGpuBuffers()
    this._queue.length = 0
    this._current = null
    this._frameCursor = 0
    this._frameQueue.length = 0
    this._chunksBuffered = 0
    this._paceStarted = false
    this._starved = false
    this._clock.reset()
  }

  /** Set the GPU-direct renderer; when present, frames draw via it (no blit). */
  setRenderer(renderer) {
    this._renderer = renderer
  }

  _disposeQueuedGpuBuffers() {
    // Each chunk's dispose rides its last frame item (paced) or the chunk (arrival).
    for (const item of this._frameQueue) {
      item.dispose?.()
    }
    for (const chunk of this._queue) {
      chunk.dispose?.()
    }
    if (this._current && this._current.gpu) {
      this._current.dispose?.()
    }
  }

  /**
   * Enqueue a decoded chunk for presentation.
   * @param {DecodedChunk} decodedChunk
   */
  enqueue(decodedChunk) {
    if (this._paced) {
      const { chunkId } = decodedChunk
      if (decodedChunk.gpu) {
        // GPU-direct: frames reference the shared chunk buffer; the chunk's
        // dispose rides the LAST frame so the buffer frees once fully presented.
        const n = decodedChunk.frameCount
        for (let i = 0; i < n; i += 1) {
          this._frameQueue.push({
            chunkId,
            buffer: decodedChunk.buffer,
            frameIndex: i,
            isLast: i === n - 1,
            dispose: i === n - 1 ? decodedChunk.dispose : null,
          })
        }
      } else {
        const frames = decodedChunk.frames
        for (let i = 0; i < frames.length; i += 1) {
          this._frameQueue.push({
            chunkId,
            frame: frames[i],
            isLast: i === frames.length - 1,
          })
        }
      }
      // Feed the clock the chunk's arrival so it can pace to the observed rate.
      this._chunksBuffered += 1
      this._clock.observeArrival(
        typeof performance !== "undefined" ? performance.now() : Date.now(),
        decodedChunk.gpu ? decodedChunk.frameCount : decodedChunk.frames.length
      )
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
      const cur = this._current
      const n = cur.gpu ? cur.frameCount : cur.frames.length
      if (cur.gpu) {
        this._renderer.present(cur.buffer, this._frameCursor)
      } else {
        this._blit(cur.frames[this._frameCursor])
        this._drawBadge()
      }
      this._onFramePresented?.(now)
      this._frameCursor += 1
      if (this._frameCursor >= n) {
        this._onChunkPresented?.(cur.chunkId)
        if (cur.gpu) {
          cur.dispose?.() // GPU-direct: free the chunk buffer after its last frame
        }
        this._current = null
      }
    }
  }

  // Adaptive paced playout (pixel-path parity): present buffered frames at the
  // clock's cadence, which tracks the observed chunk-arrival rate. An empty
  // buffer holds the last frame (a deficit stall); a whole chunk waiting behind
  // the active one drains at the ceiling rate so latency does not grow. On
  // resume the clock re-anchors, so a stall never bursts through the backlog.
  _tickPaced(now) {
    if (this._frameQueue.length === 0) {
      // A real stall is "due but empty": the x0.98 drain margin briefly empties
      // the queue within a frame's own slot, which is not a visible stall, so
      // only count a starvation once a frame is actually due.
      if (this._paceStarted && !this._starved && this._clock.isDue(now)) {
        this._starved = true // hold the last frame until frames resume
      }
      return
    }
    if (this._starved) {
      this._starved = false
    }
    this._paceStarted = true
    const backlog = this._chunksBuffered > 1
    if (!this._clock.isDue(now, backlog)) {
      return // not yet time for the next frame
    }
    const item = this._frameQueue.shift()
    this._present(item)
    this._onFramePresented?.(now)
    if (item.isLast) {
      this._chunksBuffered -= 1
      this._onChunkPresented?.(item.chunkId)
      item.dispose?.() // GPU-direct: free the chunk buffer after its last frame
    }
    this._clock.markAdvanced(now, backlog)
  }

  // Present one paced frame item: GPU-direct draws the buffer, else blits ImageData.
  _present(item) {
    if (item.buffer !== undefined) {
      this._renderer.present(item.buffer, item.frameIndex)
    } else {
      this._blit(item.frame)
      this._drawBadge()
    }
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
