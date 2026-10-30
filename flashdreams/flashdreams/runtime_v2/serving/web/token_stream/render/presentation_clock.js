// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Adaptive presentation clock for the token path, ported from the server-side
// pixel path's `_PresentationClock` (runtime_v2/presentation_manager.py) so the
// two streaming paths pace identically.
//
// The pixel path does NOT play at the content fps and freeze on the deficit; it
// presents each frame at the model's *observed* output rate (slightly faster, to
// drain), bounded by a ceiling (never faster than the content fps) and a floor
// (never slower than 5 fps before it is treated as a stall). That is why it looks
// smooth: even cadence at ~the production rate rather than 30 fps bursts + long
// holds. This client clock reproduces it, substituting the model's output rate
// with the rate decoded chunks *arrive* at the render loop — the client analog of
// "frames the model has finished", and the rate presentation must not outrun.
//
// Server parity notes (same constants as presentation_manager.py):
//   - window 2 s, throughput = frames / summed-elapsed over the window
//   - drain margin 0.98 (present ~2% faster than arrival so the buffer drains)
//   - cap 0.20 s (5 fps floor); floor 1/maxFps (the content-fps ceiling)
//   - a full backlog drains immediately (one frame per tick, since isDue ignores
//     the schedule while backlogged); a stall re-anchors with no catch-up burst

const WINDOW_MS = 2000
const DRAIN_MARGIN = 0.98
const MAX_OBSERVED_INTERVAL_MS = 200 // 5 fps floor: cap how slow a cold rate goes

export class PresentationClock {
  /**
   * @param {number} fps nominal/fallback content fps (from the session header)
   * @param {number|null} [maxFps] presentation ceiling; null uses `fps`
   */
  constructor(fps = 30, maxFps = null) {
    maxFps = maxFps || fps
    this._minIntervalMs = 1000 / maxFps // ceiling: never present faster than this
    this._fallbackIntervalMs = Math.max(1000 / fps, this._minIntervalMs)
    this._intervalMs = this._fallbackIntervalMs
    this._nextAt = null
    this._obs = [] // {t, frames, elapsedMs} within the trailing window
    this._lastArrivalAt = null
    this._hasBaseline = false
  }

  /** Current presentation rate, frames per second. */
  get fps() {
    return 1000 / this._intervalMs
  }

  /** Current per-frame interval in milliseconds. */
  get intervalMs() {
    return this._intervalMs
  }

  /**
   * Record one decoded chunk arriving at the render loop and re-estimate the
   * presentation interval from the trailing-window arrival throughput.
   *
   * The first arrival only anchors the clock (its elapsed is unknown and model
   * warmup must not set the steady rate), mirroring the server's completion
   * baseline; the fallback cadence holds until the second arrival.
   *
   * @param {number} now arrival timestamp (performance.now() ms)
   * @param {number} frames decoded frames in the chunk
   */
  observeArrival(now, frames) {
    const elapsedMs = this._lastArrivalAt === null ? 0 : now - this._lastArrivalAt
    this._lastArrivalAt = now
    if (!this._hasBaseline) {
      this._hasBaseline = true
      this._obs.length = 0
      return
    }
    this._obs.push({ t: now, frames, elapsedMs })
    const cutoff = now - WINDOW_MS
    while (this._obs.length && this._obs[0].t <= cutoff) {
      this._obs.shift()
    }
    let f = 0
    let e = 0
    for (const o of this._obs) {
      f += o.frames
      e += o.elapsedMs
    }
    if (e <= 0) {
      return // no span yet; hold the current interval
    }
    const observedFps = (f / e) * 1000
    const target = (1000 / observedFps) * DRAIN_MARGIN
    this._intervalMs = Math.max(
      Math.min(target, MAX_OBSERVED_INTERVAL_MS),
      this._minIntervalMs
    )
  }

  /**
   * Whether the next frame may be presented now. A backlog (a whole chunk
   * waiting behind the active one) drains immediately; otherwise the cadence
   * gates it.
   * @param {number} now current timestamp (ms)
   * @param {boolean} [backlog]
   */
  isDue(now, backlog = false) {
    return backlog || this._nextAt === null || now >= this._nextAt
  }

  /**
   * Record that one frame was presented. Under backlog the next slot is set one
   * ceiling-interval out (isDue drains every tick meanwhile, so it only gates
   * the first frame after the backlog clears); otherwise cadence advances,
   * re-anchoring (no catch-up burst) when the schedule has fallen more than one
   * interval behind, e.g. after a stall.
   * @param {number} now current timestamp (ms)
   * @param {boolean} [backlog]
   */
  markAdvanced(now, backlog = false) {
    if (backlog) {
      this._nextAt = now + this._minIntervalMs
      return
    }
    const interval = this._intervalMs
    if (this._nextAt === null || now - this._nextAt >= interval) {
      this._nextAt = now + interval
    } else {
      this._nextAt = this._nextAt + interval
    }
  }

  /** Discard timing state (e.g. on a reset). */
  reset() {
    this._intervalMs = this._fallbackIntervalMs
    this._nextAt = null
    this._obs.length = 0
    this._lastArrivalAt = null
    this._hasBaseline = false
  }
}
