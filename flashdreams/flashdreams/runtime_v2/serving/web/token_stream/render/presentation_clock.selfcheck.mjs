// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Runnable self-check for PresentationClock (no framework):
//   node presentation_clock.selfcheck.mjs
// Covers startup baseline, rate adaptation, both clamps, backlog drain, and
// stall re-anchor. Exits non-zero (assert throws) on any failure.

import assert from "node:assert/strict"
import { PresentationClock } from "./presentation_clock.js"

const near = (a, b, tol = 1e-6) => Math.abs(a - b) <= tol

// 1. Startup: due immediately; the first arrival is only a baseline, so the
//    fallback cadence (content fps) holds until a second arrival is observed.
{
  const c = new PresentationClock(30)
  assert.equal(c.isDue(0), true, "null schedule is due")
  assert.ok(near(c.intervalMs, 1000 / 30), "starts at fallback 30 fps")
  c.observeArrival(0, 8) // baseline only
  assert.ok(near(c.intervalMs, 1000 / 30), "baseline arrival keeps fallback")
}

// 2. Adaptation: 8 frames every 1000 ms -> ~8 fps, present slightly faster
//    (x0.98) => interval ~= (1000/8)*0.98 = 122.5 ms, stable across the window.
{
  const c = new PresentationClock(30)
  c.observeArrival(0, 8)
  c.observeArrival(1000, 8)
  assert.ok(near(c.intervalMs, 122.5, 1e-3), `adapts to ~8 fps, got ${c.intervalMs}`)
  c.observeArrival(2000, 8)
  assert.ok(near(c.intervalMs, 122.5, 1e-3), "stays ~8 fps over the window")
}

// 3. Ceiling clamp: very fast arrivals never present faster than the content fps.
{
  const c = new PresentationClock(30)
  c.observeArrival(0, 8)
  c.observeArrival(50, 8) // 160 fps arrival -> would be ~6 ms
  assert.ok(near(c.intervalMs, 1000 / 30, 1e-6), "clamped to 30 fps ceiling")
}

// 4. Floor cap: very slow arrivals never present slower than 5 fps (200 ms).
{
  const c = new PresentationClock(30)
  c.observeArrival(0, 8)
  c.observeArrival(4000, 8) // 2 fps arrival -> would be 490 ms
  assert.ok(near(c.intervalMs, 200, 1e-6), "capped at the 200 ms / 5 fps floor")
}

// 5. Backlog drains immediately at the ceiling rate, regardless of cadence.
{
  const c = new PresentationClock(30)
  c.markAdvanced(0) // schedule next at 0 + 33.33
  assert.equal(c.isDue(10, false), false, "not due mid-interval")
  assert.equal(c.isDue(10, true), true, "backlog is always due")
  c.markAdvanced(10, true)
  assert.equal(c.isDue(10 + 1000 / 30 - 0.001, false), false, "ceiling-paced after drain")
  assert.equal(c.isDue(10 + 1000 / 30 + 0.001, false), true, "due one ceiling interval later")
}

// 6. Stall re-anchor: after a long hold, resume at `now`, not catch-up from the
//    stale schedule (no burst through the backlog).
{
  const c = new PresentationClock(30) // fallback interval 33.33 ms
  const i = 1000 / 30
  c.markAdvanced(0) // nextAt = 33.33
  c.markAdvanced(i) // steady: nextAt = 66.66
  c.markAdvanced(5000) // long stall -> re-anchor to 5000 + i
  assert.equal(c.isDue(5000 + i - 0.001, false), false, "re-anchored to resume time")
  assert.equal(c.isDue(5000 + i + 0.001, false), true, "next due one interval after resume")
}

console.log("presentation_clock.selfcheck: all checks passed")
