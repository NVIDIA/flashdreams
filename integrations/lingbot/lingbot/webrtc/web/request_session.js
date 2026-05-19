// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

const connectButton = document.getElementById("connectButton")
const statusText = document.getElementById("statusText")
const flowText = document.getElementById("flowText")
const eventLog = document.getElementById("eventLog")
const logState = document.getElementById("logState")
const remoteVideo = document.getElementById("remoteVideo")
const mockCanvas = document.getElementById("mockCanvas")
const fpsValue = document.getElementById("fpsValue")
const latencyValue = document.getElementById("latencyValue")
const resolutionValue = document.getElementById("resolutionValue")
const stepValue = document.getElementById("stepValue")
const modelValue = document.getElementById("modelValue")
const controlButtons = Array.from(document.querySelectorAll("[data-control-key]"))

const params = new URLSearchParams(window.location.search)
const mockMode = params.has("mock") && params.get("mock") !== "0"
const allowedKeys = new Set(["w", "a", "s", "d", "q", "e", "i", "j", "k", "l"])
const keySources = new Map()
const heldKeyOrder = new Map()
const activeKeys = new Set()
const frameTimes = []
const pendingActions = []
const maxPendingActions = 32

let peerConnection = null
let controlChannel = null
let statsTimer = null
let inferenceInFlight = false
let connected = false
let heldKeySequence = 0
let mockChunkIndex = 0
let mockGenerationStarted = false
let mockChunkTimer = null

const metrics = {
  fps: null,
  targetFps: null,
  latencyMs: null,
  rttMs: null,
  resolution: null,
  step: null,
  model: "Lingbot",
}

function normalizeKey(rawKey) {
  return String(rawKey || "").toLowerCase()
}

function formatTime() {
  return new Date().toLocaleTimeString([], { hour12: false })
}

function firstFinite(...values) {
  for (const value of values) {
    const number = Number(value)
    if (Number.isFinite(number)) {
      return number
    }
  }
  return null
}

function formatMs(value) {
  if (!Number.isFinite(value)) {
    return "--"
  }
  if (value >= 1000) {
    return `${(value / 1000).toFixed(1)} s`
  }
  return `${Math.round(value)} ms`
}

function logEvent(message, { source = "server", level = "info" } = {}) {
  const entry = document.createElement("div")
  entry.className = `logEntry is-${source}`
  if (level === "error") {
    entry.classList.add("is-error")
  }

  const time = document.createElement("time")
  time.textContent = `[${formatTime()}]`
  const body = document.createElement("span")
  body.textContent = message
  entry.append(time, body)
  eventLog.prepend(entry)

  while (eventLog.children.length > 36) {
    eventLog.lastElementChild.remove()
  }
}

function setStatus(message, state = message.toLowerCase()) {
  statusText.textContent = message
  document.body.dataset.status = state
  logState.textContent = state === "idle" ? "Waiting" : message
}

function setFlow(message) {
  flowText.textContent = message
}

function setVideoVisible(visible) {
  document.body.classList.toggle("has-video", visible)
}

function renderMetrics() {
  const fps = firstFinite(metrics.fps, metrics.targetFps)
  fpsValue.textContent = Number.isFinite(fps) ? String(Math.round(fps)) : "--"
  latencyValue.textContent = formatMs(metrics.latencyMs)
  resolutionValue.textContent = metrics.resolution || "--"
  stepValue.textContent = metrics.step === null ? "--" : String(metrics.step)
  modelValue.textContent = metrics.model || "Lingbot"
}

function updateMetricsFromChunk(payload) {
  const observedLatencyMs = takeObservedActionLatency()
  metrics.targetFps = firstFinite(payload.fps, payload.target_fps, metrics.targetFps)
  metrics.latencyMs = firstFinite(
    payload.latency_ms,
    payload.control_latency_ms,
    observedLatencyMs,
    payload.lag_ms,
    payload.gen_ms,
    metrics.latencyMs
  )
  metrics.step = Number.isFinite(Number(payload.chunk_index))
    ? Number(payload.chunk_index)
    : metrics.step
  metrics.model = typeof payload.model === "string" && payload.model ? payload.model : metrics.model

  if (typeof payload.resolution === "string") {
    metrics.resolution = payload.resolution
  } else if (payload.resolution && typeof payload.resolution === "object") {
    const width = Number(payload.resolution.width)
    const height = Number(payload.resolution.height)
    if (Number.isFinite(width) && Number.isFinite(height)) {
      metrics.resolution = `${width}x${height}`
    }
  }
  renderMetrics()
}

function updateMetricsFromVideo() {
  if (remoteVideo.videoWidth > 0 && remoteVideo.videoHeight > 0) {
    metrics.resolution = `${remoteVideo.videoWidth}x${remoteVideo.videoHeight}`
    renderMetrics()
  }
}

function recordFrame(timestamp) {
  const now = Number.isFinite(timestamp) ? timestamp : performance.now()
  frameTimes.push(now)
  while (frameTimes.length > 0 && now - frameTimes[0] > 1200) {
    frameTimes.shift()
  }
  if (frameTimes.length >= 2) {
    const elapsed = frameTimes[frameTimes.length - 1] - frameTimes[0]
    metrics.fps = elapsed > 0 ? ((frameTimes.length - 1) * 1000) / elapsed : metrics.fps
    renderMetrics()
  }
}

function updateControlHighlights() {
  activeKeys.clear()
  for (const [key, sources] of keySources.entries()) {
    if (sources.size > 0) {
      activeKeys.add(key)
    }
  }
  for (const button of controlButtons) {
    const key = button.dataset.controlKey
    button.classList.toggle("is-active", activeKeys.has(key))
    button.setAttribute("aria-pressed", activeKeys.has(key) ? "true" : "false")
  }
}

function actionLabel(action) {
  return `${action.event}${action.key ? `:${action.key}` : ""}`
}

function recordActionSent(action) {
  pendingActions.push({
    sentAt: performance.now(),
    label: actionLabel(action),
  })
  while (pendingActions.length > maxPendingActions) {
    pendingActions.shift()
  }
}

function takeObservedActionLatency(now = performance.now()) {
  if (pendingActions.length === 0) {
    return null
  }
  const oldest = pendingActions[0]
  pendingActions.length = 0
  return Math.max(0, now - oldest.sentAt)
}

function sendControlAction(action) {
  if (mockMode && connected && !controlChannel) {
    inferenceInFlight = true
    mockGenerationStarted = true
    recordActionSent(action)
    setStatus("Generating", "generating")
    setFlow(`sent ${actionLabel(action)}, waiting=true`)
    logEvent(`control ${actionLabel(action)}`, { source: "client" })
    return true
  }

  if (!connected || !controlChannel || controlChannel.readyState !== "open") {
    return false
  }

  inferenceInFlight = true
  controlChannel.send(
    JSON.stringify({
      type: "action",
      action,
    })
  )
  recordActionSent(action)
  setStatus("Generating", "generating")
  setFlow(`sent ${actionLabel(action)}, waiting=${inferenceInFlight}`)
  logEvent(`control ${actionLabel(action)}`, { source: "client" })
  return true
}

function enqueueAction(action) {
  const sent = sendControlAction(action)
  if (!sent) {
    setFlow(connected ? `not_sent ${actionLabel(action)}` : "connect session first")
  }
}

function enqueueHeldKeyRepeats() {
  const heldKeys = Array.from(activeKeys).sort((a, b) => {
    return (heldKeyOrder.get(a) || 0) - (heldKeyOrder.get(b) || 0)
  })
  for (const key of heldKeys) {
    enqueueAction({ event: "keydown", key })
  }
}

function setKeyHeld(key, source, held) {
  const normalized = normalizeKey(key)
  if (!allowedKeys.has(normalized)) {
    return
  }

  let sources = keySources.get(normalized)
  if (!sources) {
    sources = new Set()
    keySources.set(normalized, sources)
  }

  const wasActive = sources.size > 0
  if (held) {
    sources.add(source)
  } else {
    sources.delete(source)
  }
  const isActive = sources.size > 0
  updateControlHighlights()

  if (held && !wasActive && isActive) {
    heldKeySequence += 1
    heldKeyOrder.set(normalized, heldKeySequence)
    enqueueAction({ event: "keydown", key: normalized })
  }
  if (!held && wasActive && !isActive) {
    heldKeyOrder.delete(normalized)
    enqueueAction({ event: "keyup", key: normalized })
  }
}

function releaseAllKeys() {
  for (const key of Array.from(keySources.keys())) {
    const sources = keySources.get(key)
    if (sources && sources.size > 0) {
      sources.clear()
      heldKeyOrder.delete(key)
      updateControlHighlights()
      enqueueAction({ event: "keyup", key })
    }
  }
}

function handleControlMessage(rawMessage) {
  let payload
  try {
    payload = JSON.parse(rawMessage)
  } catch (error) {
    logEvent(`invalid control payload: ${rawMessage}`, { level: "error" })
    return
  }

  if (payload.type === "chunk_done") {
    inferenceInFlight = false
    updateMetricsFromChunk(payload)
    const genMs = firstFinite(payload.gen_ms)
    const lagMs = firstFinite(payload.lag_ms)
    const queueDepth = firstFinite(payload.queue_depth)
    const parts = [
      `chunk_done index=${payload.chunk_index}`,
      `frames=${payload.num_frames}`,
      `enqueued=${payload.enqueued_frames}`,
    ]
    if (genMs !== null) {
      parts.push(`gen=${Math.round(genMs)}ms`)
    }
    if (lagMs !== null) {
      parts.push(`lag=${Math.round(lagMs)}ms`)
    }
    if (metrics.latencyMs !== null) {
      parts.push(`latency=${Math.round(metrics.latencyMs)}ms`)
    }
    if (queueDepth !== null) {
      parts.push(`queue=${queueDepth}`)
    }
    logEvent(parts.join(", "))
    setStatus(activeKeys.size > 0 ? "Generating" : "Waiting", activeKeys.size > 0 ? "generating" : "waiting")
    setFlow(`chunk ${payload.chunk_index} complete`)
    if (activeKeys.size > 0) {
      enqueueHeldKeyRepeats()
    }
    return
  }

  if (payload.type === "server_log") {
    logEvent(payload.message || "server log")
    return
  }

  if (payload.type === "busy") {
    logEvent(`server busy: ${payload.message}`, { level: "error" })
    setStatus("Waiting", "waiting")
    return
  }

  if (payload.type === "error") {
    inferenceInFlight = false
    logEvent(`server error: ${payload.message}`, { level: "error" })
    setStatus("Error", "error")
    setFlow("server error")
    return
  }

  logEvent(`server message: ${rawMessage}`)
}

async function waitForIceGatheringComplete(pc) {
  if (pc.iceGatheringState === "complete") {
    return
  }
  await new Promise((resolve) => {
    const onStateChange = () => {
      if (pc.iceGatheringState === "complete") {
        pc.removeEventListener("icegatheringstatechange", onStateChange)
        resolve()
      }
    }
    pc.addEventListener("icegatheringstatechange", onStateChange)
  })
}

async function pollWebRtcStats() {
  if (!peerConnection) {
    return
  }
  try {
    const stats = await peerConnection.getStats()
    for (const report of stats.values()) {
      if (
        report.type === "candidate-pair" &&
        report.state === "succeeded" &&
        Number.isFinite(report.currentRoundTripTime)
      ) {
        metrics.rttMs = report.currentRoundTripTime * 1000
      }
      if (
        report.type === "inbound-rtp" &&
        (report.kind === "video" || report.mediaType === "video") &&
        Number.isFinite(report.framesPerSecond)
      ) {
        metrics.fps = report.framesPerSecond
      }
    }
    renderMetrics()
  } catch (error) {
    logEvent(`stats unavailable: ${error.message}`, { source: "client" })
  }
}

function startStatsPolling() {
  if (statsTimer !== null) {
    return
  }
  statsTimer = window.setInterval(() => {
    void pollWebRtcStats()
  }, 1000)
}

function stopStatsPolling() {
  if (statsTimer !== null) {
    window.clearInterval(statsTimer)
    statsTimer = null
  }
}

async function connectSession() {
  if (mockMode) {
    await startMockSession()
    return
  }

  connectButton.disabled = true
  setStatus("Connecting", "connecting")
  setFlow("creating peer connection")
  logEvent("connecting to server...", { source: "client" })

  try {
    peerConnection = new RTCPeerConnection()
    controlChannel = peerConnection.createDataChannel("controls")
    peerConnection.addTransceiver("video", { direction: "recvonly" })

    controlChannel.onopen = () => {
      logEvent("control data channel open")
      setFlow("ready for action")
    }
    controlChannel.onclose = () => {
      logEvent("control data channel closed")
      setFlow("channel closed")
    }
    controlChannel.onmessage = (event) => {
      handleControlMessage(event.data)
    }

    peerConnection.ontrack = (event) => {
      const [stream] = event.streams
      if (stream) {
        remoteVideo.srcObject = stream
        updateMetricsFromVideo()
      }
    }

    peerConnection.onconnectionstatechange = () => {
      const state = peerConnection.connectionState
      logEvent(`connection_state=${state}`, { source: "client" })
      if (state === "connected") {
        connected = true
        setStatus("Waiting", "waiting")
        setFlow("connected; waiting for input")
        startStatsPolling()
        return
      }
      if (state === "connecting") {
        setStatus("Connecting", "connecting")
        return
      }
      if (["failed", "closed", "disconnected"].includes(state)) {
        connected = false
        stopStatsPolling()
        setStatus(state === "failed" ? "Error" : "Idle", state === "failed" ? "error" : "idle")
      }
    }

    const offer = await peerConnection.createOffer()
    await peerConnection.setLocalDescription(offer)
    await waitForIceGatheringComplete(peerConnection)

    const response = await fetch("/api/webrtc/offer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(peerConnection.localDescription),
    })
    if (!response.ok) {
      const text = await response.text()
      throw new Error(`offer failed (${response.status}): ${text}`)
    }
    const answer = await response.json()
    await peerConnection.setRemoteDescription(answer)
    logEvent("offer/answer completed")
  } catch (error) {
    connected = false
    setStatus("Error", "error")
    setFlow("failed")
    logEvent(`connect failed: ${error.message}`, { source: "client", level: "error" })
    connectButton.disabled = false
  }
}

function handleKeyDown(event) {
  const key = normalizeKey(event.key)
  if (!allowedKeys.has(key)) {
    return
  }
  event.preventDefault()

  if (event.repeat) {
    return
  }
  setKeyHeld(key, `keyboard:${key}`, true)
}

function handleKeyUp(event) {
  const key = normalizeKey(event.key)
  if (!allowedKeys.has(key)) {
    return
  }
  event.preventDefault()
  setKeyHeld(key, `keyboard:${key}`, false)
}

function attachPointerControls() {
  for (const button of controlButtons) {
    const key = button.dataset.controlKey
    button.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) {
        return
      }
      event.preventDefault()
      button.setPointerCapture(event.pointerId)
      setKeyHeld(key, `pointer:${event.pointerId}`, true)
    })
    button.addEventListener("pointerup", (event) => {
      event.preventDefault()
      setKeyHeld(key, `pointer:${event.pointerId}`, false)
    })
    button.addEventListener("pointercancel", (event) => {
      setKeyHeld(key, `pointer:${event.pointerId}`, false)
    })
    button.addEventListener("lostpointercapture", (event) => {
      setKeyHeld(key, `pointer:${event.pointerId}`, false)
    })
  }
}

function resizeCanvas(ctx) {
  const rect = mockCanvas.getBoundingClientRect()
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  const width = Math.max(1, Math.floor(rect.width * dpr))
  const height = Math.max(1, Math.floor(rect.height * dpr))
  if (mockCanvas.width !== width || mockCanvas.height !== height) {
    mockCanvas.width = width
    mockCanvas.height = height
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  return { width: rect.width, height: rect.height }
}

function drawMountain(ctx, points, fill) {
  ctx.beginPath()
  ctx.moveTo(points[0][0], points[0][1])
  for (const point of points.slice(1)) {
    ctx.lineTo(point[0], point[1])
  }
  ctx.closePath()
  ctx.fillStyle = fill
  ctx.fill()
}

function drawMockScene(now) {
  const ctx = mockCanvas.getContext("2d")
  const { width, height } = resizeCanvas(ctx)
  const t = now * 0.001
  const horizon = height * 0.44

  const sky = ctx.createLinearGradient(0, 0, width, height)
  sky.addColorStop(0, "#718697")
  sky.addColorStop(0.48, "#c8d8da")
  sky.addColorStop(1, "#f4bf77")
  ctx.fillStyle = sky
  ctx.fillRect(0, 0, width, height)

  drawMountain(
    ctx,
    [
      [0, horizon + 40],
      [width * 0.18, height * 0.15],
      [width * 0.35, horizon + 18],
      [width * 0.52, height * 0.22],
      [width * 0.72, horizon + 30],
      [width, height * 0.30],
      [width, height],
      [0, height],
    ],
    "rgba(36, 54, 56, 0.90)"
  )
  drawMountain(
    ctx,
    [
      [width * 0.18, horizon + 36],
      [width * 0.34, height * 0.25],
      [width * 0.50, horizon + 12],
      [width * 0.67, height * 0.31],
      [width, horizon + 26],
      [width, height],
      [width * 0.18, height],
    ],
    "rgba(72, 93, 88, 0.72)"
  )

  const water = ctx.createLinearGradient(width * 0.58, horizon, width, height)
  water.addColorStop(0, "rgba(166, 196, 199, 0.78)")
  water.addColorStop(1, "rgba(61, 85, 93, 0.92)")
  ctx.fillStyle = water
  ctx.beginPath()
  ctx.moveTo(width * 0.54, horizon + 36)
  ctx.lineTo(width, horizon + 8)
  ctx.lineTo(width, height)
  ctx.lineTo(width * 0.64, height)
  ctx.closePath()
  ctx.fill()

  const road = ctx.createLinearGradient(width * 0.35, horizon, width * 0.45, height)
  road.addColorStop(0, "#424a4b")
  road.addColorStop(1, "#17191a")
  ctx.fillStyle = road
  ctx.beginPath()
  ctx.moveTo(width * 0.36, horizon + 30)
  ctx.lineTo(width * 0.60, horizon + 26)
  ctx.lineTo(width * 0.70, height)
  ctx.lineTo(width * 0.18, height)
  ctx.closePath()
  ctx.fill()

  ctx.strokeStyle = "rgba(255, 220, 105, 0.72)"
  ctx.lineWidth = 3
  ctx.beginPath()
  ctx.moveTo(width * 0.50, horizon + 32)
  ctx.lineTo(width * 0.56, height)
  ctx.stroke()

  ctx.strokeStyle = "rgba(240, 246, 242, 0.58)"
  ctx.lineWidth = 2
  for (let i = 0; i < 12; i += 1) {
    const y = horizon + 50 + ((i * 52 + t * 80) % (height - horizon + 90))
    const scale = (y - horizon) / (height - horizon)
    ctx.beginPath()
    ctx.moveTo(width * (0.42 + scale * 0.02), y)
    ctx.lineTo(width * (0.46 + scale * 0.04), y + 18 + scale * 20)
    ctx.stroke()
  }

  ctx.fillStyle = "rgba(36, 38, 35, 0.88)"
  for (let i = 0; i < 7; i += 1) {
    const x = width * (0.02 + i * 0.055)
    const y = horizon + 18 - i * 3
    const buildingWidth = width * 0.046
    const buildingHeight = height * (0.16 + (i % 3) * 0.035)
    ctx.fillRect(x, y - buildingHeight, buildingWidth, buildingHeight)
    ctx.fillStyle = "rgba(255, 214, 142, 0.58)"
    ctx.fillRect(x + 8, y - buildingHeight + 18, 7, 11)
    ctx.fillRect(x + buildingWidth - 15, y - buildingHeight + 42, 7, 11)
    ctx.fillStyle = "rgba(36, 38, 35, 0.88)"
  }

  ctx.strokeStyle = "rgba(19, 42, 30, 0.94)"
  ctx.lineWidth = 5
  for (let i = 0; i < 6; i += 1) {
    const x = width * (0.19 + i * 0.035)
    const treeBase = horizon + 55 + i * 12
    ctx.beginPath()
    ctx.moveTo(x, treeBase)
    ctx.lineTo(x, treeBase - height * 0.18)
    ctx.stroke()
    ctx.fillStyle = "rgba(33, 77, 46, 0.85)"
    ctx.beginPath()
    ctx.ellipse(x, treeBase - height * 0.12, 8, 44, 0, 0, Math.PI * 2)
    ctx.fill()
  }

  ctx.fillStyle = `rgba(255, 255, 255, ${0.10 + Math.sin(t) * 0.025})`
  ctx.fillRect(0, 0, width, height)

  if (!document.body.classList.contains("has-video")) {
    recordFrame(now)
  }
  window.requestAnimationFrame(drawMockScene)
}

function startVideoFrameMonitor() {
  if (typeof remoteVideo.requestVideoFrameCallback !== "function") {
    return
  }
  const onFrame = (now) => {
    if (document.body.classList.contains("has-video")) {
      recordFrame(now)
      updateMetricsFromVideo()
    }
    remoteVideo.requestVideoFrameCallback(onFrame)
  }
  remoteVideo.requestVideoFrameCallback(onFrame)
}

function mockChunkPayload() {
  const numFrames = 12
  const targetFps = 16
  const genMs = 360 + Math.random() * 120
  const lagMs = 54 + Math.random() * 36
  return {
    type: "chunk_done",
    chunk_index: mockChunkIndex++,
    num_frames: numFrames,
    enqueued_frames: numFrames,
    fps: targetFps,
    resolution: { width: 1280, height: 720 },
    model: "lingbot-world-fast-flash",
    latency_ms: 118 + Math.random() * 48,
    consumed_actions: 1,
    gen_ms: genMs,
    enqueue_ms: 8 + Math.random() * 4,
    play_ms: (numFrames * 1000) / targetFps,
    lag_ms: lagMs,
    queue_depth: Math.floor(3 + Math.random() * 7),
  }
}

function ensureMockChunks() {
  if (mockChunkTimer !== null) {
    return
  }
  mockChunkTimer = window.setInterval(() => {
    if (!connected || !mockGenerationStarted) {
      return
    }
    handleControlMessage(JSON.stringify(mockChunkPayload()))
  }, 760)
}

async function startMockSession() {
  connectButton.disabled = true
  setStatus("Connecting", "connecting")
  setFlow("mock warmup")
  logEvent("connecting to mock server...", { source: "client" })
  await new Promise((resolve) => {
    window.setTimeout(resolve, 260)
  })
  connected = true
  metrics.targetFps = 16
  metrics.resolution = "1280x720"
  metrics.model = "lingbot-world-fast-flash"
  renderMetrics()
  setStatus("Waiting", "waiting")
  setFlow("mock ready; waiting for input")
  logEvent("Connected")
  logEvent("Warmup complete")
  ensureMockChunks()
}

function initialize() {
  document.body.dataset.status = "idle"
  if (mockMode) {
    document.body.classList.add("mock-mode")
    connectButton.textContent = "Start Mock Session"
    logEvent("mock mode ready", { source: "client" })
  } else {
    logEvent("viewer ready", { source: "client" })
  }
  setFlow("waiting")
  renderMetrics()
  attachPointerControls()
  window.requestAnimationFrame(drawMockScene)
  startVideoFrameMonitor()
}

connectButton.addEventListener("click", () => {
  void connectSession()
})
remoteVideo.addEventListener("loadedmetadata", updateMetricsFromVideo)
remoteVideo.addEventListener("playing", () => {
  setVideoVisible(true)
  updateMetricsFromVideo()
})
remoteVideo.addEventListener("emptied", () => {
  setVideoVisible(false)
})
window.addEventListener("keydown", handleKeyDown)
window.addEventListener("keyup", handleKeyUp)
window.addEventListener("blur", releaseAllKeys)

initialize()
