// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const connectButton = document.getElementById("connectButton")
const statusText = document.getElementById("statusText")
const flowText = document.getElementById("flowText")
const eventLog = document.getElementById("eventLog")
const logState = document.getElementById("logState")
const remoteVideo = document.getElementById("remoteVideo")
const idleCanvas = document.getElementById("idleCanvas")
const fpsValue = document.getElementById("fpsValue")
const latencyValue = document.getElementById("latencyValue")
const resolutionValue = document.getElementById("resolutionValue")
const stepValue = document.getElementById("stepValue")
const modelValue = document.getElementById("modelValue")
const controlButtons = Array.from(document.querySelectorAll("[data-control-key]"))

const allowedKeys = new Set(["w", "a", "s", "d"])
const keyAliases = new Map([
  ["arrowup", "w"],
  ["arrowleft", "a"],
  ["arrowdown", "s"],
  ["arrowright", "d"],
])
const keySources = new Map()
const heldKeyOrder = new Map()
const activeKeys = new Set()
const frameTimes = []
const pendingActions = []
const maxPendingActions = 32
const heartbeatIntervalMs = 2000

let peerConnection = null
let controlChannel = null
let statsTimer = null
let videoMetricsTimer = null
let heartbeatTimer = null
let inferenceInFlight = false
let connected = false
let disconnecting = false
let heldKeySequence = 0

const metrics = {
  fps: null,
  targetFps: null,
  latencyMs: null,
  rttMs: null,
  resolution: null,
  step: null,
  model: "Omnidreams",
}

function normalizeKey(rawKey) {
  const key = String(rawKey || "").toLowerCase()
  return keyAliases.get(key) || key
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
  const consoleMessage = `[Omnidreams WebRTC][${source}] ${message}`
  if (level === "error") {
    console.error(consoleMessage)
  } else {
    console.info(consoleMessage)
  }

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
  const latency = firstFinite(metrics.latencyMs, metrics.rttMs)
  fpsValue.textContent = Number.isFinite(fps) ? String(Math.round(fps)) : "--"
  latencyValue.textContent = formatMs(latency)
  resolutionValue.textContent = metrics.resolution || "--"
  stepValue.textContent = metrics.step === null ? "--" : String(metrics.step)
  modelValue.textContent = metrics.model || "Omnidreams"
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

function resizeIdleCanvas(ctx) {
  const rect = idleCanvas.getBoundingClientRect()
  const dpr = Math.min(window.devicePixelRatio || 1, 2)
  const width = Math.max(1, Math.floor(rect.width * dpr))
  const height = Math.max(1, Math.floor(rect.height * dpr))
  if (idleCanvas.width !== width || idleCanvas.height !== height) {
    idleCanvas.width = width
    idleCanvas.height = height
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  return { width: rect.width, height: rect.height }
}

function drawRouteRibbon(ctx, width, height, t) {
  const xBase = width * 0.74
  const yBase = height * 0.28
  ctx.save()
  ctx.globalAlpha = 0.62
  ctx.lineWidth = 2
  ctx.strokeStyle = "rgba(99, 216, 255, 0.72)"
  ctx.setLineDash([10, 14])
  ctx.lineDashOffset = -t * 24
  ctx.beginPath()
  ctx.moveTo(xBase - 92, yBase + 132)
  ctx.bezierCurveTo(xBase - 36, yBase + 36, xBase + 42, yBase + 76, xBase + 86, yBase - 16)
  ctx.bezierCurveTo(xBase + 116, yBase - 76, xBase + 8, yBase - 92, xBase - 20, yBase - 34)
  ctx.stroke()

  ctx.setLineDash([])
  for (let i = 0; i < 8; i += 1) {
    const phase = (i / 8 + t * 0.08) % 1
    const angle = phase * Math.PI * 2
    const x = xBase + Math.cos(angle) * 84
    const y = yBase + Math.sin(angle * 1.7) * 72
    ctx.fillStyle = i % 2 === 0 ? "rgba(142, 240, 28, 0.72)" : "rgba(99, 216, 255, 0.62)"
    ctx.beginPath()
    ctx.arc(x, y, 3.5, 0, Math.PI * 2)
    ctx.fill()
  }
  ctx.restore()
}

function drawIdleScene(now) {
  const ctx = idleCanvas.getContext("2d")
  if (!ctx) {
    return
  }

  const { width, height } = resizeIdleCanvas(ctx)
  const t = now * 0.001
  const horizon = height * 0.46

  const sky = ctx.createLinearGradient(0, 0, 0, height)
  sky.addColorStop(0, "#314553")
  sky.addColorStop(0.42, "#76919c")
  sky.addColorStop(0.66, "#152024")
  sky.addColorStop(1, "#060707")
  ctx.fillStyle = sky
  ctx.fillRect(0, 0, width, height)

  const sunGlow = ctx.createRadialGradient(width * 0.22, height * 0.22, 8, width * 0.22, height * 0.22, width * 0.42)
  sunGlow.addColorStop(0, "rgba(255, 204, 112, 0.62)")
  sunGlow.addColorStop(0.36, "rgba(255, 204, 112, 0.20)")
  sunGlow.addColorStop(1, "rgba(255, 204, 112, 0)")
  ctx.fillStyle = sunGlow
  ctx.fillRect(0, 0, width, height)

  ctx.fillStyle = "rgba(24, 39, 42, 0.82)"
  for (let i = 0; i < 12; i += 1) {
    const x = width * (0.02 + i * 0.075)
    const buildingWidth = width * (0.035 + (i % 3) * 0.012)
    const buildingHeight = height * (0.11 + ((i * 7) % 5) * 0.018)
    ctx.fillRect(x, horizon - buildingHeight, buildingWidth, buildingHeight)
  }

  const ground = ctx.createLinearGradient(0, horizon, 0, height)
  ground.addColorStop(0, "#273331")
  ground.addColorStop(1, "#0a0c0c")
  ctx.fillStyle = ground
  ctx.fillRect(0, horizon, width, height - horizon)

  const road = ctx.createLinearGradient(width * 0.5, horizon, width * 0.5, height)
  road.addColorStop(0, "#424c4f")
  road.addColorStop(1, "#121516")
  ctx.fillStyle = road
  ctx.beginPath()
  ctx.moveTo(width * 0.42, horizon + 8)
  ctx.lineTo(width * 0.58, horizon + 8)
  ctx.lineTo(width * 0.80, height)
  ctx.lineTo(width * 0.20, height)
  ctx.closePath()
  ctx.fill()

  ctx.strokeStyle = "rgba(255, 255, 255, 0.42)"
  ctx.lineWidth = 2
  ctx.beginPath()
  ctx.moveTo(width * 0.42, horizon + 8)
  ctx.lineTo(width * 0.20, height)
  ctx.moveTo(width * 0.58, horizon + 8)
  ctx.lineTo(width * 0.80, height)
  ctx.stroke()

  const dashOffset = (t * 92) % 58
  for (let i = -2; i < 14; i += 1) {
    const y = horizon + 20 + i * 58 + dashOffset
    const scale = Math.max(0, Math.min(1, (y - horizon) / (height - horizon)))
    const dashHeight = 18 + scale * 38
    const wobble = Math.sin(t * 0.8 + scale * 3.2) * width * 0.012
    ctx.strokeStyle = "rgba(255, 222, 114, 0.74)"
    ctx.lineWidth = 2 + scale * 3
    ctx.beginPath()
    ctx.moveTo(width * 0.50 + wobble, y)
    ctx.lineTo(width * 0.50 + wobble * 1.3, y + dashHeight)
    ctx.stroke()
  }

  ctx.save()
  ctx.translate(width * 0.5, height * 0.78 + Math.sin(t * 2.1) * 4)
  ctx.fillStyle = "rgba(142, 240, 28, 0.74)"
  ctx.beginPath()
  ctx.moveTo(0, -34)
  ctx.lineTo(22, 24)
  ctx.lineTo(0, 12)
  ctx.lineTo(-22, 24)
  ctx.closePath()
  ctx.fill()
  ctx.strokeStyle = "rgba(255, 255, 255, 0.52)"
  ctx.lineWidth = 2
  ctx.stroke()
  ctx.restore()

  drawRouteRibbon(ctx, width, height, t)

  ctx.fillStyle = `rgba(255, 255, 255, ${0.06 + Math.sin(t * 1.4) * 0.018})`
  ctx.fillRect(0, 0, width, height)

  if (!document.body.classList.contains("has-video")) {
    recordFrame(now)
  }
  window.requestAnimationFrame(drawIdleScene)
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

function sendControlAction(action) {
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
  } catch {
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
    ]
    if (Number.isFinite(Number(payload.enqueued_frames))) {
      parts.push(`enqueued=${payload.enqueued_frames}`)
    }
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

function resetPeerHandles(pc = peerConnection, channel = controlChannel) {
  if (peerConnection === pc) {
    peerConnection = null
  }
  if (controlChannel === channel) {
    controlChannel = null
  }
}

async function dumpPeerStats(reason) {
  if (!peerConnection) {
    return
  }
  try {
    const stats = await peerConnection.getStats()
    const reports = new Map()
    for (const report of stats.values()) {
      reports.set(report.id, report)
    }
    console.group(`[Omnidreams WebRTC] peer stats: ${reason}`)
    for (const report of stats.values()) {
      if (report.type !== "candidate-pair") {
        continue
      }
      const local = reports.get(report.localCandidateId)
      const remote = reports.get(report.remoteCandidateId)
      console.info({
        id: report.id,
        state: report.state,
        nominated: report.nominated,
        writable: report.writable,
        local: local
          ? `${local.candidateType} ${local.protocol} ${local.address || local.ip}:${local.port}`
          : report.localCandidateId,
        remote: remote
          ? `${remote.candidateType} ${remote.protocol} ${remote.address || remote.ip}:${remote.port}`
          : report.remoteCandidateId,
      })
    }
    console.groupEnd()
  } catch (error) {
    console.warn("[Omnidreams WebRTC] getStats failed", error)
  }
}

function sendHeartbeat() {
  if (!controlChannel || controlChannel.readyState !== "open") {
    return
  }
  try {
    controlChannel.send(JSON.stringify({ type: "heartbeat", t: Date.now() }))
  } catch (error) {
    logEvent(`heartbeat failed: ${error.message}`, { source: "client" })
  }
}

function startHeartbeat() {
  if (heartbeatTimer !== null) {
    return
  }
  sendHeartbeat()
  heartbeatTimer = window.setInterval(sendHeartbeat, heartbeatIntervalMs)
}

function stopHeartbeat() {
  if (heartbeatTimer !== null) {
    window.clearInterval(heartbeatTimer)
    heartbeatTimer = null
  }
}

function disconnectSession({ notify = true } = {}) {
  if (disconnecting) {
    return
  }
  disconnecting = true
  releaseAllKeys()
  stopHeartbeat()
  stopStatsPolling()
  connected = false
  connectButton.disabled = false
  if (notify && controlChannel && controlChannel.readyState === "open") {
    try {
      controlChannel.send(JSON.stringify({ type: "disconnect" }))
    } catch {
      // The browser may already be tearing the page down.
    }
  }
  if (controlChannel && controlChannel.readyState !== "closed") {
    controlChannel.close()
  }
  if (peerConnection) {
    peerConnection.close()
  }
  resetPeerHandles()
}

async function connectSession() {
  if (connected || peerConnection) {
    return
  }

  connectButton.disabled = true
  setStatus("Connecting", "connecting")
  setFlow("creating peer connection")
  logEvent("connecting to server...", { source: "client" })
  disconnecting = false

  try {
    const pc = new RTCPeerConnection()
    const channel = pc.createDataChannel("controls")
    peerConnection = pc
    controlChannel = channel
    pc.addTransceiver("video", { direction: "recvonly" })

    channel.onopen = () => {
      connected = true
      setStatus("Waiting", "waiting")
      setFlow("connected; waiting for input")
      logEvent("control data channel open")
      startHeartbeat()
    }
    channel.onclose = () => {
      connected = false
      if (document.body.dataset.status !== "error") {
        setStatus("Closed", "idle")
      }
      setFlow("channel closed")
      logEvent("control data channel closed", { source: "client" })
      stopHeartbeat()
      stopStatsPolling()
      if (!disconnecting && pc.connectionState !== "closed") {
        pc.close()
      }
      resetPeerHandles(pc, channel)
    }
    channel.onmessage = (event) => {
      handleControlMessage(event.data)
    }

    pc.ontrack = (event) => {
      const [stream] = event.streams
      if (stream) {
        remoteVideo.srcObject = stream
        updateMetricsFromVideo()
      }
      setFlow("video track attached")
      logEvent("video track attached", { source: "client" })
    }

    pc.onconnectionstatechange = () => {
      const state = pc.connectionState
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
        connectButton.disabled = false
        stopHeartbeat()
        stopStatsPolling()
        setStatus(state === "failed" ? "Error" : "Idle", state === "failed" ? "error" : "idle")
        void dumpPeerStats(`connection_state=${state}`)
        if (!disconnecting && pc.connectionState !== "closed") {
          pc.close()
        }
        resetPeerHandles(pc, channel)
      }
    }
    pc.oniceconnectionstatechange = () => {
      const state = pc.iceConnectionState
      logEvent(`ice_connection_state=${state}`, { source: "client" })
      if (state === "failed" || state === "disconnected") {
        void dumpPeerStats(`ice_connection_state=${state}`)
      }
    }
    pc.onicegatheringstatechange = () => {
      logEvent(`ice_gathering_state=${pc.iceGatheringState}`, { source: "client" })
    }
    pc.onsignalingstatechange = () => {
      logEvent(`signaling_state=${pc.signalingState}`, { source: "client" })
    }

    const offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    await waitForIceGatheringComplete(pc)
    logEvent("local offer ready", { source: "client" })

    const response = await fetch("/api/webrtc/offer", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(pc.localDescription),
    })
    if (!response.ok) {
      const text = await response.text()
      throw new Error(`offer failed (${response.status}): ${text}`)
    }
    const answer = await response.json()
    await pc.setRemoteDescription(answer)
    logEvent("remote answer applied", { source: "client" })
    setFlow("answer applied")
  } catch (error) {
    stopHeartbeat()
    stopStatsPolling()
    if (peerConnection) {
      peerConnection.close()
    }
    resetPeerHandles()
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

function startVideoFrameMonitor() {
  if (typeof remoteVideo.requestVideoFrameCallback !== "function") {
    if (videoMetricsTimer === null) {
      videoMetricsTimer = window.setInterval(updateMetricsFromVideo, 500)
    }
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

function initialize() {
  document.body.dataset.status = "idle"
  logEvent("viewer ready", { source: "client" })
  setFlow("waiting")
  renderMetrics()
  attachPointerControls()
  window.requestAnimationFrame(drawIdleScene)
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
window.addEventListener("pagehide", () => {
  disconnectSession()
})
window.addEventListener("beforeunload", () => {
  disconnectSession()
})

initialize()
