// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const connectButton = document.getElementById("connectButton")
const statusText = document.getElementById("statusText")
const flowText = document.getElementById("flowText")
const eventLog = document.getElementById("eventLog")
const remoteVideo = document.getElementById("remoteVideo")
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
const activeKeys = new Set()
const frameTimes = []
const pendingActions = []
const heartbeatIntervalMs = 2000

let peerConnection = null
let controlChannel = null
let statsTimer = null
let connected = false
let heartbeatTimer = null
let disconnecting = false

const metrics = {
  fps: null,
  targetFps: null,
  latencyMs: null,
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
  eventLog.append(entry)
  while (eventLog.children.length > 28) {
    eventLog.firstElementChild.remove()
  }
  eventLog.scrollTop = eventLog.scrollHeight
}

function setStatus(message, state = message.toLowerCase()) {
  statusText.textContent = message
  document.body.dataset.status = state
}

function setFlow(message) {
  flowText.textContent = message
}

function renderMetrics() {
  const fps = firstFinite(metrics.fps, metrics.targetFps)
  fpsValue.textContent = Number.isFinite(fps) ? String(Math.round(fps)) : "--"
  latencyValue.textContent = formatMs(metrics.latencyMs)
  resolutionValue.textContent = metrics.resolution || "--"
  stepValue.textContent = metrics.step === null ? "--" : String(metrics.step)
  modelValue.textContent = metrics.model || "Omnidreams"
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
  metrics.targetFps = firstFinite(payload.fps, payload.target_fps, metrics.targetFps)
  metrics.latencyMs = firstFinite(
    payload.latency_ms,
    payload.control_latency_ms,
    takeObservedActionLatency(),
    payload.lag_ms,
    payload.gen_ms,
    metrics.latencyMs
  )
  metrics.step = Number.isFinite(Number(payload.chunk_index))
    ? Number(payload.chunk_index)
    : metrics.step
  metrics.model = typeof payload.model === "string" && payload.model ? payload.model : metrics.model
  if (payload.resolution && typeof payload.resolution === "object") {
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
  return `${action.event}:${action.key}`
}

function sendControlAction(action) {
  if (!connected || !controlChannel || controlChannel.readyState !== "open") {
    setFlow("connect session first")
    return false
  }
  controlChannel.send(JSON.stringify({ type: "action", action }))
  pendingActions.push({ sentAt: performance.now(), label: actionLabel(action) })
  while (pendingActions.length > 32) {
    pendingActions.shift()
  }
  setStatus("Generating", "generating")
  setFlow(`sent ${actionLabel(action)}`)
  logEvent(`control ${actionLabel(action)}`, { source: "client" })
  return true
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
  stopHeartbeat()
  if (statsTimer !== null) {
    window.clearInterval(statsTimer)
    statsTimer = null
  }
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
}

function setKeyHeld(key, source, held) {
  if (!allowedKeys.has(key)) {
    return
  }
  let sources = keySources.get(key)
  if (!sources) {
    sources = new Set()
    keySources.set(key, sources)
  }
  const wasHeld = sources.size > 0
  if (held) {
    sources.add(source)
  } else {
    sources.delete(source)
  }
  const isHeld = sources.size > 0
  updateControlHighlights()
  if (wasHeld !== isHeld) {
    sendControlAction({ event: isHeld ? "keydown" : "keyup", key })
  }
}

function handleServerMessage(message) {
  let payload
  try {
    payload = JSON.parse(message)
  } catch {
    logEvent("invalid server payload", { level: "error" })
    return
  }
  if (payload.type === "chunk_done") {
    updateMetricsFromChunk(payload)
    setStatus("Connected", "connected")
    setFlow(`chunk ${payload.chunk_index} done`)
    logEvent(`chunk ${payload.chunk_index} ${payload.num_frames} frames`)
    return
  }
  if (payload.type === "error") {
    setStatus("Error", "error")
    logEvent(payload.message || "server error", { level: "error" })
  }
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

async function connectSession() {
  if (connected) {
    return
  }
  connectButton.disabled = true
  setStatus("Connecting", "connecting")
  setFlow("creating peer")
  disconnecting = false

  peerConnection = new RTCPeerConnection()
  controlChannel = peerConnection.createDataChannel("controls", { ordered: true })
  peerConnection.addTransceiver("video", { direction: "recvonly" })

  controlChannel.addEventListener("open", () => {
    connected = true
    setStatus("Connected", "connected")
    setFlow("press W A S D")
    logEvent("data channel open")
    startHeartbeat()
  })
  controlChannel.addEventListener("message", event => handleServerMessage(event.data))
  controlChannel.addEventListener("close", () => {
    connected = false
    setStatus("Closed", "idle")
    setFlow("closed")
    logEvent("data channel closed", { source: "client" })
    stopHeartbeat()
  })

  peerConnection.addEventListener("track", event => {
    remoteVideo.srcObject = event.streams[0]
    setFlow("video track attached")
    logEvent("video track attached", { source: "client" })
  })
  peerConnection.addEventListener("connectionstatechange", () => {
    const state = peerConnection.connectionState
    logEvent(`connection_state=${state}`, { source: "client" })
    if (state === "connected") {
      connected = true
      setStatus("Connected", "connected")
      setFlow("press W A S D")
      return
    }
    if (state === "connecting") {
      setStatus("Connecting", "connecting")
      return
    }
    if (state === "failed" || state === "disconnected" || state === "closed") {
      connected = false
      connectButton.disabled = false
      stopHeartbeat()
      setStatus(state, state === "failed" ? "error" : "idle")
      void dumpPeerStats(`connection_state=${state}`)
    }
  })
  peerConnection.addEventListener("iceconnectionstatechange", () => {
    const state = peerConnection.iceConnectionState
    logEvent(`ice_connection_state=${state}`, { source: "client" })
    if (state === "failed" || state === "disconnected") {
      void dumpPeerStats(`ice_connection_state=${state}`)
    }
  })
  peerConnection.addEventListener("icegatheringstatechange", () => {
    logEvent(`ice_gathering_state=${peerConnection.iceGatheringState}`, { source: "client" })
  })
  peerConnection.addEventListener("signalingstatechange", () => {
    logEvent(`signaling_state=${peerConnection.signalingState}`, { source: "client" })
  })

  const offer = await peerConnection.createOffer()
  await peerConnection.setLocalDescription(offer)
  await waitForIceGatheringComplete(peerConnection)
  logEvent("local offer ready", { source: "client" })
  const response = await fetch("/api/webrtc/offer", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(peerConnection.localDescription),
  })
  if (!response.ok) {
    const reason = await response.text()
    connectButton.disabled = false
    setStatus("Error", "error")
    setFlow(`offer failed ${response.status}`)
    logEvent(reason || "offer failed", { level: "error" })
    return
  }
  const answer = await response.json()
  await peerConnection.setRemoteDescription(answer)
  logEvent("remote answer applied", { source: "client" })
  setFlow("answer applied")
}

for (const button of controlButtons) {
  const key = button.dataset.controlKey
  button.addEventListener("pointerdown", event => {
    event.preventDefault()
    button.setPointerCapture(event.pointerId)
    setKeyHeld(key, `pointer:${event.pointerId}`, true)
  })
  button.addEventListener("pointerup", event => {
    event.preventDefault()
    setKeyHeld(key, `pointer:${event.pointerId}`, false)
  })
  button.addEventListener("pointercancel", event => {
    setKeyHeld(key, `pointer:${event.pointerId}`, false)
  })
}

window.addEventListener("keydown", event => {
  const key = normalizeKey(event.key)
  if (!allowedKeys.has(key) || event.repeat) {
    return
  }
  event.preventDefault()
  setKeyHeld(key, "keyboard", true)
})

window.addEventListener("keyup", event => {
  const key = normalizeKey(event.key)
  if (!allowedKeys.has(key)) {
    return
  }
  event.preventDefault()
  setKeyHeld(key, "keyboard", false)
})

connectButton.addEventListener("click", () => {
  connectSession().catch(error => {
    connectButton.disabled = false
    setStatus("Error", "error")
    setFlow("connect failed")
    logEvent(error instanceof Error ? error.message : String(error), { level: "error" })
  })
})

remoteVideo.addEventListener("loadedmetadata", updateMetricsFromVideo)

function pollVideoFrames() {
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    remoteVideo.requestVideoFrameCallback(function onFrame(now) {
      recordFrame(now)
      updateMetricsFromVideo()
      remoteVideo.requestVideoFrameCallback(onFrame)
    })
  } else {
    statsTimer = window.setInterval(updateMetricsFromVideo, 500)
  }
}

pollVideoFrames()
renderMetrics()
logEvent("ready")

window.addEventListener("beforeunload", () => {
  disconnectSession()
})
window.addEventListener("pagehide", () => {
  disconnectSession()
})
