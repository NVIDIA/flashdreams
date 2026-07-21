// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const connectButton = document.getElementById("connectButton")
const statusText = document.getElementById("statusText")
const flowText = document.getElementById("flowText")
const eventLog = document.getElementById("eventLog")
const logState = document.getElementById("logState")
const remoteVideo = document.getElementById("remoteVideo")
const idleCanvas = document.getElementById("idleCanvas")
const browserFpsValue = document.getElementById("browserFpsValue")
const presentedFpsValue = document.getElementById("presentedFpsValue")
const serverFpsValue = document.getElementById("serverFpsValue")
const dropValue = document.getElementById("dropValue")
const jitterValue = document.getElementById("jitterValue")
const decodeValue = document.getElementById("decodeValue")
const bitrateValue = document.getElementById("bitrateValue")
const latencyValue = document.getElementById("latencyValue")
const resolutionValue = document.getElementById("resolutionValue")
const stepValue = document.getElementById("stepValue")
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
const videoFrameCallbackTimes = []
const presentedFrameSamples = []
const pendingActions = []
const maxPendingActions = 32
const heartbeatIntervalMs = 2000
const browserProfileLogIntervalMs = 5000
const metricsRenderIntervalMs = 500

let peerConnection = null
let controlChannel = null
let statsTimer = null
let videoMetricsTimer = null
let heartbeatTimer = null
let metricsRenderTimer = null
let idleAnimationFrame = null
let previousInboundVideoStats = null
let previousBrowserProfileLogAt = 0
let previousMetricsRenderAt = 0
let inferenceInFlight = false
let connected = false
let disconnecting = false
let heldKeySequence = 0

const metrics = {
  browserFps: null,
  browserRtpFps: null,
  receivedFps: null,
  decodedFps: null,
  presentedFps: null,
  videoCallbackFps: null,
  serverFps: null,
  targetFps: null,
  latencyMs: null,
  rttMs: null,
  droppedFrames: null,
  droppedFps: null,
  dropPercent: null,
  jitterMs: null,
  jitterBufferMs: null,
  decodeMs: null,
  processingMs: null,
  bitrateMbps: null,
  packetsLost: null,
  packetsLostPerSec: null,
  freezeCount: null,
  nackCount: null,
  decoder: null,
  presentationDelayMs: null,
  serverQueueDepth: null,
  serverLagMs: null,
  serverDeliveryMs: null,
  serverEncodeMs: null,
  serverTrackDroppedPackets: null,
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
    if (value === null || value === undefined || value === "") {
      continue
    }
    const number = Number(value)
    if (Number.isFinite(number)) {
      return number
    }
  }
  return null
}

function toFiniteNumber(value) {
  return firstFinite(value)
}

function formatFps(value) {
  const number = toFiniteNumber(value)
  return number === null ? "--" : number.toFixed(1)
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

function formatMbps(value) {
  const number = toFiniteNumber(value)
  if (number === null) {
    return "--"
  }
  if (number >= 10) {
    return `${number.toFixed(1)} Mb/s`
  }
  return `${number.toFixed(2)} Mb/s`
}

function formatDropValue() {
  const total = toFiniteNumber(metrics.droppedFrames)
  if (total === null) {
    return "--"
  }
  const droppedFps = toFiniteNumber(metrics.droppedFps)
  if (droppedFps !== null && droppedFps > 0.05) {
    return `${Math.round(total)} (${droppedFps.toFixed(1)}/s)`
  }
  return String(Math.round(total))
}

function setTextIfChanged(element, text) {
  if (element.textContent !== text) {
    element.textContent = text
  }
}

function logEvent(
  message,
  { source = "server", level = "info", dom = true, consoleOutput = true } = {}
) {
  if (consoleOutput) {
    const consoleMessage = `[Omnidreams WebRTC][${source}] ${message}`
    if (level === "error") {
      console.error(consoleMessage)
    } else {
      console.info(consoleMessage)
    }
  }
  if (!dom) {
    return
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
  setTextIfChanged(statusText, message)
  if (document.body.dataset.status !== state) {
    document.body.dataset.status = state
  }
  setTextIfChanged(logState, state === "idle" ? "Waiting" : message)
}

function setFlow(message) {
  setTextIfChanged(flowText, message)
}

function setVideoVisible(visible) {
  document.body.classList.toggle("has-video", visible)
  if (visible) {
    stopIdleAnimation()
  } else {
    startIdleAnimation()
  }
}

function renderMetrics({ force = false } = {}) {
  const now = performance.now()
  if (!force && previousMetricsRenderAt > 0) {
    const elapsed = now - previousMetricsRenderAt
    if (elapsed < metricsRenderIntervalMs) {
      if (metricsRenderTimer === null) {
        metricsRenderTimer = window.setTimeout(() => {
          metricsRenderTimer = null
          renderMetrics({ force: true })
        }, metricsRenderIntervalMs - elapsed)
      }
      return
    }
  }

  previousMetricsRenderAt = now
  const browserFps = firstFinite(
    metrics.browserRtpFps,
    metrics.decodedFps,
    metrics.receivedFps,
    metrics.presentedFps
  )
  const presentedFps = firstFinite(metrics.presentedFps)
  const serverFps = firstFinite(metrics.serverFps, metrics.targetFps)
  const latency = firstFinite(metrics.latencyMs, metrics.rttMs)
  metrics.browserFps = browserFps
  setTextIfChanged(browserFpsValue, formatFps(browserFps))
  setTextIfChanged(presentedFpsValue, formatFps(presentedFps))
  setTextIfChanged(serverFpsValue, formatFps(serverFps))
  setTextIfChanged(dropValue, formatDropValue())
  setTextIfChanged(
    jitterValue,
    formatMs(firstFinite(metrics.jitterBufferMs, metrics.jitterMs))
  )
  setTextIfChanged(decodeValue, formatMs(firstFinite(metrics.decodeMs, metrics.processingMs)))
  setTextIfChanged(bitrateValue, formatMbps(metrics.bitrateMbps))
  setTextIfChanged(latencyValue, formatMs(latency))
  setTextIfChanged(resolutionValue, metrics.resolution || "--")
  setTextIfChanged(stepValue, metrics.step === null ? "--" : String(metrics.step))
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
  metrics.serverFps = firstFinite(payload.chunk_fps, metrics.serverFps)
  metrics.serverQueueDepth = firstFinite(payload.queue_depth, metrics.serverQueueDepth)
  metrics.serverLagMs = firstFinite(payload.lag_ms, metrics.serverLagMs)
  metrics.serverDeliveryMs = firstFinite(payload.delivery_ms, payload.enqueue_ms, metrics.serverDeliveryMs)
  metrics.serverEncodeMs = firstFinite(payload.delivery_encode_ms, metrics.serverEncodeMs)
  metrics.serverTrackDroppedPackets = firstFinite(
    payload.track_dropped_packets,
    metrics.serverTrackDroppedPackets
  )
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
    const resolution = `${remoteVideo.videoWidth}x${remoteVideo.videoHeight}`
    if (metrics.resolution !== resolution) {
      metrics.resolution = resolution
      renderMetrics({ force: true })
    }
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
  idleAnimationFrame = null
  if (document.body.classList.contains("has-video")) {
    return
  }

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

  idleAnimationFrame = window.requestAnimationFrame(drawIdleScene)
}

function startIdleAnimation() {
  if (
    idleAnimationFrame !== null ||
    document.body.classList.contains("has-video")
  ) {
    return
  }
  idleAnimationFrame = window.requestAnimationFrame(drawIdleScene)
}

function stopIdleAnimation() {
  if (idleAnimationFrame !== null) {
    window.cancelAnimationFrame(idleAnimationFrame)
    idleAnimationFrame = null
  }
}

function recordPresentedFrame(timestamp, metadata = {}) {
  const now = Number.isFinite(timestamp) ? timestamp : performance.now()
  videoFrameCallbackTimes.push(now)
  while (
    videoFrameCallbackTimes.length > 0 &&
    now - videoFrameCallbackTimes[0] > 1200
  ) {
    videoFrameCallbackTimes.shift()
  }
  if (videoFrameCallbackTimes.length >= 2) {
    const elapsed =
      videoFrameCallbackTimes[videoFrameCallbackTimes.length - 1] -
      videoFrameCallbackTimes[0]
    metrics.videoCallbackFps =
      elapsed > 0
        ? ((videoFrameCallbackTimes.length - 1) * 1000) / elapsed
        : metrics.videoCallbackFps
  }

  const presentedFrames = toFiniteNumber(metadata.presentedFrames)
  if (presentedFrames !== null) {
    presentedFrameSamples.push({ count: presentedFrames, timestamp: now })
    while (
      presentedFrameSamples.length > 0 &&
      now - presentedFrameSamples[0].timestamp > 1200
    ) {
      presentedFrameSamples.shift()
    }
    if (presentedFrameSamples.length >= 2) {
      const firstSample = presentedFrameSamples[0]
      const lastSample = presentedFrameSamples[presentedFrameSamples.length - 1]
      const elapsed = lastSample.timestamp - firstSample.timestamp
      const frameDelta = lastSample.count - firstSample.count
      if (elapsed > 0 && frameDelta >= 0) {
        metrics.presentedFps = (frameDelta * 1000) / elapsed
      }
    }
  } else {
    metrics.presentedFps = metrics.videoCallbackFps
  }

  const presentationTime = toFiniteNumber(metadata.presentationTime)
  if (presentationTime !== null) {
    metrics.presentationDelayMs = Math.max(0, now - presentationTime)
  }

  const processingDuration = toFiniteNumber(metadata.processingDuration)
  if (processingDuration !== null) {
    metrics.processingMs = processingDuration * 1000
  }

  renderMetrics()
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

function sendControlAction(action, { log = true } = {}) {
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
  if (log) {
    logEvent(`control ${actionLabel(action)}`, {
      source: "client",
      dom: !document.body.classList.contains("has-video"),
    })
  }
  return true
}

function enqueueAction(action, options = {}) {
  const sent = sendControlAction(action, options)
  if (!sent) {
    setFlow(connected ? `not_sent ${actionLabel(action)}` : "connect session first")
  }
}

function enqueueHeldKeyRepeats() {
  const heldKeys = Array.from(activeKeys).sort((a, b) => {
    return (heldKeyOrder.get(a) || 0) - (heldKeyOrder.get(b) || 0)
  })
  for (const key of heldKeys) {
    enqueueAction({ event: "keydown", key }, { log: false })
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
    const runtimeCallMs = firstFinite(payload.runtime_call_ms)
    const renderMs = firstFinite(payload.wrapper_render_ms)
    const modelMs = firstFinite(payload.wrapper_generate_ms)
    const syncMs = firstFinite(payload.runtime_sync_ms)
    const deliveryMs = firstFinite(payload.delivery_ms, payload.enqueue_ms)
    const encodeMs = firstFinite(payload.delivery_encode_ms)
    const chunkFps = firstFinite(payload.chunk_fps)
    const lagMs = firstFinite(payload.lag_ms)
    const queueDepth = firstFinite(payload.queue_depth)
    const trackDroppedPackets = firstFinite(payload.track_dropped_packets)
    const parts = [
      `chunk_done index=${payload.chunk_index}`,
      `frames=${payload.num_frames}`,
    ]
    if (typeof payload.encoder_backend === "string" && payload.encoder_backend) {
      parts.push(`encoder=${payload.encoder_backend}`)
    }
    if (Number.isFinite(Number(payload.enqueued_frames))) {
      parts.push(`enqueued=${payload.enqueued_frames}`)
    }
    if (chunkFps !== null) {
      parts.push(`chunk_fps=${Math.round(chunkFps)}`)
    }
    if (genMs !== null) {
      parts.push(`gen=${Math.round(genMs)}ms`)
    }
    if (runtimeCallMs !== null) {
      parts.push(`runtime=${Math.round(runtimeCallMs)}ms`)
    }
    if (renderMs !== null) {
      parts.push(`render=${Math.round(renderMs)}ms`)
    }
    if (modelMs !== null) {
      parts.push(`model=${Math.round(modelMs)}ms`)
    }
    if (syncMs !== null) {
      parts.push(`sync=${Math.round(syncMs)}ms`)
    }
    if (deliveryMs !== null) {
      parts.push(`delivery=${Math.round(deliveryMs)}ms`)
    }
    if (encodeMs !== null) {
      parts.push(`encode=${Math.round(encodeMs)}ms`)
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
    if (trackDroppedPackets !== null && trackDroppedPackets > 0) {
      parts.push(`track_dropped=${trackDroppedPackets}`)
    }
    const chunkIndex = Number(payload.chunk_index)
    const hasVideo = document.body.classList.contains("has-video")
    const importantChunk =
      !Number.isFinite(chunkIndex) ||
      chunkIndex <= 2 ||
      chunkIndex % 10 === 0 ||
      (trackDroppedPackets !== null && trackDroppedPackets > 0)
    const showInPanel = !hasVideo && importantChunk
    const showInConsole = importantChunk
    logEvent(parts.join(", "), {
      dom: showInPanel,
      consoleOutput: showInConsole,
    })
    setStatus(activeKeys.size > 0 ? "Generating" : "Waiting", activeKeys.size > 0 ? "generating" : "waiting")
    if (showInPanel || !hasVideo) {
      setFlow(`chunk ${payload.chunk_index} complete`)
    }
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

function setLowLatencyReceiverHint(receiver) {
  if (!receiver || !("playoutDelayHint" in receiver)) {
    return
  }
  try {
    receiver.playoutDelayHint = 0
  } catch (error) {
    logEvent(`could not set low-latency playout hint: ${error.message}`, {
      source: "client",
    })
  }
}

function applyLowLatencyReceiverHints(pc) {
  for (const receiver of pc.getReceivers()) {
    if (receiver.track && receiver.track.kind === "video") {
      setLowLatencyReceiverHint(receiver)
    }
  }
}

function preferH264VideoCodec(transceiver) {
  if (
    !transceiver ||
    typeof transceiver.setCodecPreferences !== "function" ||
    !window.RTCRtpReceiver ||
    typeof RTCRtpReceiver.getCapabilities !== "function"
  ) {
    return
  }
  const capabilities = RTCRtpReceiver.getCapabilities("video")
  const codecs = capabilities && Array.isArray(capabilities.codecs)
    ? capabilities.codecs
    : []
  const h264Codecs = codecs.filter((codec) => {
    return String(codec.mimeType || "").toLowerCase() === "video/h264"
  })
  if (h264Codecs.length === 0) {
    return
  }
  try {
    transceiver.setCodecPreferences(h264Codecs)
    logEvent("preferred H.264 receive codec for NVENC compatibility", {
      source: "client",
    })
  } catch (error) {
    logEvent(`could not prefer H.264 codec: ${error.message}`, {
      source: "client",
    })
  }
}

function snapshotInboundVideoStats(report) {
  const timestamp = toFiniteNumber(report.timestamp)
  return {
    timestamp: timestamp === null ? performance.now() : timestamp,
    framesPerSecond: toFiniteNumber(report.framesPerSecond),
    framesReceived: toFiniteNumber(report.framesReceived),
    framesDecoded: toFiniteNumber(report.framesDecoded),
    framesDropped: toFiniteNumber(report.framesDropped),
    jitter: toFiniteNumber(report.jitter),
    jitterBufferDelay: toFiniteNumber(report.jitterBufferDelay),
    jitterBufferEmittedCount: toFiniteNumber(report.jitterBufferEmittedCount),
    totalDecodeTime: toFiniteNumber(report.totalDecodeTime),
    totalProcessingDelay: toFiniteNumber(report.totalProcessingDelay),
    bytesReceived: toFiniteNumber(report.bytesReceived),
    packetsLost: toFiniteNumber(report.packetsLost),
    nackCount: toFiniteNumber(report.nackCount),
    freezeCount: toFiniteNumber(report.freezeCount),
    decoderImplementation:
      typeof report.decoderImplementation === "string" ? report.decoderImplementation : null,
  }
}

function deltaValue(current, previous, key) {
  if (!previous) {
    return null
  }
  const currentValue = toFiniteNumber(current[key])
  const previousValue = toFiniteNumber(previous[key])
  if (currentValue === null || previousValue === null) {
    return null
  }
  const delta = currentValue - previousValue
  return delta >= 0 ? delta : null
}

function deltaRate(current, previous, key, elapsedSeconds) {
  const delta = deltaValue(current, previous, key)
  if (delta === null || elapsedSeconds <= 0) {
    return null
  }
  return delta / elapsedSeconds
}

function deltaAverageMs(current, previous, totalKey, countKey) {
  const totalDelta = deltaValue(current, previous, totalKey)
  const countDelta = deltaValue(current, previous, countKey)
  if (totalDelta === null || countDelta === null || countDelta <= 0) {
    return null
  }
  return (totalDelta * 1000) / countDelta
}

function updateInboundVideoMetrics(report) {
  const current = snapshotInboundVideoStats(report)
  const previous = previousInboundVideoStats
  const elapsedMs = previous ? current.timestamp - previous.timestamp : null
  const elapsedSeconds = elapsedMs !== null && elapsedMs > 0 ? elapsedMs / 1000 : null

  metrics.browserRtpFps = firstFinite(current.framesPerSecond, metrics.browserRtpFps)
  metrics.jitterMs = current.jitter === null ? metrics.jitterMs : current.jitter * 1000
  metrics.droppedFrames = firstFinite(current.framesDropped, metrics.droppedFrames)
  metrics.packetsLost = firstFinite(current.packetsLost, metrics.packetsLost)
  metrics.freezeCount = firstFinite(current.freezeCount, metrics.freezeCount)
  metrics.nackCount = firstFinite(current.nackCount, metrics.nackCount)
  metrics.decoder = current.decoderImplementation || metrics.decoder

  if (elapsedSeconds !== null) {
    const receivedFps = deltaRate(current, previous, "framesReceived", elapsedSeconds)
    const decodedFps = deltaRate(current, previous, "framesDecoded", elapsedSeconds)
    const droppedFps = deltaRate(current, previous, "framesDropped", elapsedSeconds)
    const bytesPerSecond = deltaRate(current, previous, "bytesReceived", elapsedSeconds)
    const packetsLostPerSecond = deltaRate(current, previous, "packetsLost", elapsedSeconds)
    const jitterBufferMs = deltaAverageMs(
      current,
      previous,
      "jitterBufferDelay",
      "jitterBufferEmittedCount"
    )
    const decodeMs = deltaAverageMs(current, previous, "totalDecodeTime", "framesDecoded")
    const processingMs = deltaAverageMs(
      current,
      previous,
      "totalProcessingDelay",
      "framesDecoded"
    )

    metrics.receivedFps = firstFinite(receivedFps, metrics.receivedFps)
    metrics.decodedFps = firstFinite(decodedFps, metrics.decodedFps)
    metrics.droppedFps = firstFinite(droppedFps, metrics.droppedFps)
    metrics.packetsLostPerSec = firstFinite(packetsLostPerSecond, metrics.packetsLostPerSec)
    metrics.jitterBufferMs = firstFinite(jitterBufferMs, metrics.jitterBufferMs)
    metrics.decodeMs = firstFinite(decodeMs, metrics.decodeMs)
    metrics.processingMs = firstFinite(processingMs, metrics.processingMs)
    if (bytesPerSecond !== null) {
      metrics.bitrateMbps = (bytesPerSecond * 8) / 1_000_000
    }

    const dropped = deltaValue(current, previous, "framesDropped")
    const decoded = deltaValue(current, previous, "framesDecoded")
    if (dropped !== null && decoded !== null && dropped + decoded > 0) {
      metrics.dropPercent = (dropped * 100) / (dropped + decoded)
    }
  }

  previousInboundVideoStats = current
  renderMetrics()
}

function diagnoseBrowserProfile() {
  const serverFps = firstFinite(metrics.serverFps, metrics.targetFps)
  const mediaFps = firstFinite(
    metrics.browserFps,
    metrics.browserRtpFps,
    metrics.decodedFps,
    metrics.receivedFps
  )
  const presentedFps = firstFinite(metrics.presentedFps)
  if (serverFps === null || mediaFps === null) {
    return "collecting"
  }
  const frameBudgetMs = 1000 / Math.max(firstFinite(metrics.targetFps, 30), 1)
  if (firstFinite(metrics.droppedFps, 0) > 1 || firstFinite(metrics.dropPercent, 0) > 5) {
    return "browser_dropping_frames"
  }
  if (
    firstFinite(metrics.jitterBufferMs, 0) > frameBudgetMs * 2 ||
    firstFinite(metrics.jitterMs, 0) > 20
  ) {
    return "network_or_jitter_buffer"
  }
  if (firstFinite(metrics.decodeMs, 0) > frameBudgetMs) {
    return "decode_bound"
  }
  if (presentedFps !== null && mediaFps - presentedFps > 3) {
    return "presentation_throttled"
  }
  if (serverFps - mediaFps <= 3) {
    return "server_browser_matched"
  }
  if (
    metrics.receivedFps !== null &&
    metrics.decodedFps !== null &&
    metrics.receivedFps - metrics.decodedFps > 3
  ) {
    return "decode_backlog"
  }
  if (
    firstFinite(metrics.serverQueueDepth, 0) > 2 ||
    firstFinite(metrics.serverLagMs, 0) > frameBudgetMs * 2
  ) {
    return "server_queue_lag"
  }
  return "receiver_or_display_pacing"
}

function maybeLogBrowserProfile(now = performance.now()) {
  if (!connected || now - previousBrowserProfileLogAt < browserProfileLogIntervalMs) {
    return
  }
  previousBrowserProfileLogAt = now

  const browserFps = firstFinite(metrics.browserFps, metrics.browserRtpFps)
  const serverFps = firstFinite(metrics.serverFps, metrics.targetFps)
  const parts = [`diagnosis=${diagnoseBrowserProfile()}`]
  if (browserFps !== null) {
    parts.push(`browser_fps=${browserFps.toFixed(1)}`)
  }
  if (serverFps !== null) {
    parts.push(`server_fps=${serverFps.toFixed(1)}`)
  }
  if (metrics.receivedFps !== null) {
    parts.push(`recv_fps=${metrics.receivedFps.toFixed(1)}`)
  }
  if (metrics.decodedFps !== null) {
    parts.push(`decoded_fps=${metrics.decodedFps.toFixed(1)}`)
  }
  if (metrics.presentedFps !== null) {
    parts.push(`presented_fps=${metrics.presentedFps.toFixed(1)}`)
  }
  if (metrics.videoCallbackFps !== null) {
    parts.push(`callback_fps=${metrics.videoCallbackFps.toFixed(1)}`)
  }
  if (metrics.droppedFps !== null) {
    parts.push(`dropped_fps=${metrics.droppedFps.toFixed(1)}`)
  }
  if (metrics.jitterBufferMs !== null) {
    parts.push(`jitter_buffer=${Math.round(metrics.jitterBufferMs)}ms`)
  }
  if (metrics.decodeMs !== null) {
    parts.push(`decode=${metrics.decodeMs.toFixed(1)}ms`)
  }
  if (metrics.bitrateMbps !== null) {
    parts.push(`bitrate=${metrics.bitrateMbps.toFixed(2)}Mb/s`)
  }
  if (metrics.serverQueueDepth !== null) {
    parts.push(`server_queue=${metrics.serverQueueDepth}`)
  }
  if (metrics.serverLagMs !== null) {
    parts.push(`server_lag=${Math.round(metrics.serverLagMs)}ms`)
  }
  if (
    metrics.serverTrackDroppedPackets !== null &&
    metrics.serverTrackDroppedPackets > 0
  ) {
    parts.push(`server_track_dropped=${metrics.serverTrackDroppedPackets}`)
  }
  if (metrics.decoder) {
    parts.push(`decoder=${metrics.decoder}`)
  }
  logEvent(`browser_profile ${parts.join(", ")}`, {
    source: "client",
    dom: !document.body.classList.contains("has-video"),
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
        (report.kind === "video" || report.mediaType === "video")
      ) {
        updateInboundVideoMetrics(report)
      }
    }
    renderMetrics()
    maybeLogBrowserProfile()
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

function resetBrowserProfileMetrics() {
  videoFrameCallbackTimes.length = 0
  presentedFrameSamples.length = 0
  previousInboundVideoStats = null
  previousBrowserProfileLogAt = 0
  previousMetricsRenderAt = 0
  if (metricsRenderTimer !== null) {
    window.clearTimeout(metricsRenderTimer)
    metricsRenderTimer = null
  }
  Object.assign(metrics, {
    browserFps: null,
    browserRtpFps: null,
    receivedFps: null,
    decodedFps: null,
    presentedFps: null,
    videoCallbackFps: null,
    serverFps: null,
    targetFps: null,
    latencyMs: null,
    rttMs: null,
    droppedFrames: null,
    droppedFps: null,
    dropPercent: null,
    jitterMs: null,
    jitterBufferMs: null,
    decodeMs: null,
    processingMs: null,
    bitrateMbps: null,
    packetsLost: null,
    packetsLostPerSec: null,
    freezeCount: null,
    nackCount: null,
    decoder: null,
    presentationDelayMs: null,
    serverQueueDepth: null,
    serverLagMs: null,
    serverDeliveryMs: null,
    serverEncodeMs: null,
    serverTrackDroppedPackets: null,
    resolution: null,
    step: null,
  })
  renderMetrics({ force: true })
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
  resetBrowserProfileMetrics()
  disconnecting = false

  try {
    const pc = new RTCPeerConnection()
    const channel = pc.createDataChannel("controls")
    peerConnection = pc
    controlChannel = channel
    const videoTransceiver = pc.addTransceiver("video", { direction: "recvonly" })
    preferH264VideoCodec(videoTransceiver)
    setLowLatencyReceiverHint(videoTransceiver.receiver)

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
      setLowLatencyReceiverHint(event.receiver)
      applyLowLatencyReceiverHints(pc)
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
        applyLowLatencyReceiverHints(pc)
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
  const onFrame = (now, metadata) => {
    if (document.body.classList.contains("has-video")) {
      recordPresentedFrame(now, metadata)
      updateMetricsFromVideo()
    }
    remoteVideo.requestVideoFrameCallback(onFrame)
  }
  remoteVideo.requestVideoFrameCallback(onFrame)
}

function getBrowserProfileSnapshot() {
  return {
    ...metrics,
    connected,
    inferenceInFlight,
    activeKeys: Array.from(activeKeys),
    diagnosis: diagnoseBrowserProfile(),
  }
}

async function getPeerStatsSnapshot() {
  if (!peerConnection) {
    return []
  }
  const stats = await peerConnection.getStats()
  return Array.from(stats.values()).map((report) => ({ ...report }))
}

function getRecentLogMessages(limit = 20) {
  return Array.from(eventLog.querySelectorAll(".logEntry span"))
    .slice(0, limit)
    .map((entry) => entry.textContent)
}

function exposeBrowserProfile() {
  window.omnidreamsWebRTCProfile = {
    snapshot: getBrowserProfileSnapshot,
    peerStats: getPeerStatsSnapshot,
    logs: getRecentLogMessages,
  }
}

function initialize() {
  document.body.dataset.status = "idle"
  exposeBrowserProfile()
  logEvent("viewer ready", { source: "client" })
  setFlow("waiting")
  renderMetrics()
  attachPointerControls()
  startIdleAnimation()
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
