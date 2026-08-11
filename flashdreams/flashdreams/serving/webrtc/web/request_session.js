// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const mockMode = new URLSearchParams(window.location.search).has("mock")

/**
 * @typedef {Object} WebRTCModelAdapter
 * @property {string=} modelName
 * @property {string=} stylesheet
 * @property {Array<{label: string, keys: Array<string|{key: string, label?: string}>}>=} controls
 * @property {{postprocess?: boolean}=} capabilities
 * @property {{endpoint: string, label?: string, placeholder?: string, generateLabel?: string}=} promptGeneration
 * @property {(context: Object) => (void|Promise<void>)=} mount
 * @property {(context: Object) => (void|Promise<void>)=} beforeConnect
 * @property {(action: Object, context: Object) => void=} onActionSent
 * @property {(payload: Object, context: Object) => boolean=} onControlMessage
 * @property {(visible: boolean, context: Object) => void=} onVideoVisibilityChanged
 * @property {(context: Object) => void=} onDisconnect
 */

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
const postprocessField = document.getElementById("postprocessField")
const postprocessSelect = document.getElementById("postprocessSelect")
const modelStageSlot = document.getElementById("modelStageSlot")
const modelStatusSlot = document.getElementById("modelStatusSlot")
const modelPanelSlot = document.getElementById("modelPanelSlot")
const modelControlSlot = document.getElementById("modelControlSlot")
const controlRows = document.getElementById("controlRows")

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

let allowedKeys = new Set()
let controlButtons = []
/** @type {WebRTCModelAdapter|null} */
let modelAdapter = null

let peerConnection = null
let controlChannel = null
let statsTimer = null
let videoMetricsTimer = null
let heartbeatTimer = null
let inferenceInFlight = false
let connected = false
let disconnecting = false
let heldKeySequence = 0
let postprocessAvailable = false
let liveVideoStream = null

const metrics = {
  fps: null,
  targetFps: null,
  latencyMs: null,
  rttMs: null,
  resolution: null,
  step: null,
  model: "World Model",
}

function normalizeKey(rawKey) {
  const key = String(rawKey || "").toLowerCase()
  return keyAliases.get(key) || key
}

function isEditableControlTarget(target) {
  if (!target || typeof target !== "object") {
    return false
  }
  if (target.isContentEditable === true) {
    return true
  }
  const tagName = typeof target.tagName === "string" ? target.tagName.toLowerCase() : ""
  if (["input", "textarea", "select"].includes(tagName)) {
    return true
  }
  return typeof target.closest === "function"
    && target.closest("input, textarea, select, [contenteditable]") !== null
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
  const consoleMessage = `[FlashDreams WebRTC][${source}] ${message}`
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
  modelAdapter?.onVideoVisibilityChanged?.(visible, modelContext)
}

function renderControls(groups) {
  controlRows.replaceChildren()
  allowedKeys = new Set()
  for (const group of groups) {
    if (!group || !Array.isArray(group.keys) || group.keys.length === 0) {
      continue
    }
    const row = document.createElement("div")
    row.className = "controlRow"
    const cluster = document.createElement("div")
    cluster.className = group.keys.length > 2 ? "keyCluster keyClusterWide" : "keyCluster"
    for (const item of group.keys) {
      const key = normalizeKey(typeof item === "string" ? item : item?.key)
      if (!key) {
        continue
      }
      allowedKeys.add(key)
      const button = document.createElement("button")
      button.className = "controlKey"
      button.type = "button"
      button.dataset.controlKey = key
      button.textContent = key.toUpperCase()
      button.setAttribute("aria-label", typeof item === "string" ? key : (item.label || key))
      cluster.append(button)
    }
    const label = document.createElement("span")
    label.textContent = String(group.label || "Controls")
    row.append(cluster, label)
    controlRows.append(row)
  }
  controlButtons = Array.from(controlRows.querySelectorAll("[data-control-key]"))
}

function setPostprocessDisabled(disabled) {
  postprocessSelect.disabled = disabled || !postprocessAvailable
}

async function loadPostprocessOptions() {
  const payload = mockMode
    ? { default_preset: "rtx-super-resolution", presets: ["rtx-super-resolution"] }
    : await fetch("/api/postprocess/options").then(async (response) => {
        if (!response.ok) {
          throw new Error(`post-process options failed (${response.status})`)
        }
        return response.json()
      })
  const presets = Array.isArray(payload.presets) ? payload.presets : []
  const defaultPreset = typeof payload.default_preset === "string"
    ? payload.default_preset
    : ""
  postprocessAvailable = Boolean(defaultPreset && presets.includes(defaultPreset))
  postprocessField.hidden = !postprocessAvailable
  postprocessSelect.replaceChildren(new Option("Off", ""))
  for (const preset of presets) {
    if (typeof preset === "string" && preset) {
      postprocessSelect.append(new Option(preset, preset))
    }
  }
  postprocessSelect.value = postprocessAvailable ? defaultPreset : ""
  setPostprocessDisabled(false)
  if (postprocessAvailable) {
    logEvent(`post-process=${postprocessSelect.value}`, { source: "client" })
  }
}

async function configurePostprocessSession() {
  if (!postprocessAvailable || mockMode) {
    return
  }
  const postprocessPreset = postprocessSelect.value
  const response = await fetch("/api/session/input", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ postprocess_preset: postprocessPreset }),
  })
  if (!response.ok) {
    const text = await response.text()
    throw new Error(`session configuration failed (${response.status}): ${text}`)
  }
  logEvent(`post-process=${postprocessPreset || "off"}`, { source: "client" })
}

function sendModelMessage(payload) {
  if (!connected || !controlChannel || controlChannel.readyState !== "open") {
    return false
  }
  controlChannel.send(JSON.stringify(payload))
  return true
}

function sendModelCommand(payload, label = "model command") {
  if (!sendModelMessage(payload)) {
    setFlow("connect session first")
    return false
  }
  inferenceInFlight = true
  if (promptGenerationControls) promptGenerationControls.generate.disabled = true
  setStatus("Generating", "generating")
  setFlow(`sent ${label}`)
  logEvent(label, { source: "client" })
  return true
}

const modelContext = {
  slots: {
    stage: modelStageSlot,
    status: modelStatusSlot,
    panel: modelPanelSlot,
    controls: modelControlSlot,
  },
  isVideoVisible: () => document.body.classList.contains("has-video"),
  logEvent,
  releaseControls: releaseAllKeys,
  sendCommand: sendModelCommand,
  setModelName(name) {
    if (typeof name === "string" && name) {
      metrics.model = name
      renderMetrics()
    }
  },
  setResolution(width, height) {
    if (Number.isFinite(Number(width)) && Number.isFinite(Number(height))) {
      metrics.resolution = `${Number(width)}x${Number(height)}`
      renderMetrics()
    }
  },
}

async function loadModelAdapter() {
  let adapter = {}
  const stylesheetHrefs = new Set()
  try {
    const response = await fetch("/api/ui/config")
    if (response.ok) {
      const config = await response.json()
      if (typeof config.model_stylesheet === "string" && config.model_stylesheet) {
        stylesheetHrefs.add(config.model_stylesheet)
      }
      if (typeof config.adapter_module === "string" && config.adapter_module) {
        const module = await import(config.adapter_module)
        if (module.default && typeof module.default === "object") {
          adapter = module.default
        }
      }
    }
  } catch (error) {
    logEvent(`model UI unavailable: ${error.message}`, { source: "client", level: "error" })
  }

  modelAdapter = adapter
  if (typeof adapter.stylesheet === "string" && adapter.stylesheet) {
    stylesheetHrefs.add(adapter.stylesheet)
  }
  for (const href of stylesheetHrefs) {
    const stylesheet = document.createElement("link")
    stylesheet.rel = "stylesheet"
    stylesheet.href = href
    document.head.append(stylesheet)
  }
  const modelControls = Array.isArray(adapter.controls) ? adapter.controls : []
  renderControls(modelControls)
  if (typeof adapter.modelName === "string") {
    modelContext.setModelName(adapter.modelName)
  }
  if (adapter.capabilities?.postprocess === true) {
    try {
      await loadPostprocessOptions()
    } catch (error) {
      postprocessAvailable = false
      postprocessField.hidden = true
      setPostprocessDisabled(false)
      logEvent(`post-process unavailable: ${error.message}`, {
        source: "client",
        level: "error",
      })
    }
  }
  configurePromptGeneration(adapter.promptGeneration)
  document.querySelector(".controlCard")?.toggleAttribute("hidden", adapter.promptGeneration?.hideControls === true)
  await adapter.mount?.(modelContext)
}

function renderMetrics() {
  const fps = firstFinite(metrics.fps, metrics.targetFps)
  const latency = firstFinite(metrics.latencyMs, metrics.rttMs)
  fpsValue.textContent = Number.isFinite(fps) ? String(Math.round(fps)) : "--"
  latencyValue.textContent = formatMs(latency)
  resolutionValue.textContent = metrics.resolution || "--"
  stepValue.textContent = metrics.step === null ? "--" : String(metrics.step)
  modelValue.textContent = metrics.model || "World Model"
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
  modelAdapter?.onActionSent?.(action, modelContext)
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
    modelAdapter?.onControlMessage?.(payload, modelContext)
    return
  }

  if (modelAdapter?.onControlMessage?.(payload, modelContext)) {
    return
  }

  if (payload.type === "generation_complete") {
    inferenceInFlight = false
    if (promptGenerationControls) {
      promptGenerationControls.generate.disabled = false
      promptGenerationControls.download.disabled = false
    }
    setStatus("Waiting", "waiting")
    setFlow("generation complete; ready for another prompt")
    logEvent("generation complete", { source: "server" })
    stopPromptRecording()
    const playbackEndpoint = promptGenerationControls?.config.playbackEndpoint
    if (playbackEndpoint) {
      remoteVideo.pause()
      remoteVideo.srcObject = null
      remoteVideo.src = `${playbackEndpoint}?generation=${Date.now()}`
      remoteVideo.loop = true
      remoteVideo.playbackRate = 1
      void remoteVideo.play()
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
    if (promptGenerationControls) promptGenerationControls.generate.disabled = false
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
    console.group(`[FlashDreams WebRTC] peer stats: ${reason}`)
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
    console.warn("[FlashDreams WebRTC] getStats failed", error)
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
  setPostprocessDisabled(false)
  stopPromptRecording()
  modelAdapter?.onDisconnect?.(modelContext)
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
  setPostprocessDisabled(true)
  setStatus("Connecting", "connecting")
  setFlow("creating peer connection")
  logEvent("connecting to server...", { source: "client" })
  disconnecting = false

  try {
    await configurePostprocessSession()
    await modelAdapter?.beforeConnect?.(modelContext)
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
      if (pendingPromptGeneration) {
        pendingPromptGeneration = false
        triggerPromptGeneration()
      }
    }
    channel.onclose = () => {
      connected = false
      setPostprocessDisabled(false)
      if (document.body.dataset.status !== "error") {
        setStatus("Closed", "idle")
      }
      setFlow("channel closed")
      logEvent("control data channel closed", { source: "client" })
      stopHeartbeat()
      stopStatsPolling()
      modelAdapter?.onDisconnect?.(modelContext)
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
        liveVideoStream = stream
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
        setPostprocessDisabled(false)
        modelAdapter?.onDisconnect?.(modelContext)
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
    setPostprocessDisabled(false)
    modelAdapter?.onDisconnect?.(modelContext)
  }
}

function handleKeyDown(event) {
  if (isEditableControlTarget(event.target)) {
    return
  }
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
  if (isEditableControlTarget(event.target)) {
    return
  }
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

async function initialize() {
  document.body.dataset.status = "idle"
  logEvent("viewer ready", { source: "client" })
  setFlow("waiting")
  renderMetrics()
  await loadModelAdapter()
  attachPointerControls()
  window.requestAnimationFrame(drawIdleScene)
  startVideoFrameMonitor()
  connectButton.disabled = false
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

void initialize()

// Shared finite-generation controls. Models opt in through
// `adapter.promptGeneration`; the transport remains model-agnostic.
let promptGenerationControls = null
let pendingPromptGeneration = false
let recording = null
let recordedChunks = []

function configurePromptGeneration(config) {
  if (!config || typeof config.endpoint !== "string" || !config.endpoint) return
  const panel = document.createElement("section")
  panel.className = "promptGenerationPanel overlayPanel"
  const label = config.label || "Prompt"
  const placeholder = config.placeholder || "Describe the video to generate"
  const generateLabel = config.generateLabel || "Generate"
  panel.innerHTML = `<label class="promptGenerationField"><span>${label}</span><textarea rows="5" placeholder="${placeholder}"></textarea></label><div class="promptGenerationActions"><button type="button" class="promptGenerateButton">${generateLabel}</button><button type="button" class="promptPlaybackButton" disabled>Pause</button><button type="button" class="promptDownloadButton" disabled>Download recording</button></div><p class="promptGenerationHint">Keep this session open and submit another prompt whenever you are ready.</p>`
  modelPanelSlot.append(panel)
  const prompt = panel.querySelector("textarea")
  const duration = document.createElement("input")
  duration.type = "number"; duration.min = "1"; duration.max = "60"; duration.value = "5"; duration.className = "promptDurationInput"; duration.setAttribute("aria-label", "Video duration in seconds")
  panel.querySelector(".promptGenerationActions").before(duration)
  const generate = panel.querySelector(".promptGenerateButton")
  const playback = panel.querySelector(".promptPlaybackButton")
  const download = panel.querySelector(".promptDownloadButton")
  promptGenerationControls = { config, prompt, duration, generate, playback, download }
  generate.addEventListener("click", () => void requestPromptGeneration())
  playback.addEventListener("click", () => {
    if (remoteVideo.paused) { void remoteVideo.play(); playback.textContent = "Pause" } else { remoteVideo.pause(); playback.textContent = "Play" }
  })
  download.addEventListener("click", () => downloadPromptRecording())
  remoteVideo.addEventListener("play", startPromptRecording)
}

async function requestPromptGeneration() {
  const controls = promptGenerationControls
  if (!controls) return
  const prompt = controls.prompt.value.trim()
  const duration_s = Number(controls.duration.value)
  if (!Number.isFinite(duration_s) || duration_s < 1 || duration_s > 60) { controls.duration.focus(); setFlow("choose 1–60 seconds"); return }
  if (!prompt) { controls.prompt.focus(); setFlow("enter a prompt"); return }
  controls.generate.disabled = true
  try {
    const response = await fetch(controls.config.endpoint, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ prompt, duration_s }) })
    if (!response.ok) throw new Error(await response.text())
    logEvent("prompt accepted", { source: "client" })
    if (!connected) {
      pendingPromptGeneration = true
      await connectSession()
      return
    }
    triggerPromptGeneration()
  } catch (error) {
    logEvent(`prompt failed: ${error.message}`, { source: "client", level: "error" })
    setStatus("Error", "error")
  } finally {
    if (!inferenceInFlight) controls.generate.disabled = false
  }
}

function triggerPromptGeneration() {
  if (remoteVideo.src) { remoteVideo.pause(); URL.revokeObjectURL(remoteVideo.src); remoteVideo.removeAttribute("src"); remoteVideo.srcObject = liveVideoStream; remoteVideo.loop = false }
  if (!sendModelMessage({ type: "action", action: { event: "step" } })) return
  inferenceInFlight = true
  if (promptGenerationControls) promptGenerationControls.generate.disabled = true
  setStatus("Generating", "generating")
  setFlow("generation started")
  logEvent("generation started", { source: "client" })
}

function startPromptRecording() {
  if (!promptGenerationControls || recording || !remoteVideo.srcObject || !window.MediaRecorder) return
  recordedChunks = []
  recording = new MediaRecorder(remoteVideo.srcObject)
  recording.ondataavailable = event => { if (event.data.size) recordedChunks.push(event.data) }
  recording.onstop = () => { if (promptGenerationControls) promptGenerationControls.download.disabled = recordedChunks.length === 0 }
  recording.start(1000)
  promptGenerationControls.playback.disabled = false
}

function stopPromptRecording() {
  if (recording?.state === "recording") recording.stop()
  recording = null
}

function downloadPromptRecording() {
  const endpoint = promptGenerationControls?.config.downloadEndpoint
  if (endpoint) { window.location.assign(endpoint); return }
}
