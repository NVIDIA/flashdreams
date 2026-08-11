// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const START_ACTION = {
  type: "action",
  action: { event: "step" },
}

let context = null
let uploadCard = null
let uploadInput = null
let uploadStatus = null
let startButton = null
let uploadRequired = true
let hasDefaultInput = false
let inputReady = false
let connected = false
let started = false

function makeUploadCard() {
  const panel = document.createElement("section")
  panel.className = "flashvsrUploadCard overlayPanel"
  panel.setAttribute("aria-label", "FlashVSR video input")
  panel.setAttribute("aria-busy", "false")
  panel.innerHTML =
    '<div class="flashvsrUploadHeader">' +
    '<div><span class="panelLabel">FlashVSR</span>' +
    '<h2>Upscale a video</h2></div>' +
    '<span class="flashvsrUploadBadge">MP4</span>' +
    "</div>" +
    '<p class="flashvsrUploadHint">Choose a source video, connect the session, then start FlashVSR.</p>' +
    '<label class="flashvsrUploadControl">' +
    '<span class="flashvsrFieldLabel">Source video</span>' +
    '<input class="flashvsrVideoInput" type="file" accept="video/mp4,.mp4" aria-describedby="flashvsrUploadStatus">' +
    "</label>" +
    '<span id="flashvsrUploadStatus" class="flashvsrUploadStatus" role="status"></span>' +
    '<button class="flashvsrStartButton" type="button" disabled>Start FlashVSR</button>'
  return panel
}

function setStatus(message, state = "idle") {
  uploadStatus.textContent = message
  uploadStatus.dataset.state = state
}

function updateStartButton() {
  startButton.disabled = !connected || !inputReady || started
  startButton.textContent = started ? "FlashVSR Running" : "Start FlashVSR"
  startButton.setAttribute("aria-busy", started ? "true" : "false")
}

function showInputPrompt() {
  const [file] = uploadInput.files
  if (file) {
    setStatus("Selected " + file.name + ". Connect the session to upload it.", "pending")
  } else if (hasDefaultInput) {
    setStatus("Server input ready. Connect the session or choose an MP4 to override it.", "ready")
  } else {
    setStatus("Choose an MP4 before connecting.", "pending")
  }
}

function applyVideoMetadata(payload) {
  const width = Number(payload?.resolution?.width)
  const height = Number(payload?.resolution?.height)
  if (Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0) {
    context.setResolution(width, height)
  }
}

async function loadInputStatus() {
  const response = await fetch("/api/session/input")
  if (!response.ok) {
    throw new Error("input status failed (" + response.status + ")")
  }
  const payload = await response.json()
  uploadRequired = Boolean(payload.upload_required)
  hasDefaultInput = Boolean(payload.has_default_input)
  inputReady = !uploadRequired
  applyVideoMetadata(payload)
  showInputPrompt()
}

async function uploadSelectedVideo() {
  const [file] = uploadInput.files
  if (!file) {
    if (uploadRequired) {
      throw new Error("Choose an MP4 before connecting.")
    }
    inputReady = true
    return
  }
  if (!file.name.toLowerCase().endsWith(".mp4")) {
    throw new Error("Choose an .mp4 video.")
  }

  setStatus("Uploading and decoding " + file.name + "...", "pending")
  const form = new FormData()
  form.append("video", file, file.name)
  const response = await fetch("/api/session/input", {
    method: "POST",
    body: form,
  })
  if (!response.ok) {
    const text = (await response.text()).trim().replace(/^\d+:\s*/, "")
    throw new Error(text || "video upload failed (" + response.status + ")")
  }
  const payload = await response.json()
  uploadRequired = false
  hasDefaultInput = Boolean(payload.has_default_input)
  inputReady = true
  applyVideoMetadata(payload)
  setStatus(
    "Uploaded " +
      payload.num_frames +
      " frames at " +
      payload.resolution.width +
      "x" +
      payload.resolution.height +
      ". Waiting for connection.",
    "ready",
  )
  context.logEvent("uploaded " + file.name, { source: "client" })
}

function startFlashVSR() {
  if (!connected || !inputReady || started) {
    return
  }
  started = true
  updateStartButton()
  if (!context.sendCommand(START_ACTION, "start FlashVSR")) {
    started = false
    updateStartButton()
    setStatus("The WebRTC control channel is not ready yet.", "error")
    return
  }
  context.setFlow("FlashVSR running")
  setStatus("FlashVSR is running.", "running")
}

export default {
  modelName: "FlashVSR",
  stylesheet: new URL("./adapter.css?v=flashvsr-upload-v3", import.meta.url).href,

  async mount(sharedContext) {
    context = sharedContext
    uploadCard = makeUploadCard()
    context.slots.panel.append(uploadCard)
    uploadInput = uploadCard.querySelector(".flashvsrVideoInput")
    uploadStatus = uploadCard.querySelector(".flashvsrUploadStatus")
    startButton = uploadCard.querySelector(".flashvsrStartButton")
    updateStartButton()
    uploadInput.addEventListener("focus", context.releaseControls)
    uploadInput.addEventListener("change", () => {
      const [file] = uploadInput.files
      uploadRequired = !file && !hasDefaultInput
      inputReady = !file && hasDefaultInput
      showInputPrompt()
      updateStartButton()
      context.releaseControls()
    })
    startButton.addEventListener("click", startFlashVSR)
    try {
      await loadInputStatus()
    } catch (error) {
      setStatus(error.message, "error")
      context.logEvent("input status unavailable: " + error.message, {
        source: "client",
        level: "error",
      })
    }
  },

  async beforeConnect() {
    connected = false
    started = false
    uploadCard.setAttribute("aria-busy", "true")
    uploadInput.disabled = true
    updateStartButton()
    await uploadSelectedVideo()
  },

  onConnect() {
    connected = true
    uploadCard.setAttribute("aria-busy", "false")
    uploadInput.disabled = true
    updateStartButton()
    context.setFlow("ready; click Start FlashVSR")
    setStatus("Connected. Select Start FlashVSR to begin.", "ready")
  },

  onControlMessage(payload) {
    if (payload.type === "chunk_done" && started) {
      context.setStatus("Generating", "generating")
      context.setFlow("FlashVSR running; chunk " + payload.chunk_index + " complete")
    }
    return false
  },

  onDisconnect() {
    connected = false
    started = false
    inputReady = hasDefaultInput
    uploadRequired = !hasDefaultInput
    uploadCard.setAttribute("aria-busy", "false")
    uploadInput.disabled = false
    updateStartButton()
    showInputPrompt()
  },
}
