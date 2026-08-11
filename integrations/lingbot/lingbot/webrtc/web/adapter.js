// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const mockMode = new URLSearchParams(window.location.search).has("mock")

const controls = [
  {
    label: "Drive / Turn",
    keys: [
      { key: "w", label: "Forward" },
      { key: "a", label: "Turn left" },
      { key: "s", label: "Backward" },
      { key: "d", label: "Turn right" },
    ],
  },
  {
    label: "Strafe",
    keys: [
      { key: "q", label: "Strafe left" },
      { key: "e", label: "Strafe right" },
    ],
  },
  {
    label: "Pitch",
    keys: [
      { key: "i", label: "Pitch up" },
      { key: "k", label: "Pitch down" },
    ],
  },
  {
    label: "Look",
    keys: [
      { key: "j", label: "Look left" },
      { key: "l", label: "Look right" },
    ],
  },
]

let context = null
let initialScene = null
let initialSceneLocked = false
let promptEdited = false
let textEventsEdited = false
let firstFrameUrlEdited = false
let firstFrameInputMode = "url"
let selectedFirstFrameFile = null
let selectedFirstFrameUrl = null
let firstFrameSelectionCommitted = false
let activeEventId = null
let textEventDrafts = []
let textEventSequence = 0

let preview = null
let sceneCard = null
let firstFrameSourceRow = null
let uploadModeButton = null
let urlModeButton = null
let firstFrameInput = null
let firstFrameUrlInput = null
let firstFrameUrlUpdateButton = null
let firstFrameUrlStatus = null
let firstFrameName = null
let promptInput = null
let textEventList = null
let addTextEventButton = null
let eventControls = null
let eventButtons = null
let clearEventButton = null

function makeSceneCard() {
  const panel = document.createElement("section")
  panel.className = "sceneCard overlayPanel"
  panel.setAttribute("aria-label", "Initial Scene")
  panel.innerHTML = `
    <span class="panelLabel">Initial Scene</span>
    <div class="firstFrameSourceRow" data-mode="url">
      <div class="sourcePane sourcePaneUpload">
        <button class="sourceModeButton uploadModeButton" type="button">Upload</button>
        <label class="uploadControl">
          <input class="firstFrameInput" type="file" accept="image/*">
          <span class="firstFrameName">Choose Image</span>
        </label>
      </div>
      <div class="sourcePane sourcePaneUrl">
        <button class="sourceModeButton urlModeButton" type="button">URL</button>
        <div class="urlControl">
          <label>Image URL</label>
          <input class="firstFrameUrlInput" type="url" inputmode="url" autocomplete="off">
        </div>
      </div>
      <button class="urlUpdateButton" type="button">Update</button>
    </div>
    <div class="firstFrameUpdateRow">
      <span class="fieldStatus" role="status" hidden></span>
    </div>
    <label class="promptControl">
      <span>Prompt</span>
      <textarea rows="4" maxlength="2000"></textarea>
    </label>
    <div class="textEventEditor">
      <div class="textEventHeader">
        <span>Text Events</span>
        <button class="textEventAddButton" type="button">Add</button>
      </div>
      <div class="textEventList"></div>
    </div>
  `
  return panel
}

function makeEventControls() {
  const root = document.createElement("div")
  root.className = "eventControls"
  root.hidden = true
  root.innerHTML = `
    <div class="eventButtons"></div>
    <button class="eventButton eventButtonClear" type="button">Clear</button>
  `
  return root
}

function bindElements() {
  firstFrameSourceRow = sceneCard.querySelector(".firstFrameSourceRow")
  uploadModeButton = sceneCard.querySelector(".uploadModeButton")
  urlModeButton = sceneCard.querySelector(".urlModeButton")
  firstFrameInput = sceneCard.querySelector(".firstFrameInput")
  firstFrameUrlInput = sceneCard.querySelector(".firstFrameUrlInput")
  firstFrameUrlUpdateButton = sceneCard.querySelector(".urlUpdateButton")
  firstFrameUrlStatus = sceneCard.querySelector(".fieldStatus")
  firstFrameName = sceneCard.querySelector(".firstFrameName")
  promptInput = sceneCard.querySelector(".promptControl textarea")
  textEventList = sceneCard.querySelector(".textEventList")
  addTextEventButton = sceneCard.querySelector(".textEventAddButton")
  eventButtons = eventControls.querySelector(".eventButtons")
  clearEventButton = eventControls.querySelector(".eventButtonClear")
}

function makeTextEventId(label = "") {
  const slug = String(label)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48)
  textEventSequence += 1
  return `${slug || "event"}-${textEventSequence}`
}

function makeTextEventDraft(item = {}) {
  const label = String(item.label || "").trim()
  return {
    event_id: String(item.event_id || item.id || "").trim() || makeTextEventId(label),
    label,
    prompt: String(item.prompt || "").trim(),
  }
}

function setFirstFrameInputMode(mode) {
  if (mode !== "upload" && mode !== "url") {
    return
  }
  firstFrameInputMode = mode
  firstFrameSourceRow.dataset.mode = mode
  uploadModeButton.setAttribute("aria-pressed", mode === "upload" ? "true" : "false")
  urlModeButton.setAttribute("aria-pressed", mode === "url" ? "true" : "false")
}

function setFirstFrameStatus(message = "", state = "idle") {
  firstFrameUrlStatus.textContent = message
  firstFrameUrlStatus.hidden = !message
  firstFrameUrlStatus.dataset.state = state
}

function defaultFirstFrameName() {
  return initialScene?.has_first_frame ? "Example Image" : "Choose Image"
}

function clearSelectedFile() {
  selectedFirstFrameFile = null
  firstFrameSelectionCommitted = false
  firstFrameInput.value = ""
  if (selectedFirstFrameUrl) {
    URL.revokeObjectURL(selectedFirstFrameUrl)
    selectedFirstFrameUrl = null
  }
}

function updatePreview() {
  const selected = selectedFirstFrameUrl && firstFrameSelectionCommitted
  const initial = initialScene?.has_first_frame && initialScene?.first_frame_url
  if (selected) {
    preview.src = selectedFirstFrameUrl
  } else if (initial) {
    const separator = initialScene.first_frame_url.includes("?") ? "&" : "?"
    preview.src = `${initialScene.first_frame_url}${separator}t=${Date.now()}`
  }
  document.body.classList.toggle(
    "is-ready-preview",
    !context.isVideoVisible() && Boolean(selected || initial),
  )
}

function setSessionLocked(locked) {
  initialSceneLocked = locked
  sceneCard.hidden = locked
  for (const input of sceneCard.querySelectorAll("input, textarea, button")) {
    input.disabled = locked
  }
}

function renderTextEventEditor() {
  textEventList.replaceChildren()
  for (const [index, draft] of textEventDrafts.entries()) {
    const row = document.createElement("div")
    row.className = "textEventRow"
    const fields = document.createElement("div")
    fields.className = "textEventFields"
    const label = document.createElement("input")
    label.className = "textEventLabel"
    label.maxLength = 64
    label.placeholder = "Label"
    label.value = draft.label
    const prompt = document.createElement("textarea")
    prompt.className = "textEventPrompt"
    prompt.rows = 2
    prompt.maxLength = 1000
    prompt.placeholder = "Event prompt"
    prompt.value = draft.prompt
    const remove = document.createElement("button")
    remove.className = "textEventRemoveButton"
    remove.type = "button"
    remove.textContent = "X"
    remove.setAttribute("aria-label", `Remove text event ${index + 1}`)
    for (const input of [label, prompt]) {
      input.disabled = initialSceneLocked
      input.addEventListener("focus", context.releaseControls)
    }
    label.addEventListener("input", () => {
      draft.label = label.value
      textEventsEdited = true
    })
    prompt.addEventListener("input", () => {
      draft.prompt = prompt.value
      textEventsEdited = true
    })
    remove.disabled = initialSceneLocked
    remove.addEventListener("click", () => {
      textEventDrafts.splice(index, 1)
      textEventsEdited = true
      renderTextEventEditor()
    })
    fields.append(label, prompt)
    row.append(fields, remove)
    textEventList.append(row)
  }
}

function collectTextEvents() {
  const events = []
  const usedIds = new Set()
  for (const draft of textEventDrafts) {
    const label = draft.label.trim()
    const prompt = draft.prompt.trim()
    if (!label && !prompt) {
      continue
    }
    if (!prompt) {
      throw new Error("Each text event needs a prompt.")
    }
    let eventId = String(draft.event_id || "").trim() || makeTextEventId(label)
    while (usedIds.has(eventId)) {
      eventId = makeTextEventId(label)
    }
    draft.event_id = eventId
    usedIds.add(eventId)
    events.push({ event_id: eventId, label: label || eventId, prompt, category: "custom" })
  }
  return events
}

function renderEventControls() {
  const catalog = Array.isArray(initialScene?.event_catalog) ? initialScene.event_catalog : []
  eventControls.hidden = catalog.length === 0
  eventButtons.replaceChildren()
  for (const item of catalog) {
    const eventId = String(item.event_id || "").trim()
    if (!eventId) {
      continue
    }
    const button = document.createElement("button")
    button.className = "eventButton"
    button.type = "button"
    button.textContent = String(item.label || eventId)
    button.classList.toggle("is-active", activeEventId === eventId)
    button.addEventListener("click", () => sendTextEvent(eventId, "trigger"))
    eventButtons.append(button)
  }
  clearEventButton.classList.toggle("is-active", activeEventId === null)
}

function applyInitialScene(scene) {
  initialScene = scene
  if (!promptEdited && typeof scene.prompt === "string") {
    promptInput.value = scene.prompt
  }
  const imageUrl = typeof scene.image_url === "string"
    ? scene.image_url
    : (typeof scene.default_image_url === "string" ? scene.default_image_url : "")
  if (!selectedFirstFrameFile && !firstFrameUrlEdited && imageUrl) {
    firstFrameUrlInput.value = imageUrl
    setFirstFrameInputMode("url")
  }
  firstFrameName.textContent = firstFrameUrlInput.value.trim() ? "Upload Image" : defaultFirstFrameName()
  activeEventId = scene.active_event_id || null
  if (!textEventsEdited) {
    textEventDrafts = Array.isArray(scene.event_catalog)
      ? scene.event_catalog.map((item) => makeTextEventDraft(item))
      : []
    renderTextEventEditor()
  }
  renderEventControls()
  context.setModelName(scene.model || "Lingbot")
  applyVideoSizing(scene.resolution)
  context.setResolution(scene.resolution?.width, scene.resolution?.height)
  updatePreview()
}

function applyVideoSizing(resolution) {
  const width = Number(resolution?.width)
  const height = Number(resolution?.height)
  if (
    !Number.isFinite(width) ||
    !Number.isFinite(height) ||
    width <= 0 ||
    height <= 0
  ) {
    return
  }
  const style = document.documentElement.style
  style.setProperty("--lingbot-video-width", `${width}px`)
  style.setProperty("--lingbot-video-height", `${height}px`)
  style.setProperty("--lingbot-video-width-from-vh", `${(width / height) * 100}vh`)
  style.setProperty("--lingbot-video-aspect", `${width} / ${height}`)
}

function mockInitialScene() {
  return {
    prompt: "Drive through a cinematic city street at sunset.",
    has_first_frame: false,
    model: "Lingbot",
    resolution: { width: 832, height: 464 },
    event_catalog: [
      { event_id: "portal", label: "Portal", prompt: "A luminous portal opens." },
      { event_id: "storm", label: "Storm", prompt: "A dramatic storm rolls in." },
    ],
  }
}

async function loadInitialScene() {
  if (mockMode) {
    applyInitialScene(mockInitialScene())
    return
  }
  const response = await fetch("/api/session/initial_scene")
  if (!response.ok) {
    throw new Error(`initial scene failed (${response.status})`)
  }
  applyInitialScene(await response.json())
}

function validateImageUrl(value) {
  const imageUrl = value.trim()
  let parsed = null
  try {
    parsed = new URL(imageUrl)
  } catch {
    throw new Error("Enter a valid http(s) image URL.")
  }
  if (!["http:", "https:"].includes(parsed.protocol)) {
    throw new Error("Enter a valid http(s) image URL.")
  }
  return imageUrl
}

async function uploadSessionInput({ includeFirstFrame = false } = {}) {
  const prompt = promptInput.value.trim()
  const hasPrompt = promptEdited && Boolean(prompt)
  const hasFile = includeFirstFrame && firstFrameInputMode === "upload" && selectedFirstFrameFile
  let imageUrl = firstFrameUrlInput.value.trim()
  const hasUrl = includeFirstFrame && firstFrameInputMode === "url" && Boolean(imageUrl)
  const textEvents = textEventsEdited ? collectTextEvents() : null
  if (!hasPrompt && !hasFile && !hasUrl && textEvents === null) {
    return
  }
  if (hasUrl) {
    imageUrl = validateImageUrl(imageUrl)
  }
  if (mockMode) {
    applyInitialScene({
      ...mockInitialScene(),
      prompt: hasPrompt ? prompt : initialScene.prompt,
      event_catalog: textEvents ?? initialScene.event_catalog,
      active_event_id: activeEventId,
    })
  } else {
    const form = new FormData()
    if (hasPrompt) form.append("prompt", prompt)
    if (hasFile) form.append("image", selectedFirstFrameFile, selectedFirstFrameFile.name)
    if (hasUrl) form.append("image_url", imageUrl)
    if (textEvents !== null) form.append("text_events", JSON.stringify(textEvents))
    const response = await fetch("/api/session/input", { method: "POST", body: form })
    if (!response.ok) {
      const text = (await response.text()).trim().replace(/^\d+:\s*/, "")
      throw new Error(text || `input upload failed (${response.status})`)
    }
    applyInitialScene(await response.json())
  }
  promptEdited = false
  textEventsEdited = false
  firstFrameUrlEdited = false
}

async function updateFirstFrame() {
  if (initialSceneLocked) return
  try {
    if (firstFrameInputMode === "upload" && !selectedFirstFrameFile) {
      throw new Error("Choose an image file.")
    }
    if (firstFrameInputMode === "url") {
      firstFrameUrlInput.value = validateImageUrl(firstFrameUrlInput.value)
      clearSelectedFile()
    }
    setFirstFrameStatus("Updating...", "pending")
    firstFrameUrlUpdateButton.disabled = true
    await uploadSessionInput({ includeFirstFrame: true })
    firstFrameSelectionCommitted = true
    setFirstFrameStatus("Updated", "success")
    updatePreview()
  } catch (error) {
    setFirstFrameStatus(error.message, "error")
    context.logEvent(`first frame update failed: ${error.message}`, { source: "client", level: "error" })
  } finally {
    firstFrameUrlUpdateButton.disabled = initialSceneLocked
  }
}

function sendTextEvent(eventId, state) {
  const label = state === "clear" ? "clear event" : `event:${eventId}`
  if (!context.sendCommand({ type: "event", event_id: eventId, state }, label)) {
    return
  }
  setSessionLocked(true)
}

function attachListeners() {
  uploadModeButton.addEventListener("click", () => {
    setFirstFrameInputMode("upload")
    context.releaseControls()
  })
  urlModeButton.addEventListener("click", () => {
    setFirstFrameInputMode("url")
    context.releaseControls()
  })
  firstFrameInput.addEventListener("change", () => {
    setFirstFrameInputMode("upload")
    const [file] = firstFrameInput.files
    selectedFirstFrameFile = file || null
    firstFrameSelectionCommitted = false
    if (selectedFirstFrameUrl) URL.revokeObjectURL(selectedFirstFrameUrl)
    selectedFirstFrameUrl = selectedFirstFrameFile ? URL.createObjectURL(selectedFirstFrameFile) : null
    firstFrameName.textContent = selectedFirstFrameFile?.name || defaultFirstFrameName()
    firstFrameUrlInput.value = ""
    firstFrameUrlEdited = false
    setFirstFrameStatus(selectedFirstFrameFile ? "Image not updated" : "", "pending")
  })
  firstFrameUrlInput.addEventListener("input", () => {
    setFirstFrameInputMode("url")
    if (selectedFirstFrameFile) clearSelectedFile()
    firstFrameUrlEdited = true
    firstFrameName.textContent = firstFrameUrlInput.value.trim() ? "Upload Image" : defaultFirstFrameName()
    setFirstFrameStatus(firstFrameUrlInput.value.trim() ? "URL not updated" : "", "pending")
  })
  firstFrameUrlUpdateButton.addEventListener("click", () => void updateFirstFrame())
  promptInput.addEventListener("input", () => { promptEdited = true })
  addTextEventButton.addEventListener("click", () => {
    textEventDrafts.push(makeTextEventDraft())
    textEventsEdited = true
    renderTextEventEditor()
    context.releaseControls()
  })
  clearEventButton.addEventListener("click", () => sendTextEvent(activeEventId || "clear", "clear"))
  for (const input of [firstFrameUrlInput, promptInput, addTextEventButton]) {
    input.addEventListener("focus", context.releaseControls)
  }
}

export default {
  modelName: "Lingbot",
  stylesheet: new URL("./adapter.css?v=lingbot-video-size-v2", import.meta.url).href,
  controls,

  async mount(sharedContext) {
    context = sharedContext
    preview = document.createElement("img")
    preview.className = "firstFramePreview"
    preview.alt = ""
    preview.setAttribute("aria-hidden", "true")
    sceneCard = makeSceneCard()
    eventControls = makeEventControls()
    context.slots.stage.append(preview)
    context.slots.panel.append(sceneCard)
    context.slots.controls.append(eventControls)
    bindElements()
    setFirstFrameInputMode("url")
    attachListeners()
    try {
      await loadInitialScene()
    } catch (error) {
      context.logEvent(`initial scene unavailable: ${error.message}`, { source: "client", level: "error" })
    }
  },

  async beforeConnect() {
    await uploadSessionInput()
  },

  onActionSent() {
    setSessionLocked(true)
    updatePreview()
  },

  onControlMessage(payload) {
    if (payload.type === "chunk_done" && Object.prototype.hasOwnProperty.call(payload, "active_event_id")) {
      activeEventId = payload.active_event_id || null
      renderEventControls()
      return false
    }
    if (payload.type === "event_ack") {
      activeEventId = payload.active_event_id || null
      renderEventControls()
      context.logEvent(`event ${payload.event_id} ${payload.state}`)
      return true
    }
    return false
  },

  onVideoVisibilityChanged() {
    updatePreview()
  },

  onDisconnect() {
    setSessionLocked(false)
    updatePreview()
  },
}
