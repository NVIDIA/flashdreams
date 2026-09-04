// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

const mockMode = new URLSearchParams(window.location.search).has("mock")
// ?director shows the separate Director Controls panel (environment/pacing
// events, e.g. weather and hazards) alongside Player Controls -- hidden by
// default since this is a second operator's role, not the regular player's.
// Also toggleable at runtime via the "Director Mode" button (no URL edit
// needed); mutable for that reason.
let directorMode = new URLSearchParams(window.location.search).has("director")

// Letter hotkeys for event buttons, in assignment order. Excludes the
// reserved movement keys (w/a/s/d/q/e/i/j/k/l) and "c" (Clear's own
// hotkey) and "r" (Restart's) so they never collide with driving input
// or each other. Letters
// run out before director-heavy presets do -- events past the pool size
// simply render without a hotkey, same as the old 9-key digit cap.
const EVENT_HOTKEY_LETTERS = ["b", "f", "g", "h", "m", "n", "o", "p", "t", "u", "v", "x", "y", "z"]

// Jump/Crouch are player movement actions, not narrative/pacing triggers --
// rendered in their own row next to the movement key grid instead of the
// general event button list.
const MOVEMENT_ACTION_EVENT_IDS = new Set(["jump", "crouch"])

function presetSlug(name) {
  return name.toLowerCase().trim().replace(/\s+/g, "-")
}

function findPresetIndexBySlug(slug) {
  if (!slug) return -1
  const normalized = slug.toLowerCase().trim()
  return scenePresets.findIndex((preset) => presetSlug(preset.name) === normalized)
}

// Loaded from scene_presets.json (a sibling file, fetched relative to this
// module) rather than inlined here, so presets/events can be hand-edited
// without touching JS. Populated by loadScenePresets(), awaited in mount()
// before anything that reads it (dropdown build, default-preset apply).
let scenePresets = []

async function loadScenePresets() {
  const url = new URL("./scene_presets.json", import.meta.url).href
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`scene_presets.json fetch failed (${response.status})`)
  }
  const data = await response.json()
  if (!Array.isArray(data)) {
    throw new Error("scene_presets.json must contain a JSON array.")
  }
  scenePresets = data
}

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
// The upload that carries the current catalog to the server, if one is
// still in flight. An event triggered before it lands would be rejected
// with "Unknown event_id", since the server only knows what it was told.
let pendingCatalogUpload = null
let firstFrameUrlEdited = false
let firstFrameInputMode = "url"
let selectedFirstFrameFile = null
let selectedFirstFrameUrl = null
let firstFrameSelectionCommitted = false
let activeEventId = null
let textEventDrafts = []
let textEventSequence = 0

let preview = null
let gameOverOverlay = null
let sceneCard = null
let presetSelect = null
let savePresetButton = null
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
let actionButtons = null
let clearEventButton = null
let restartButton = null
let livePromptInput = null
let livePromptSubmitButton = null
let directorButtons = null
let controlsModeToggle = null
let enableDirectorModeButton = null
let playerPromptGroup = null
// The shared movement key grid (w/a/s/d/q/e/i/j/k/l) lives outside our own
// panel in the shared page, addressable by its fixed id -- hidden while
// the Director tab is active since a director doesn't drive movement.
const movementControlRows = document.getElementById("controlRows")
// The shared panel's own heading text, also addressable by a fixed id --
// swapped between "Player Controls" / "Director Controls" to match
// whichever mode is active.
const controlsPanelTitleText = document.getElementById("controlsPanelTitleText")
let healthBar = null
let healthBarFill = null
let healthBarValue = null
let healthBarLabelText = null
// Purely a client-side cosmetic HUD -- the server/runtime has no concept of
// health, this just tracks event `health` deltas locally (matching the
// original REACTOR case files' HUD) so the presets feel game-like.
let currentHealth = 100
let maxHealth = 100
let directorPromptGroup = null
let directorPromptInput = null
let directorPromptSubmitButton = null
// Player Controls is always the default view, even when ?director is set
// (which only reveals the toggle switch) -- Director Controls requires an
// explicit toggle click.
let showingDirectorControls = false
let currentPreset = null
// The image the selected preset asks for. Held separately because the
// firstFrameUrlEdited guard is cleared once an upload completes, after
// which a scene arriving from the server would overwrite the field with
// whatever the session happens to be running -- showing dragon.jpg while
// every other control said Jet Ski Cruise.
let presetImageUrl = null

function makeSceneCard() {
  const panel = document.createElement("section")
  panel.className = "sceneCard overlayPanel"
  panel.setAttribute("aria-label", "Initial Scene")
  panel.innerHTML = `
    <span class="panelLabel">Initial Scene</span>
    <div class="presetsControl">
      <label for="scenePresetsSelect">Quick Start</label>
      <div class="presetsRow">
        <select id="scenePresetsSelect">
          <option value="">-- Choose a preset --</option>
          ${scenePresets.map((p, i) => `<option value="${i}">${p.name}</option>`).join("")}
        </select>
        <button class="savePresetButton" type="button">Save</button>
      </div>
    </div>
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
    <div class="promptControlGroup">
      <label class="promptControl">
        <span>Prompt</span>
        <textarea rows="4" maxlength="2000"></textarea>
      </label>
      <button class="promptSubmitButton" type="button">Send</button>
    </div>
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
    <button class="controlsModeToggle" type="button" role="switch" aria-checked="false" hidden>
      <span class="controlsModeToggleTrack"><span class="controlsModeToggleKnob"></span></span>
      <span class="controlsModeToggleLabel">Director Mode</span>
    </button>
    <div class="healthBar">
      <div class="healthBarLabel">
        <span class="healthBarLabelText">Health</span>
        <span class="healthBarValue">100/100</span>
      </div>
      <div class="healthBarTrack"><div class="healthBarFill"></div></div>
    </div>
    <div class="eventButtons actionButtons" hidden></div>
    <div class="eventButtons"></div>
    <div class="eventButtons directorButtons" hidden></div>
    <button class="eventButton eventButtonClear" type="button">Clear (C)</button>
    <button class="eventButton eventButtonRestart" type="button">Restart (R)</button>
    <button class="enableDirectorModeButton" type="button" hidden>Enable Director Mode</button>
    <div class="promptControlGroup playerPromptGroup">
      <label class="promptControl">
        <span>Custom Prompt</span>
        <input type="text" maxlength="2000">
      </label>
      <button class="promptSubmitButton" type="button">Send</button>
    </div>
    <div class="promptControlGroup directorPromptGroup" hidden>
      <label class="promptControl">
        <span>Director Prompt</span>
        <input type="text" maxlength="2000">
      </label>
      <button class="promptSubmitButton" type="button">Send</button>
    </div>
  `
  return root
}

function bindElements() {
  presetSelect = sceneCard.querySelector("#scenePresetsSelect")
  savePresetButton = sceneCard.querySelector(".savePresetButton")
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
  eventButtons = eventControls.querySelector(".eventButtons:not(.actionButtons):not(.directorButtons)")
  actionButtons = eventControls.querySelector(".actionButtons")
  // Jump/Crouch render as their own row right after the shared movement
  // key grid, so they read as part of "movement" rather than the general
  // event/trigger list below.
  if (movementControlRows) movementControlRows.after(actionButtons)
  clearEventButton = eventControls.querySelector(".eventButtonClear")
  restartButton = eventControls.querySelector(".eventButtonRestart")
  directorButtons = eventControls.querySelector(".directorButtons")
  controlsModeToggle = eventControls.querySelector(".controlsModeToggle")
  // Physically move ahead of the shared movement key grid (a separate,
  // earlier DOM sibling this panel doesn't otherwise control) so it's
  // genuinely the first control in the panel, not just first within our
  // own content.
  if (movementControlRows) movementControlRows.before(controlsModeToggle)
  enableDirectorModeButton = eventControls.querySelector(".enableDirectorModeButton")
  playerPromptGroup = eventControls.querySelector(".playerPromptGroup")
  healthBar = eventControls.querySelector(".healthBar")
  // Query healthBar's own descendants BEFORE relocating it below -- once
  // moved out of eventControls, eventControls.querySelector(...) can no
  // longer find them (they're no longer inside its subtree), which left
  // these all null and silently broke every health bar update.
  healthBarFill = healthBar.querySelector(".healthBarFill")
  healthBarValue = healthBar.querySelector(".healthBarValue")
  healthBarLabelText = healthBar.querySelector(".healthBarLabelText")
  // Also moved ahead of the movement grid, right after the toggle (so
  // order is: toggle, health bar, movement grid, then the rest of this
  // panel's own content).
  if (movementControlRows) movementControlRows.before(healthBar)
  livePromptInput = eventControls.querySelector(".playerPromptGroup .promptControl input")
  livePromptSubmitButton = eventControls.querySelector(".playerPromptGroup .promptSubmitButton")
  directorPromptGroup = eventControls.querySelector(".directorPromptGroup")
  directorPromptInput = eventControls.querySelector(".directorPromptGroup .promptControl input")
  directorPromptSubmitButton = eventControls.querySelector(".directorPromptGroup .promptSubmitButton")
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
    // Player by default -- only true for a preset's own directorEvents
    // (tagged explicitly in applyPreset) or a custom event flipped via the
    // Director checkbox in the Text Events editor.
    isDirector: Boolean(item.isDirector),
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

function forgetPresetImage() {
  // A hand-typed URL or an uploaded file replaces the preset's own image, so
  // it should stop overriding what the panel shows.
  presetImageUrl = null
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
  // A picked preset (or manually typed URL) that hasn't been pushed to the
  // server yet via "Update" -- must take priority over the server's own
  // default first-frame endpoint below, otherwise applyInitialScene()'s
  // unconditional updatePreview() call stomps the just-applied preset
  // image back to the server default on every load/re-fetch.
  const pendingUrl = firstFrameUrlEdited && firstFrameUrlInput.value.trim()
  const initial = initialScene?.has_first_frame && initialScene?.first_frame_url
  if (selected) {
    preview.src = selectedFirstFrameUrl
  } else if (pendingUrl) {
    preview.src = pendingUrl
  } else if (initial) {
    const separator = initialScene.first_frame_url.includes("?") ? "&" : "?"
    preview.src = `${initialScene.first_frame_url}${separator}t=${Date.now()}`
  }
  document.body.classList.toggle(
    "is-ready-preview",
    !context.isVideoVisible() && Boolean(selected || pendingUrl || initial),
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
    const directorToggle = document.createElement("label")
    directorToggle.className = "textEventDirectorToggle"
    const directorCheckbox = document.createElement("input")
    directorCheckbox.type = "checkbox"
    directorCheckbox.checked = Boolean(draft.isDirector)
    const directorToggleText = document.createElement("span")
    directorToggleText.textContent = "Director"
    directorToggle.append(directorCheckbox, directorToggleText)
    const remove = document.createElement("button")
    remove.className = "textEventRemoveButton"
    remove.type = "button"
    remove.textContent = "X"
    remove.setAttribute("aria-label", `Remove text event ${index + 1}`)
    for (const input of [label, prompt, directorCheckbox]) {
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
    directorCheckbox.addEventListener("change", () => {
      draft.isDirector = directorCheckbox.checked
      textEventsEdited = true
      // Live Player/Director Controls buttons need to reflect which panel
      // this event now belongs to immediately, same as label/prompt edits
      // becoming visible on the next render.
      renderEventControls()
    })
    remove.disabled = initialSceneLocked
    remove.addEventListener("click", () => {
      textEventDrafts.splice(index, 1)
      textEventsEdited = true
      renderTextEventEditor()
      renderEventControls()
    })
    fields.append(label, prompt, directorToggle)
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

let eventHotkeyMap = new Map()

function getEventHealthDelta(eventId) {
  // Looked up from currentPreset's own definitions (not the transient
  // render catalog) so it still works once connected, after the server's
  // echoed event_catalog -- which has no health field -- takes over as the
  // render source.
  const fromPlayer = currentPreset?.events?.find((item) => item.event_id === eventId)
  const fromDirector = currentPreset?.directorEvents?.find((item) => item.event_id === eventId)
  const health = (fromPlayer ?? fromDirector)?.health
  return Number.isFinite(health) ? health : 0
}

function resetHealth(preset) {
  maxHealth = Number(preset?.hud?.maxHealth) || 100
  currentHealth = maxHealth
  if (healthBarLabelText) healthBarLabelText.textContent = preset?.hud?.healthLabel || "Health"
  renderHealthBar()
}

function renderHealthBar() {
  if (gameOverOverlay) gameOverOverlay.hidden = currentHealth > 0
  if (!healthBarFill) return
  const pct = maxHealth > 0 ? Math.max(0, Math.min(100, (currentHealth / maxHealth) * 100)) : 0
  healthBarFill.style.width = `${pct}%`
  healthBarFill.classList.toggle("is-low", pct <= 25)
  healthBarValue.textContent = `${Math.round(currentHealth)}/${Math.round(maxHealth)}`
}

function applyHealthDelta(delta, label = "") {
  if (!Number.isFinite(delta) || delta === 0) return
  currentHealth = Math.max(0, Math.min(maxHealth, currentHealth + delta))
  renderHealthBar()
  const sign = delta > 0 ? "+" : ""
  const healthLabel = healthBarLabelText?.textContent || "health"
  context?.logEvent(
    `${label ? `${label}: ` : ""}${healthLabel.toLowerCase()} ${sign}${delta} (now ${Math.round(currentHealth)}/${Math.round(maxHealth)})`,
    { source: "client" },
  )
}

function isDirectorEventId(eventId) {
  return Boolean(currentPreset?.directorEvents?.some((item) => item.event_id === eventId))
}

// textEventDrafts items carry their own isDirector flag (set in
// applyPreset()/the Text Events editor's Director checkbox); anything else
// (e.g. the server-echoed event_catalog, before any draft carries the
// field) falls back to the preset-membership check above.
function eventIsDirector(item) {
  return typeof item.isDirector === "boolean" ? item.isDirector : isDirectorEventId(item.event_id)
}

function makeEventButton(item, hotkeyLetter, healthDelta = 0) {
  const eventId = String(item.event_id || "").trim()
  if (!eventId) return null
  const label = String(item.label || eventId)
  const button = document.createElement("button")
  button.className = "eventButton"
  button.type = "button"
  const hotkeyText =
    hotkeyLetter === "space" ? "Space" : hotkeyLetter === "control" ? "Ctrl" : hotkeyLetter ? hotkeyLetter.toUpperCase() : null
  button.append(document.createTextNode(hotkeyText ? `${label} (${hotkeyText})` : label))
  if (Number.isFinite(healthDelta) && healthDelta !== 0) {
    const healthTag = document.createElement("span")
    healthTag.className = `eventButtonHealth ${healthDelta > 0 ? "is-positive" : "is-negative"}`
    healthTag.textContent = healthDelta > 0 ? `+${healthDelta}` : String(healthDelta)
    button.append(healthTag)
  }
  button.classList.toggle("is-active", activeEventId === eventId)
  button.addEventListener("click", () => sendTextEvent(eventId, "trigger"))
  return button
}

function renderEventControls() {
  // A picked preset's events (textEventDrafts, not yet pushed to the
  // server via connect/Send) must take priority over the server's last-
  // known event_catalog -- otherwise applyInitialScene()'s unconditional
  // call here stomps the just-applied preset's events back to whatever
  // the server currently has (its default catalog on first load).
  const catalog = textEventsEdited
    ? textEventDrafts
    : Array.isArray(initialScene?.event_catalog) ? initialScene.event_catalog : []
  const playerItems = catalog.filter((item) => !eventIsDirector(item))
  const directorItems = catalog.filter((item) => eventIsDirector(item))

  // Player events get digit hotkeys (1-9), director events get letter
  // hotkeys from EVENT_HOTKEY_LETTERS -- the two keyspaces never collide,
  // so both live in the same eventHotkeyMap. Jump always gets Space and
  // Crouch always gets Ctrl instead of a digit, matching common game
  // convention. Events past either pool's size still render, just without
  // a hotkey.
  eventHotkeyMap = new Map()
  let digitIndex = 0
  const nextDigit = () => (digitIndex < 9 ? String(++digitIndex) : null)
  let letterIndex = 0
  const nextLetter = () => EVENT_HOTKEY_LETTERS[letterIndex++] ?? null

  eventButtons.replaceChildren()
  actionButtons.replaceChildren()
  for (const item of playerItems) {
    const eventId = String(item.event_id || "").trim()
    const hotkey = eventId === "jump" ? "space" : eventId === "crouch" ? "control" : nextDigit()
    const button = makeEventButton(item, hotkey, getEventHealthDelta(eventId))
    if (!button) continue
    if (hotkey) eventHotkeyMap.set(hotkey, eventId)
    const container = MOVEMENT_ACTION_EVENT_IDS.has(eventId) ? actionButtons : eventButtons
    container.append(button)
  }
  clearEventButton.classList.toggle("is-active", activeEventId === null)

  directorButtons.replaceChildren()
  for (const item of directorItems) {
    const letter = nextLetter()
    const eventId = String(item.event_id || "").trim()
    const button = makeEventButton(item, letter, getEventHealthDelta(eventId))
    if (!button) continue
    if (letter) eventHotkeyMap.set(letter, eventId)
    directorButtons.append(button)
  }

  // In director mode, Player and Director share one panel with a single
  // toggle switch on top swapping which button grid (and which
  // custom-prompt box) is visible -- otherwise (the common case) it's just
  // Player Controls with no toggle at all. When director mode isn't on yet
  // but this preset actually has director events, offer "Enable Director
  // Mode" instead of requiring a URL edit.
  const hasDirectorContent = directorMode && directorItems.length > 0
  enableDirectorModeButton.hidden = directorMode || directorItems.length === 0
  controlsModeToggle.hidden = !hasDirectorContent
  if (!hasDirectorContent) showingDirectorControls = false
  const showDirector = hasDirectorContent && showingDirectorControls
  controlsModeToggle.classList.toggle("is-on", showDirector)
  controlsModeToggle.setAttribute("aria-checked", String(showDirector))
  controlsPanelTitleText.textContent = showDirector ? "Director Controls" : "Player Controls"
  eventButtons.hidden = showDirector
  actionButtons.hidden = showDirector || actionButtons.children.length === 0
  directorButtons.hidden = !showDirector
  directorPromptGroup.hidden = !showDirector
  playerPromptGroup.hidden = showDirector
  if (movementControlRows) movementControlRows.hidden = showDirector
  eventControls.hidden = playerItems.length === 0 && directorItems.length === 0
}

function enableDirectorMode() {
  directorMode = true
  const url = new URL(window.location.href)
  url.searchParams.set("director", "")
  window.history.replaceState(null, "", url)
  renderEventControls()
}

function setDirectorView(showDirector) {
  showingDirectorControls = showDirector
  renderEventControls()
}

function saveCurrentPreset() {
  const name = prompt("Preset name:", "My Scene").trim()
  if (!name) return
  const preset = {
    name,
    prompt: promptInput.value.trim(),
    events: textEventDrafts.map(d => ({ event_id: d.event_id, label: d.label, prompt: d.prompt }))
  }
  scenePresets.push(preset)
  localStorage.setItem("lingbot-presets", JSON.stringify(scenePresets))
  updatePresetDropdown()
  alert(`Preset "${name}" saved!`)
}

function updatePresetDropdown() {
  presetSelect.innerHTML = `
    <option value="">-- Choose a preset --</option>
    ${scenePresets.map((p, i) => `<option value="${i}">${p.name}</option>`).join("")}
  `
}

function loadSavedPresets() {
  try {
    const saved = localStorage.getItem("lingbot-presets")
    if (saved) {
      const customPresets = JSON.parse(saved)
      scenePresets.push(...customPresets)
    }
  } catch (err) {
    console.error("Failed to load saved presets:", err)
  }
}

function applyPreset(presetIndex) {
  const preset = scenePresets[Number(presetIndex)]
  if (!preset) return
  currentPreset = preset
  resetHealth(preset)
  context.logEvent(`preset selected: ${preset.name}`, { source: "client" })
  // Server hard cap (session.py: _MAX_TEXT_EVENTS). Catch an over-budget
  // preset here, at selection time, instead of only discovering it via a
  // failed connect attempt later.
  const totalEventCount = preset.events.length + (preset.directorEvents?.length ?? 0)
  if (totalEventCount > 20) {
    context.logEvent(
      `preset "${preset.name}" has ${totalEventCount} events (player + director combined), `
        + "over the server's 20-event limit -- connecting will fail until it's trimmed.",
      { source: "client", level: "error" },
    )
  }
  const url = new URL(window.location.href)
  url.searchParams.set("preset", presetSlug(preset.name))
  window.history.replaceState(null, "", url)
  promptInput.value = preset.prompt
  promptEdited = true
  // The full catalog (player + director) always uploads to the server --
  // "director" is a client-side UI distinction only (which panel a button
  // renders in, and whether that panel is visible at all), the shared
  // WebRTC protocol has no such concept, so both must be known server-side
  // for either panel's buttons to actually do anything once clicked.
  textEventDrafts = [
    ...preset.events.map((item) => makeTextEventDraft({ ...item, isDirector: false })),
    ...(preset.directorEvents ?? []).map((item) => makeTextEventDraft({ ...item, isDirector: true })),
  ]
  textEventsEdited = true
  renderTextEventEditor()
  // Push the preset to the server now rather than waiting for a connect.
  // The event buttons are live before connecting, and the server rejects an
  // id it has not been told about -- "Unknown event_id='jump'" while the page
  // showed Jump, because the catalog only travelled in beforeConnect.
  pendingCatalogUpload = uploadSessionInput({ includeFirstFrame: true })
    .catch((error) => {
      context.logEvent(`preset not applied: ${error.message}`, {
        source: "client",
        level: "error",
      })
    })
    .finally(() => {
      pendingCatalogUpload = null
    })
  // Also refresh the live Player Controls buttons immediately, not just
  // the editable Text Events list -- otherwise switching games mid-session
  // only updates on the next connect/upload, not on selection itself.
  renderEventControls()
  if (preset.image) {
    clearSelectedFile()
    setFirstFrameInputMode("url")
    firstFrameUrlInput.value = preset.image
    firstFrameUrlEdited = true
    presetImageUrl = preset.image
    firstFrameName.textContent = "Upload Image"
    setFirstFrameStatus("URL not updated", "pending")
    // Show the preset's image immediately, ahead of the "Update" commit
    // step -- picking a preset should visibly change the panel, not just
    // silently populate the URL field.
    preview.src = preset.image
    document.body.classList.toggle("is-ready-preview", !context.isVideoVisible())
  }
  context.releaseControls()
}

function applyInitialScene(scene) {
  initialScene = scene
  if (!promptEdited && typeof scene.prompt === "string") {
    promptInput.value = scene.prompt
  }
  const sceneImageUrl = typeof scene.image_url === "string"
    ? scene.image_url
    : (typeof scene.default_image_url === "string" ? scene.default_image_url : "")
  // A chosen preset wins over the session's own image: the page should show
  // the scene the user picked, not the one the rollout has not switched to
  // yet.
  const imageUrl = presetImageUrl || sceneImageUrl
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
    // Resolved against the page, so a packaged relative path -- which is how
    // the built-in presets ship their images ("assets/circuit.jpg") -- is as
    // valid as an absolute URL. The entered value is returned unchanged, so a
    // relative preset URL stays relative.
    parsed = new URL(imageUrl, window.location.href)
  } catch {
    throw new Error("Enter an http(s) image URL or a path like assets/name.jpg.")
  }
  if (!["http:", "https:"].includes(parsed.protocol)) {
    throw new Error("Enter an http(s) image URL or a path like assets/name.jpg.")
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

async function sendTextEvent(eventId, state, promptValue = null) {
  const label = state === "clear" ? "clear event" : `event:${eventId}`
  const payload = { type: "event", event_id: eventId, state }
  if (promptValue !== null) {
    payload.prompt = promptValue
  }
  // Wait for the catalog this event belongs to, so clicking a button the
  // moment a preset loads cannot beat its own events to the server.
  if (pendingCatalogUpload) {
    await pendingCatalogUpload
  }
  if (!context.sendCommand(payload, label)) {
    return
  }
  if (state === "trigger") {
    const eventLabel =
      currentPreset?.events?.find((item) => item.event_id === eventId)?.label
        ?? currentPreset?.directorEvents?.find((item) => item.event_id === eventId)?.label
        ?? eventId
    applyHealthDelta(getEventHealthDelta(eventId), eventLabel)
  }
  setSessionLocked(true)
}

function attachListeners() {
  presetSelect.addEventListener("change", (e) => {
    if (e.target.value) applyPreset(e.target.value)
  })
  // Digit keys (1-9) trigger player events, letter keys trigger director
  // events (see the "(X)" hotkey suffix rendered in renderEventControls()
  // / eventHotkeyMap), "c" triggers Clear (present on every game, not tied
  // to any preset's catalog) -- this only fires once controls are
  // actually live.
  window.addEventListener("keydown", (event) => {
    // Control itself is a valid hotkey (Crouch) -- only block it as a
    // held modifier for some other key (e.g. Ctrl+C), same as Meta/Alt.
    if (event.metaKey || event.altKey || event.repeat) return
    if (event.ctrlKey && event.key !== "Control") return
    if (eventControls.hidden) return
    const activeTag = document.activeElement?.tagName
    if (activeTag === "INPUT" || activeTag === "TEXTAREA") return
    const key = event.key === " " ? "space" : event.key === "Control" ? "control" : event.key.toLowerCase()
    if (key === "c") {
      sendTextEvent(activeEventId || "clear", "clear")
      return
    }
    if (key === "r") {
      restartButton?.click()
      return
    }
    const eventId = eventHotkeyMap.get(key)
    if (eventId) {
      if (key === "space") event.preventDefault()
      sendTextEvent(eventId, "trigger")
    }
  })
  controlsModeToggle.addEventListener("click", () => setDirectorView(!showingDirectorControls))
  enableDirectorModeButton.addEventListener("click", enableDirectorMode)
  savePresetButton.addEventListener("click", saveCurrentPreset)
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
    forgetPresetImage()
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
    forgetPresetImage()
    firstFrameName.textContent = firstFrameUrlInput.value.trim() ? "Upload Image" : defaultFirstFrameName()
    setFirstFrameStatus(firstFrameUrlInput.value.trim() ? "URL not updated" : "", "pending")
  })
  firstFrameUrlUpdateButton.addEventListener("click", () => void updateFirstFrame())
  promptInput.addEventListener("input", () => { promptEdited = true })
  const promptSubmitButton = sceneCard.querySelector(".promptSubmitButton")
  promptSubmitButton.addEventListener("click", () => {
    const promptText = promptInput.value.trim()
    if (promptText) {
      sendTextEvent("user_prompt", "trigger", promptText)
    }
  })
  addTextEventButton.addEventListener("click", () => {
    textEventDrafts.push(makeTextEventDraft())
    textEventsEdited = true
    renderTextEventEditor()
    context.releaseControls()
  })
  clearEventButton.addEventListener("click", () => sendTextEvent(activeEventId || "clear", "clear"))
  // A reset discards the rollout cache, so the next chunk re-seeds from
  // the session's first frame. Prompt swaps leave the world looking like
  // where it started -- a jet ski in a castle jungle -- and this is what
  // clears that without restarting the server.
  restartButton.addEventListener("click", () => {
    if (context.sendCommand({ type: "reset" }, "restart scene")) {
      resetHealth(currentPreset)
      context.logEvent("scene restarted", { source: "client" })
    }
  })
  const submitLivePrompt = () => {
    const promptText = livePromptInput.value.trim()
    if (promptText) {
      sendTextEvent("user_prompt", "trigger", promptText)
    }
  }
  livePromptSubmitButton.addEventListener("click", submitLivePrompt)
  livePromptInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault()
      submitLivePrompt()
    }
  })
  const submitDirectorPrompt = () => {
    const promptText = directorPromptInput.value.trim()
    if (promptText) {
      sendTextEvent("user_prompt", "trigger", promptText)
    }
  }
  directorPromptSubmitButton.addEventListener("click", submitDirectorPrompt)
  directorPromptInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault()
      submitDirectorPrompt()
    }
  })
  for (const input of [firstFrameUrlInput, promptInput, addTextEventButton, livePromptInput, directorPromptInput]) {
    input.addEventListener("focus", context.releaseControls)
  }
}

export default {
  modelName: "Lingbot",
  stylesheet: new URL("./adapter.css?v=lingbot-video-size-v13", import.meta.url).href,
  controls,

  async mount(sharedContext) {
    context = sharedContext
    try {
      await loadScenePresets()
    } catch (error) {
      context.logEvent(`scene_presets.json unavailable: ${error.message}`, { source: "client", level: "error" })
    }
    loadSavedPresets()
    preview = document.createElement("img")
    preview.className = "firstFramePreview"
    preview.alt = ""
    preview.setAttribute("aria-hidden", "true")
    gameOverOverlay = document.createElement("div")
    gameOverOverlay.className = "gameOverOverlay"
    gameOverOverlay.textContent = "GAME OVER"
    gameOverOverlay.hidden = true
    sceneCard = makeSceneCard()
    eventControls = makeEventControls()
    context.slots.stage.append(preview, gameOverOverlay)
    context.slots.panel.append(sceneCard)
    context.slots.controls.append(eventControls)
    bindElements()
    updatePresetDropdown()
    setFirstFrameInputMode("url")
    attachListeners()
    // ?preset=<slug> (e.g. "water-blaster") shares a direct link to a
    // specific game; falls back to index 0 ("Dragon") otherwise.
    const requestedPreset = new URLSearchParams(window.location.search).get("preset")
    const presetIndex = findPresetIndexBySlug(requestedPreset)
    const defaultPresetIndex = presetIndex >= 0 ? presetIndex : 0
    presetSelect.value = String(defaultPresetIndex)
    applyPreset(defaultPresetIndex)
    try {
      await loadInitialScene()
    } catch (error) {
      context.logEvent(`initial scene unavailable: ${error.message}`, { source: "client", level: "error" })
    }
  },

  async beforeConnect() {
    // resetHealth() otherwise only runs on preset selection -- reconnecting
    // without re-picking a preset left currentHealth carried over from
    // whatever it was at disconnect (e.g. 0, from a prior playthrough),
    // instead of a fresh session actually starting at full health/fuel.
    resetHealth(currentPreset)
    await uploadSessionInput({ includeFirstFrame: true })
  },

  onActionSent() {
    // No updatePreview() here: it refetched the first frame on every key
    // press, cache-busted, and the preview is hidden behind the video by the
    // time any action is sent.
    setSessionLocked(true)
  },

  onControlMessage(payload) {
    if (payload.type === "chunk_done" && Object.prototype.hasOwnProperty.call(payload, "active_event_id")) {
      activeEventId = payload.active_event_id || null
      renderEventControls()
      // Previously the Initial Scene panel only hid once onActionSent()
      // fired (the first movement key or event trigger) -- generation
      // already starts right after connect, so it sat visible/editable
      // over a live session until the player happened to press something.
      setSessionLocked(true)
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

  onInitialScene(scene) {
    // The v1 server acknowledged a text event over the control channel; the
    // v2 endpoint answers the POST with the resulting scene instead, so the
    // active-event highlight is taken from here. Only that is read: the
    // panel's own fields may hold edits the player is still making.
    activeEventId = scene?.active_event_id || null
    renderEventControls()
  },

  onVideoVisibilityChanged(visible) {
    // Video becoming visible is what "a session is live" looks like on v2.
    // The v1 server announced it with a "chunk_done" control message, which
    // v2 never sends -- so the Initial Scene panel used to sit visible and
    // editable over a running session until the player happened to press a
    // key, which is what finally triggered onActionSent().
    setSessionLocked(Boolean(visible))
    updatePreview()
  },

  onDisconnect() {
    setSessionLocked(false)
    updatePreview()
  },
}
