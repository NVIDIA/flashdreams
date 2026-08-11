// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

let promptInput
let generateButton
let downloadButton
let recorder
let chunks = []

async function savePrompt(context) {
  const response = await fetch("/api/t2v/prompt", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ prompt: promptInput.value }),
  })
  if (!response.ok) throw new Error(await response.text())
  context.logEvent("prompt queued for this generation", { source: "client" })
}

export default {
  modelName: "Text-to-Video",
  controls: [{ label: "Generate", keys: [{ key: "g", label: "Generate" }] }],
  async mount(context) {
    const panel = document.createElement("section")
    panel.className = "t2vPanel"
    panel.innerHTML = `
      <label>Prompt<textarea rows="5" placeholder="Describe the video you want to generate"></textarea></label>
      <p>Press Generate to stream chunks to the player. The browser records the live stream for download.</p>
      <div><button type="button" class="t2vGenerate">Generate</button><button type="button" class="t2vDownload" disabled>Download MP4/WebM</button></div>`
    context.slots.panel.append(panel)
    promptInput = panel.querySelector("textarea")
    generateButton = panel.querySelector(".t2vGenerate")
    downloadButton = panel.querySelector(".t2vDownload")
    generateButton.addEventListener("click", async () => {
      try {
        await savePrompt(context)
        context.logEvent("click the G control to begin the shared WebRTC session", { source: "client" })
      } catch (error) { context.logEvent(error.message, { source: "client", level: "error" }) }
    })
    downloadButton.addEventListener("click", () => {
      const blob = new Blob(chunks, { type: recorder?.mimeType || "video/webm" })
      const anchor = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "flashdreams-t2v.webm" })
      anchor.click(); URL.revokeObjectURL(anchor.href)
    })
    const video = document.getElementById("remoteVideo")
    video.addEventListener("play", () => {
      if (recorder || !video.srcObject || !window.MediaRecorder) return
      chunks = []
      recorder = new MediaRecorder(video.srcObject)
      recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data) }
      recorder.onstop = () => { downloadButton.disabled = chunks.length === 0 }
      recorder.start(1000)
    })
  },
  onDisconnect() { if (recorder?.state === "recording") recorder.stop(); recorder = null },
}
