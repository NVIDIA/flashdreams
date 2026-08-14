/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

export default {
  modelName: "Video to Video",
  autoConnect: false,
  stylesheet: "/model-static/adapter.css?v=v2v-ui-v2",
  async mount(context) {
    const panel = document.createElement("section")
    panel.className = "v2vPanel"
    panel.innerHTML = `
      <h2>Source video</h2>
      <input id="v2vUpload" type="file" accept="video/*">
      <p id="v2vStatus">Choose a video to begin.</p>
      <video id="v2vSource" controls muted playsinline></video>
      <a id="v2vDownload" class="v2vDownload" href="/api/v2v/download" download hidden>Download MP4 + metadata ZIP</a>`
    context.slots.panel.replaceChildren(panel)
    const upload = panel.querySelector("#v2vUpload")
    const source = panel.querySelector("#v2vSource")
    const status = panel.querySelector("#v2vStatus")
    const download = panel.querySelector("#v2vDownload")

    upload.addEventListener("change", async () => {
      const file = upload.files?.[0]
      if (!file) return
      source.src = URL.createObjectURL(file)
      status.textContent = "Uploading source video…"
      const form = new FormData()
      form.append("video", file, file.name)
      const response = await fetch("/api/v2v/upload", { method: "POST", body: form })
      if (!response.ok) throw new Error(await response.text())
      status.textContent = "Upload accepted. Model is loading; the generated stream will start shortly."
      context.logEvent(`uploaded ${file.name}`, { source: "client" })
      download.hidden = true
      await context.connectSession()
    })
  },
  onControlMessage(payload) {
    if (payload?.type !== "chunk_done" || payload.rollout_complete !== true) return false
    document.querySelector("#v2vStatus").textContent = "Generation complete. Upload another video or download the result."
    document.querySelector("#v2vDownload").hidden = false
    return false
  },
}
