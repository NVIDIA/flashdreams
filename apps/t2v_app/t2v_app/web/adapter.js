// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** T2V metadata and finite-generation controls for the shared WebRTC UI. */
export default {
  modelName: "Text-to-Video",
  async mount(context) {
    const response = await fetch("/api/t2v/config")
    if (!response.ok) return
    const config = await response.json()
    context.setModelName(config.model_id || "Text-to-Video")
    const panel = document.querySelector(".promptGenerationPanel")
    const prompt = panel?.querySelector("textarea")
    const duration = panel?.querySelector(".promptDurationInput")
    if (prompt && typeof config.default_prompt === "string") {
      prompt.value = config.default_prompt
    }
    if (duration && Number.isFinite(Number(config.default_duration_s))) {
      duration.value = String(config.default_duration_s)
    }
  },
  promptGeneration: {
    endpoint: "/api/t2v/prompt",
    label: "Describe the video",
    placeholder: "A cinematic drone shot over snowy mountains at sunrise",
    generateLabel: "Generate video",
    downloadEndpoint: "/api/t2v/download",
    playbackEndpoint: "/api/t2v/playback",
    hideControls: true,
  },
}
