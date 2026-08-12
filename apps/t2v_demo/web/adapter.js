// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Model metadata only; shared WebRTC UI renders prompt/video controls. */
export default {
  modelName: "Text-to-Video",
  async mount(context) {
    const response = await fetch("/api/t2v/config")
    if (!response.ok) return
    const config = await response.json()
    const selected = config.backends?.find((backend) => backend.key === config.selected_backend)
    context.setModelName(selected?.label || config.selected_backend || "Text-to-Video")
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
