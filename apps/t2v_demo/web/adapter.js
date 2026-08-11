// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/** Model metadata only; shared WebRTC UI renders prompt/video controls. */
export default {
  modelName: "Text-to-Video",
  promptGeneration: {
    endpoint: "/api/t2v/prompt",
    label: "Describe the video",
    placeholder: "A cinematic drone shot over snowy mountains at sunrise",
    generateLabel: "Generate video",
  },
}
