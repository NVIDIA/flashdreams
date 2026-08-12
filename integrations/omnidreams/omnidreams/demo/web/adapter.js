// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

export default {
  modelName: "OmniDreams",
  stylesheet: "/model-static/adapter.css?v=model-ui-v2",
  controls: [
    {
      label: "Drive / Turn",
      keys: [
        { key: "w", label: "Forward" },
        { key: "a", label: "Turn left" },
        { key: "s", label: "Backward" },
        { key: "d", label: "Turn right" },
      ],
    },
  ],
}
