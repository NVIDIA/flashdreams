<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Causal-Forcing T2V application

```bash
uv run --package flashdreams-causal-forcing flashdreams-run-v2 \
  t2v-causal-forcing-wan2.1-t2v-1.3b-chunkwise --output-path clip.mp4 -- \
  --prompt "A cat surfing" --total-blocks 7 --no-compile
```

For the framewise config, replace the application name with
`t2v-causal-forcing-wan2.1-t2v-1.3b-framewise`.

See the [shared T2V guide](../../../../apps/t2v/README.md) for controls, common arguments, defaults, and tests.
