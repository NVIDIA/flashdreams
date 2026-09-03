<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Self-Forcing T2V application

```bash
uv run --package flashdreams-self-forcing flashdreams-run-v2 \
  t2v-self-forcing-wan2.1-t2v-1.3b --output-path clip.mp4 -- \
  --prompt "A cat surfing" --total-blocks 7 --no-compile
```

TAEHV and long-rollout variants use
`t2v-self-forcing-wan2.1-t2v-1.3b-taehv` and
`t2v-self-forcing-wan2.1-t2v-1.3b-sink5-window7-rerope`.

See the [shared T2V guide](../../../../apps/t2v/README.md) for controls, common arguments, defaults, and tests.
