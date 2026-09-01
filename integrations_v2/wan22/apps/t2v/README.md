<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Wan 2.2 TI2V application

Launch the single-block Wan 2.2 application with a prompt and first frame:

```bash
uv run --package flashdreams-wan22 flashdreams-run-v2 \
  t2v-wan22-ti2v-5b --output-path clip.mp4 -- \
  --prompt "A cinematic ocean wave at sunset" \
  --image-path /path/to/first-frame.png --no-compile
```

See the [shared T2V guide](../../../../apps/t2v/README.md) for controls, common arguments, defaults, and tests.
