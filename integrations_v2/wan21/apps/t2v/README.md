<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Wan 2.1 T2V application

```bash
uv run --package flashdreams-wan21 flashdreams-run-v2 \
  t2v-wan21-t2v-1.3b-480p --output-path clip.mp4 -- \
  --prompt "A cat surfing" --no-compile
```

See the [shared T2V guide](../../../../apps/t2v/README.md) for controls, common arguments, defaults, and tests.
