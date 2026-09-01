<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FastVideo CausalWan 2.2 T2V application

```bash
uv run --package flashdreams-fastvideo-causal-wan22 flashdreams-run-v2 \
  t2v-fastvideo-causal-wan2.2-t2v-14b --output-path clip.mp4 -- \
  --prompt "A cat surfing" --total-blocks 7 --no-compile
```

See the [shared T2V guide](../../../../apps/t2v/README.md) for controls, common arguments, defaults, and tests.
