<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LingBot-VA V2 application

This package adapts the LingBot-VA Robotwin I2AV model to the FlashDreams V2
application/session/model-loop API. It registers the slug
`lingbot-va-robotwin-i2av` in `flashdreams.applications_v2`.

```bash
uv run --project integrations_v2/lingbot_va flashdreams-run-v2 \
    lingbot-va-robotwin-i2av \
    --mode mp4 \
    --output-path outputs/lingbot_va/demo.mp4 \
    --stats-path outputs/lingbot_va/metrics.json \
    --tensor-artifact-dir outputs/lingbot_va \
    -- \
    --checkpoint-root robbyant/lingbot-va-posttrain-robotwin \
    --checkpoint-revision 8c9dea8abbc5c91cc9e18bc3264b8915083bbe70 \
    --input-image-dir /path/to/robotwin-images \
    --num-chunks 10
```

The application describes its natural session before initialization: TCHW,
256x320, 10 FPS, blocking backpressure, present-only-new behavior, and an
`actions[step, channel]` tensor artifact. Runtime backpressure and presentation
overrides are preserved; fixed model output properties are validated. Model
loading is lazy on the model thread, and the finite loop emits one complete
rollout before reporting finished.

`--input-image-dir` is required because no camera images are bundled. Use
`-- --help` after the application slug for checkpoint, input, compilation,
offload, seed, guidance, inference-step, and scheduler-shift overrides. The
[model package README](../../integrations/lingbot_va/README.md) documents input
provenance, checkpoint modes, camera/action contracts, lifecycle, opt-in GPU
tests, measured parity evidence, and limitations.
