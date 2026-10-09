---
title: 'Models'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

FlashDreams runs a growing family of world and video models (text-to-video, image-to-video, camera-controlled, ...).
Model overviews identify currently supported models.

## Running a model

All models run via our `flashdreams-run-v2` CLI command
```bash
cd flashdreams

# Almost all models download their own model weights from Hugging Face, please set `HF_TOKEN` to avoid rate limiting.
export HF_TOKEN=<your-hugging-face-token>

uv run flashdreams-run-v2 <Demo Preset> <System Arguments> -- <Demo Arguments>
```

- `<Demo Preset>` is a preset that contains the **model/demo pair** to run. Refer to the model-overviews below to find a preset.

- `<System Arguments>` are arguments that apply to all `flashdreams-run-v2` demos, reference is here: [System arguments](../documentation/cli.md#system-arguments).

- `<Demo Arguments>` are arguments that apply to a particular demo. Refer to a particular [demo overview](../demos/index.md).


## Models - Streaming and autoregressive

These models advance a video or world incrementally for responsive, stateful generation.

<div class="fd-card-grid fd-card-grid-three fd-overview-grid">
  <div class="fd-card fd-model-card"><a class="fd-model-card-link" href="omnidreams/"><img class="fd-card-preview" src="../_static/model_clips/omnidreams/omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-239560dc-33d1-11ef-9720-00044bcbccac-pip.avif" alt="" /><span class="fd-card-title">NVIDIA OmniDreams</span><span>HDMap-conditioned streaming world model for autonomous driving.</span></a></div>
  <div class="fd-card fd-model-card"><a class="fd-model-card-link" href="self_forcing/"><img class="fd-card-preview" src="../_static/model_clips/self_forcing/self-forcing-wan2.1-t2v-1.3b-flash-1.avif" alt="" /><span class="fd-card-title">Self-Forcing</span><span>Autoregressive text-to-video based on Wan 2.1.</span></a></div>
  <div class="fd-card fd-model-card"><a class="fd-model-card-link" href="causal_forcing/"><img class="fd-card-preview" src="../_static/model_clips/causal_forcing/causal-forcing-wan2.1-t2v-1.3b-framewise.avif" alt="" /><span class="fd-card-title">Causal-Forcing</span><span>Real-time causal text- and image-to-video generation.</span></a></div>
  <div class="fd-card fd-model-card"><a class="fd-model-card-link" href="causal_wan22/"><img class="fd-card-preview" src="../_static/model_clips/causal_wan22/fastvideo-causal-wan2.2-t2v-14b-1.avif" alt="" /><span class="fd-card-title">Causal Wan 2.2</span><span>Autoregressive video generation based on Wan 2.2.</span></a></div>
  <div class="fd-card fd-model-card"><a class="fd-model-card-link" href="lingbot_world/"><img class="fd-card-preview" src="../_static/model_clips/lingbot_world/lingbot-world-fast-01.avif" alt="" /><span class="fd-card-title">LingBot-World</span><span>Camera-controllable streaming image-to-video world model.</span></a></div>
  <div class="fd-card fd-model-card"><a class="fd-model-card-link" href="waypoint/"><img class="fd-card-preview" src="../_static/model_clips/waypoint/waypoint-1.5.avif" alt="" /><span class="fd-card-title">Waypoint 1.5</span><span>Interactive world model controlled by keyboard and mouse.</span></a></div>
  <div class="fd-card fd-model-card"><a class="fd-model-card-link" href="hy_worldplay/"><img class="fd-card-preview" src="../_static/model_clips/hy_worldplay/hy-worldplay-wan-i2v-5b-2.avif" alt="" /><span class="fd-card-title">HY-WorldPlay</span><span>Action- and camera-controllable image-to-video world model.</span></a></div>
  <div class="fd-card fd-model-card"><a class="fd-model-card-link" href="sana_wm_streaming/"><img class="fd-card-preview" src="../_static/model_clips/sana_wm/sana-wm-streaming.avif" alt=""><span class="fd-card-title">SANA-WM</span><span>Camera-controlled world model with streaming and bidirectional variants.</span></a></div>
</div>

## Models - Bidirectional generation

These models render a complete clip in one pass and provide full-sequence reference implementations.

<div class="fd-card-grid fd-card-grid-three fd-overview-grid">
  <div class="fd-card fd-model-card"><a class="fd-model-card-link" href="sana_wm_streaming/"><img class="fd-card-preview" src="../_static/model_clips/sana_wm/sana-wm-bidirectional.avif" alt=""><span class="fd-card-title">SANA-WM</span><span>Full-sequence camera-controlled image-to-video generation.</span></a></div>
  <div class="fd-card fd-model-card"><a class="fd-model-card-link" href="wan21/"><img class="fd-card-preview" src="../_static/model_clips/wan21/wan21-t2v-1.3b-480p.avif" alt="" /><span class="fd-card-title">Wan 2.1</span><span>Text-to-video and image-to-video generation.</span></a></div>
  <div class="fd-card fd-model-card"><a class="fd-model-card-link" href="wan22/"><img class="fd-card-preview" src="../_static/model_clips/wan22/wan22-ti2v-5b.avif" alt="" /><span class="fd-card-title">Wan 2.2 TI2V-5B</span><span>Text-and-image-to-video generation in one full-clip rollout.</span></a></div>
  <div class="fd-card fd-model-card"><a class="fd-model-card-link" href="cosmos_predict2/"><img class="fd-card-preview" src="../_static/model_clips/cosmos_predict2/cosmos2-t2v-2b-720p.avif" alt="" /><span class="fd-card-title">Cosmos-Predict2.5</span><span>Text2World, Image2World, and Video2World generation.</span></a></div>
</div>

## Models - Upscaling and restoration

These models enhance an existing video stream rather than generating a scene from scratch.

<div class="fd-card-grid fd-card-grid-three fd-overview-grid">
  <div class="fd-card fd-model-card"><a class="fd-model-card-link" href="flashvsr/"><img class="fd-card-preview" src="../_static/model_clips/flashvsr/flashvsr-v1.1-sparse-ratio-2.0.avif" alt="" /><span class="fd-card-title">FlashVSR</span><span>One-step streaming video super-resolution.</span></a></div>
  <div class="fd-card fd-model-card"><a class="fd-model-card-link" href="swiftvr/"><img class="fd-card-preview" src="../_static/model_clips/swiftvr/swiftvr-2x.avif" alt="" /><span class="fd-card-title">SwiftVR</span><span>Real-time streaming video restoration with 2× and 4× presets.</span></a></div>
</div>

## Adding your own model

Follow
[Create a model](../documentation/inferencing_api/guides/create_model.md) for
the model package, then
[Integrate a model with a demo](../documentation/demo_api/guides/integrate_model.md)
to register a runnable application.

## Related

- Follow the [/quickstart/index](../quickstart/index.md) for the shortest path to
  running a model on your own hardware.
- The [API guides](../documentation/index.md) cover the architecture behind the
  models you can run today.
- [/community/index](../community/index.md) lists the channels to use if a process on
  this page does not run on your hardware.
- Browse the source on GitHub at [NVIDIA/flashdreams](https://github.com/NVIDIA/flashdreams).
