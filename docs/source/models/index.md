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

- `<Demo Preset>` is a preset that contains a pair of **model to run with a demo**. Refer to the model-overviews below to find a preset.

- `<System Arguments>` are arguments that apply to all `flashdreams-run-v2` demos, reference is here: [System arguments](../documentation/cli.md#system-arguments).

- `<Demo Arguments>` are arguments that apply to a particular demo. Refer to a particular [demo overview](../demos/index.md).



## Available models

The models come in three flavors:

- Streaming and autoregressive generation: build a video step by step and stay fast once warmed up, aiming for sub-second latency per step

- Bidirectional generation: produce a clip in a single pass and serve as the quality reference for their streaming counterparts

- Super-resolution: upscale existing frames in chunks, so their latency scales with output resolution rather than step count

<hr>

### [OmniDreams](omnidreams.md)

<video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
  <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/omnidreams/omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-239560dc-33d1-11ef-9720-00044bcbccac-pip.mp4" type="video/mp4">
</video>

<figcaption>
  Interactive world simulator for autonomous vehicles.
</figcaption>

<hr>

### [Self-Forcing](self_forcing.md)

<video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
  <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/self_forcing/self-forcing-wan2.1-t2v-1.3b-flash_1.mp4" type="video/mp4">
</video>

<figcaption>
  Autoregressive text-to-video based on Wan 2.1.
</figcaption>

<hr>

### [Causal-Forcing](causal_forcing.md)

<video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
  <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/causal_forcing/causal-forcing-wan2.1-t2v-1.3b-framewise.mp4" type="video/mp4">
</video>

<figcaption>
  Autoregressive text/image-to-video based on Wan 2.1.
</figcaption>

<hr>

### [Causal Wan 2.2](causal_wan22.md)

<video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
  <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/causal_wan22/fastvideo-causal-wan2.2-t2v-14b_1.mp4" type="video/mp4">
</video>

<figcaption>
  Autoregressive text-to-video based on Wan 2.2 from FastVideo.
</figcaption>

<hr>

### [LingBot-World](lingbot_world.md)

<div class="fd-card-video-wrap">
  <video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
    <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/lingbot_world/lingbot-world-fast-01.mp4" type="video/mp4">
  </video>
</div>

<figcaption>
  Camera-controllable image-to-video world model.
</figcaption>

<hr>

### [Waypoint 1.5](waypoint.md)

<video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
  <source src="https://huggingface.co/Overworld/Waypoint-1.5-1B/resolve/main/assets/wp_1.5.mp4" type="video/mp4">
</video>

<figcaption>
  Interactive image-established world model controlled by keyboard and mouse.
</figcaption>

<hr>

### [HY-WorldPlay](hy_worldplay.md)

<video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
  <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/hy_worldplay/hy-worldplay-wan-i2v-5b-2.mp4" type="video/mp4">
</video>

<figcaption>
  Action- and camera-controllable image-to-video world model.
</figcaption>

<hr>

### [SANA-WM](sana_wm_streaming.md)

<img alt="SANA-WM streaming FlashDreams sample clip." src="../_static/model_clips/sana_wm/sana-wm-streaming.avif" />
<figcaption>
  Bidirectional & streaming video generation capable world model that is camera-controlled.
</figcaption>

<hr>

### [Wan 2.1](wan21.md)

<div class="model-video-card" style="width: 100%; margin: 10px auto 14px;">
  <video class="model-video-player" autoplay muted loop playsinline preload="metadata">
    <source src="../_static/model_clips/wan21/wan21-t2v-1.3b-480p.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

<figcaption>
  Bidirectional video generation model that supports both text-to-video and image-to-video.
</figcaption>

<hr>

### [Wan 2.2 TI2V-5B](wan22.md)

<video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
  <source src="../_static/model_clips/wan22/wan22-ti2v-5b.mp4" type="video/mp4">
</video>

<figcaption>
  Bidirectional text-and-image-to-video generation in one full-clip rollout.
</figcaption>

<hr>

### [Cosmos-Predict2.5](cosmos_predict2.md)

<video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
  <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/cosmos_predict2/cosmos2-t2v-2b-720p.mp4" type="video/mp4">
</video>

<figcaption>
  Bidirectional Cosmos-Predict2 reference implementations (T2V / I2V, 2B).
</figcaption>

<hr>

### [FlashVSR](flashvsr.md)

<video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
  <source src="https://research.nvidia.com/labs/sil/projects/flashdreams/assets/flashvsr/flashvsr-v1.1-sparse-ratio-2.0.mp4" type="video/mp4">
</video>

<figcaption>
  Streaming video super-resolution.
</figcaption>

<hr>

### [SwiftVR](swiftvr.md)

<video class="fd-card-video" autoplay muted loop playsinline preload="metadata">
  <source src="../_static/model_clips/swiftvr/swiftvr-2x.mp4" type="video/mp4">
</video>

<figcaption>
  Realtime (an run via one step) capable super-resolution with 2x and 4x presets.
</figcaption>

<hr>

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
