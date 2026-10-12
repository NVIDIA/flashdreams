---
title: 'FlashDreams'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

<div class="fd-hero">
  <div class="fd-hero-copy">
    <h1 class="fd-hero-title">FlashDreams</h1>
    <p class="fd-hero-lede">FlashDreams is an inference and serving runtime for turning
    autoregressive video and world models into live, controllable simulations. It runs
    the model in a continuous loop, carrying state forward and streaming frames while
    new actions or sensor inputs change what happens next, whether the application is a
    game world, an autonomous-vehicle simulator, robotic policy testing, or a virtual
    training environment.</p>
    <div class="fd-cta-row">
      <a class="fd-button fd-button-primary" href="quickstart/">Get Started!</a>
      <a class="fd-button" href="https://github.com/NVIDIA/flashdreams">GitHub</a>
      <a class="fd-button" href="community/">Contribute</a>
    </div>
  </div>
  <div class="fd-hero-visual">
    <img alt="FlashDreams quick intro animation" src="_static/promo/flashdreams-promo.avif" />
  </div>
</div>

<div class="transparent-section">
  <h2>Why FlashDreams?</h2>

  <div class="fd-split">
    <div class="fd-split-visual">
      <img alt="Offline one-shot video inference compared with online autoregressive world-model serving." src="_static/diagrams/compare-offline-online-video-model-v2.jpg" />
    </div>
    <div class="fd-split-copy">
      <p>A world model learns to generate and evolve an environment over time. In practice
      that usually means video, but the same idea extends to actions, state, audio, sensor
      input, and control signals. Serving one means keeping a session alive while input,
      model state, GPU inference, and output advance together, rather than producing a
      single static clip, which is what makes interactive simulation, robotics, autonomy,
      and game-like experiences possible.</p>
    </div>
  </div>

  FlashDreams is built for that real-time case: a closed-loop world-model
  demo, a driving simulator, an interactive scene rollout. Generating
  high-quality video is not enough on its own. The runtime has to keep an
  interactive session responsive while the model continues to advance the
  world. That comes down to four things:

  <div class="fd-card-grid fd-card-grid-four">
    <div class="fd-card">
      <div class="fd-card-title">Low latency</div>
      <p>Keep the interaction responsive when controls, sensors, or user input change.</p>
    </div>
    <div class="fd-card">
      <div class="fd-card-title">High throughput</div>
      <p>Keep the GPU busy across autoregressive steps and multi-GPU execution.</p>
    </div>
    <div class="fd-card">
      <div class="fd-card-title">Steady streaming generation</div>
      <p>Stream frames or chunks at a steady pace while the session continues.</p>
    </div>
    <div class="fd-card">
      <div class="fd-card-title">World-state evolution</div>
      <p>Carry rolling state forward so the generated world evolves across steps.</p>
    </div>
  </div>
</div>

<div class="grey-section">
  <h2>Performance</h2>

  Each tile shows the speedup over a separate existing implementation of
  the same model. Both runs use the same weights on the same GPU, so the
  gain comes from FlashDreams' runtime alone. Each tile links to the
  profiling chart on its model page.

  <div class="fd-card-grid fd-card-grid-four">
    <a class="fd-card fd-stat-card" href="models/self_forcing/#performance-outdated">
      <span class="fd-stat-value">2.12×</span>
      <span class="fd-stat-label">Self-Forcing speedup</span>
    </a>
    <a class="fd-card fd-stat-card" href="models/lingbot_world/#performance-outdated">
      <span class="fd-stat-value">3.10×</span>
      <span class="fd-stat-label">LingBot-World speedup</span>
    </a>
    <a class="fd-card fd-stat-card" href="models/wan21/#performance-outdated">
      <span class="fd-stat-value">1.40×</span>
      <span class="fd-stat-label">Wan2.1 speedup</span>
    </a>
    <a class="fd-card fd-stat-card" href="models/flashvsr/#performance-outdated">
      <span class="fd-stat-value">1.42×</span>
      <span class="fd-stat-label">FlashVSR speedup</span>
    </a>
  </div>
</div>

<div class="transparent-section" markdown="1">
  <h2>Try FlashDreams!</h2>

  FlashDreams brings best-in-class per-step latency to interactive
  autoregressive video and world models: multiple integrated models across
  streaming and bidirectional methods, multi-GPU execution, and CLI entry
  points for runner presets, demos, and v2 applications.

  The [Get Started guide](quickstart/index.md) walks from a fresh
  checkout to running OmniDreams, an interactive driving world-model demo
  built on FlashDreams.
</div>

<div class="grey-section">
  <h2>Supported Models</h2>

  Streaming and autoregressive model implementations emit per-step output with
  sub-second latency once warm; bidirectional model implementations are kept as
  full-block parity references. Each model page carries the canonical
  invocation, the checkpoint source, and the per-implementation knobs.

  <div class="fd-card-grid fd-card-grid-three">
    <a class="fd-card fd-model-card" href="models/omnidreams/">
      <span class="fd-card-title">OmniDreams</span>
      <span>Interactive world simulator for autonomous vehicles.</span>
    </a>
    <a class="fd-card fd-model-card" href="models/self_forcing/">
      <span class="fd-card-title">Self-Forcing</span>
      <span>Autoregressive text-to-video based on Wan 2.1.</span>
    </a>
    <a class="fd-card fd-model-card" href="models/causal_forcing/">
      <span class="fd-card-title">Causal-Forcing</span>
      <span>Autoregressive text/image-to-video based on Wan 2.1.</span>
    </a>
    <a class="fd-card fd-model-card" href="models/causal_wan22/">
      <span class="fd-card-title">Causal Wan 2.2</span>
      <span>Autoregressive text-to-video based on Wan 2.2 from FastVideo.</span>
    </a>
    <a class="fd-card fd-model-card" href="models/lingbot_world/">
      <span class="fd-card-title">LingBot-World</span>
      <span>Camera-controllable image-to-video world model.</span>
    </a>
    <a class="fd-card fd-model-card" href="models/waypoint/">
      <span class="fd-card-title">Waypoint 1.5</span>
      <span>Interactive image-established world model controlled by keyboard and mouse.</span>
    </a>
    <a class="fd-card fd-model-card" href="models/sana_wm_streaming/">
      <span class="fd-card-title">SANA-WM</span>
      <span>Camera-controlled world model with streaming and bidirectional variants.</span>
    </a>
    <a class="fd-card fd-model-card" href="models/flashvsr/">
      <span class="fd-card-title">FlashVSR</span>
      <span>Streaming video super-resolution.</span>
    </a>
    <a class="fd-card fd-model-card" href="models/swiftvr/">
      <span class="fd-card-title">SwiftVR</span>
      <span>Real-time one-step streaming video restoration with 2x and 4x presets.</span>
    </a>
    <a class="fd-card fd-model-card" href="models/wan21/">
      <span class="fd-card-title">Wan 2.1 (bidirectional)</span>
      <span>Bidirectional video generation model that supports both text-to-video and image-to-video.</span>
    </a>
    <a class="fd-card fd-model-card" href="models/wan22/">
      <span class="fd-card-title">Wan 2.2 TI2V-5B (bidirectional)</span>
      <span>Bidirectional text-and-image-to-video generation in one full-clip rollout.</span>
    </a>
    <a class="fd-card fd-model-card" href="models/cosmos_predict2/">
      <span class="fd-card-title">Cosmos-Predict2.5 (bidirectional)</span>
      <span>Bidirectional Cosmos-Predict2 reference implementations (T2V / I2V, 2B).</span>
    </a>
  </div>
</div>
