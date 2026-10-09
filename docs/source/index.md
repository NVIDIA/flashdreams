---
title: 'FlashDreams'
hide:
  - navigation
  - toc
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

<div class="fd-hero">
  <div class="fd-hero-copy">
    <h1 class="fd-hero-title">FlashDreams</h1>
    <p class="fd-hero-lede">
    FlashDreams is an inference and serving runtime for turning
    world models into live, controllable simulations. FlashDreams runs models in a continuous
    loop, carrying state forward and streaming frames while new actions or sensor inputs change
    what happens next step.
    <br>
    <br>
    Whether the application is a game world, an autonomous-vehicle simulator, robotic policy testing, or a virtual training environment, FlashDreams is the runtime to make it possible.
  </p>
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
  <h2>Explore FlashDreams</h2>
  <p>Choose a section to get started or browse the project.</p>

  <div class="fd-card-grid fd-card-grid-three fd-overview-grid">
    <a class="fd-card fd-model-card" href="quickstart/"><span class="fd-card-title">Get Started</span><span>Install FlashDreams and run your first model.</span></a>
    <a class="fd-card fd-model-card" href="models/"><span class="fd-card-title">Models</span><span>Browse default supported models.</span></a>
    <a class="fd-card fd-model-card" href="demos/"><span class="fd-card-title">Demos</span><span>Explore our game-like experiences and model interfaces.</span></a>
    <a class="fd-card fd-model-card" href="documentation/"><span class="fd-card-title">Documentation</span><span>Checkout our CLI, Demo API, Inferencing API, tooling, or related guides.</span></a>
    <a class="fd-card fd-model-card" href="community/"><span class="fd-card-title">Community</span><span>Find contribution documentation and support channels.</span></a>
  </div>
</div>

<div class="transparent-section">
  <h2>Why FlashDreams?</h2>

  <div class="grey-section fd-why-section">
    <h2>Serving World Models is Hard</h2>
    <p class="fd-why-lede">Building a <strong>static clip generator is not enough</strong> for world-model serving.</p>

    <img class="fd-why-comparison" alt="Offline video generation runs once from a prompt to a finished clip. Online world-model serving loops live controls through streaming encode, state update, inference, and decode for low-latency output while carrying world state forward." src="_static/diagrams/compare-offline-online-video-model-v2-transparent.webp" />

    <div class="fd-why-row">
      <div class="fd-why-copy">
        <h4>Worlds evolve over time</h4>
        <p>World models are taught to generate and evolve an environment over time. They support robotics-policy testing, autonomous-vehicle simulation, interactive worlds, and other applications.</p>
      </div>
      <img class="fd-why-supporting-image" alt="World models support autonomous systems using sensor input, robotics policy testing, and interactive worlds with live controls." data-generated-by="GPT-5.6 Sol - High" src="_static/diagrams/world-model-use-cases-simple.webp" />
    </div>

    <div class="fd-why-row">
      <div class="fd-why-copy">
        <h4>Serving has two parts</h4>
        <ol class="fd-why-list">
          <li>Inference of a video model that consumes inputs such as clock time and sensors while evolving world state.</li>
          <li>A separate application that coordinates inputs, model state, GPU inference, and output.</li>
        </ol>
      </div>
      <img class="fd-why-supporting-image" alt="World-model serving has two parts. Model Inference consumes time and sensors to evolve world state. In Application Coordination, controls update world state, a connected upper path feeds world state back to the controls, the GPU produces stacked frames, those frames stream to the monitor, and the monitor also feeds back to the controls." data-generated-by="GPT-5.6 Sol - High" src="_static/diagrams/world-model-serving-two-parts-v5.webp" />
    </div>

    <div class="fd-why-row">
      <div class="fd-why-copy">
        <h4>FlashDreams was built for both parts:</h4>
        <ol class="fd-why-list">
          <li>The <a href="documentation/inferencing_api/">Inferencing API</a> helps model designers implement stateful world-model inference.</li>
          <li>The <a href="documentation/demo_api/">Demo API</a> helps application developers build experiences around the constraints of world-model serving.</li>
        </ol>
      </div>
      <img class="fd-why-supporting-image" alt="Model Inference maintains state and cache for GPU inference. The Demo API coordinates controls, sensors, and presentation. Actions and state flow to Model Inference; streamed frames flow to the Demo API." data-generated-by="GPT-5.6 Sol - High" src="_static/diagrams/flashdreams-model-inference-demo-api.webp" />
    </div>
  </div>

  <div class="grey-section">
    <h2>Performance</h2>

    Each tile shows speedup over an existing implementation of
    the same model. Both run the same weights on the same GPU, so the
    gain comes from the FlashDreams runtime alone.

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
</div>

<br><br>

<div class="transparent-section" markdown="1">
  <h2>Try FlashDreams!</h2>

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

  <div class="fd-card-grid fd-card-grid-three fd-overview-grid">
    <a class="fd-card fd-model-card" href="models/omnidreams/"><img class="fd-card-preview" src="_static/model_clips/omnidreams/omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-239560dc-33d1-11ef-9720-00044bcbccac-pip.avif" alt="" /><span class="fd-card-title">OmniDreams</span><span>Interactive world simulator for autonomous vehicles.</span></a>
    <a class="fd-card fd-model-card" href="models/self_forcing/"><img class="fd-card-preview" src="_static/model_clips/self_forcing/self-forcing-wan2.1-t2v-1.3b-flash-1.avif" alt="" /><span class="fd-card-title">Self-Forcing</span><span>Autoregressive text-to-video based on Wan 2.1.</span></a>
    <a class="fd-card fd-model-card" href="models/causal_forcing/"><img class="fd-card-preview" src="_static/model_clips/causal_forcing/causal-forcing-wan2.1-t2v-1.3b-framewise.avif" alt="" /><span class="fd-card-title">Causal-Forcing</span><span>Autoregressive text/image-to-video based on Wan 2.1.</span></a>
    <a class="fd-card fd-model-card" href="models/causal_wan22/"><img class="fd-card-preview" src="_static/model_clips/causal_wan22/fastvideo-causal-wan2.2-t2v-14b-1.avif" alt="" /><span class="fd-card-title">Causal Wan 2.2</span><span>Autoregressive text-to-video based on Wan 2.2 from FastVideo.</span></a>
    <a class="fd-card fd-model-card" href="models/lingbot_world/"><img class="fd-card-preview" src="_static/model_clips/lingbot_world/lingbot-world-fast-01.avif" alt="" /><span class="fd-card-title">LingBot-World</span><span>Camera-controllable image-to-video world model.</span></a>
    <a class="fd-card fd-model-card" href="models/waypoint/"><img class="fd-card-preview" src="_static/model_clips/waypoint/waypoint-1.5.avif" alt="" /><span class="fd-card-title">Waypoint 1.5</span><span>Interactive world model controlled by keyboard and mouse.</span></a>
    <a class="fd-card fd-model-card" href="models/hy_worldplay/"><img class="fd-card-preview" src="_static/model_clips/hy_worldplay/hy-worldplay-wan-i2v-5b-2.avif" alt="" /><span class="fd-card-title">HY-WorldPlay</span><span>Action- and camera-controllable image-to-video world model.</span></a>
    <a class="fd-card fd-model-card" href="models/sana_wm_streaming/"><img class="fd-card-preview" src="_static/model_clips/sana_wm/sana-wm-streaming.avif" alt=""><span class="fd-card-title">SANA-WM</span><span>Camera-controlled world model with streaming and bidirectional variants.</span></a>
    <a class="fd-card fd-model-card" href="models/flashvsr/"><img class="fd-card-preview" src="_static/model_clips/flashvsr/flashvsr-v1.1-sparse-ratio-2.0.avif" alt="" /><span class="fd-card-title">FlashVSR</span><span>Streaming video super-resolution.</span></a>
    <a class="fd-card fd-model-card" href="models/swiftvr/"><img class="fd-card-preview" src="_static/model_clips/swiftvr/swiftvr-2x.avif" alt="" /><span class="fd-card-title">SwiftVR</span><span>Real-time streaming video restoration with 2× and 4× presets.</span></a>
    <a class="fd-card fd-model-card" href="models/wan21/"><img class="fd-card-preview" src="_static/model_clips/wan21/wan21-t2v-1.3b-480p.avif" alt="" /><span class="fd-card-title">Wan 2.1</span><span>Bidirectional text-to-video and image-to-video generation.</span></a>
    <a class="fd-card fd-model-card" href="models/wan22/"><img class="fd-card-preview" src="_static/model_clips/wan22/wan22-ti2v-5b.avif" alt="" /><span class="fd-card-title">Wan 2.2 TI2V-5B</span><span>Bidirectional text-and-image-to-video generation.</span></a>
    <a class="fd-card fd-model-card" href="models/cosmos_predict2/"><img class="fd-card-preview" src="_static/model_clips/cosmos_predict2/cosmos2-t2v-2b-720p.avif" alt="" /><span class="fd-card-title">Cosmos-Predict2.5</span><span>Bidirectional Text2World, Image2World, and Video2World generation.</span></a>
  </div>
</div>
