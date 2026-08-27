<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Waypoint 1.5 architecture

This package is the model layer for the published
[Overworld/Waypoint-1.5-1B](https://huggingface.co/Overworld/Waypoint-1.5-1B)
checkpoint. It implements checkpoint loading, controls, the autoregressive
diffusion pipeline, sparse K/V history, and the shared TAEHV codec. The
[V2 application](../../integrations_v2/waypoint/README.md) owns CLI arguments,
sessions, browser events, and presentation.

This document is the concise design-review reference. Exact validation commands,
hardware, parity metrics, and long-rollout evidence are recorded in
[VALIDATION.md](../../integrations_v2/waypoint/VALIDATION.md).

## Scope and decisions

| Area | Design |
|---|---|
| Model | Dense BF16 autoregressive DiT, 24 blocks, width 2048, 32 query heads, 16 K/V heads |
| Parameter accounting | Upstream model card: 1.2B; pinned BF16 checkpoint: 1,860,823,096 serialized tensor elements |
| Inputs | Four-frame image seed plus one keyboard/mouse/wheel control per action |
| Output | One 32-channel latent and four RGB frames per action |
| Denoising | Fixed four-step rectified-flow Euler schedule: 1.0, 0.9, 0.75, 0.3, 0.0 |
| History | Dense 16-action local window; sparse 128-action global horizon in every fourth block |
| Context | 128 latent actions / 512 presented RGB frames on global-attention blocks |
| Presentation | Native TAEHV canvas, 1024x512 TCHW in [-1, 1], four frames per result |
| State | Model weights shared by the application; cache, RNG, controls, and seed isolated per session |
| Concurrency | An application lock serializes shared model/RNG work; output tensors leave the model loop detached |

The published config has `prompt_conditioning: null`; this integration therefore
does not load a text encoder or expose a prompt. It also does not currently
implement the upstream 360P checkpoint, quantization, or multi-GPU execution.

## Use cases and modalities

```mermaid
flowchart LR
    user([Interactive user])
    replay([Replay, test, or benchmark])
    seed[RGB or RGBA seed image]
    live[Keyboard, mouse buttons,<br/>relative motion, wheel]
    file[Versioned JSON controls]
    app((waypoint-1-5-1b))
    model[Waypoint 1.5 DiT + TAEHV]
    browser[Live WebRTC video]
    mp4[Deterministic MP4]
    metrics[Per-action metrics JSON]
    prompt[Text prompt<br/>not supported]

    user --> seed
    user --> live
    replay --> seed
    replay --> file
    seed --> app
    live --> app
    file --> app
    prompt -. excluded by checkpoint .-> app
    app --> model
    model --> browser
    model --> mp4
    model --> metrics
```

The seed establishes visual state; it is not a continuing image-conditioning
stream. File controls select finite, reproducible MP4/metrics runs. Omitting a
control file selects an open-ended live session driven by V2 browser events.

## Component and file view

```mermaid
flowchart TB
    subgraph core[flashdreams core and V2 runtime]
        registry[application_registry.py<br/>slug discovery]
        runner[application_runner.py / session_runner.py]
        api[IApplication / ISession / IModelLoop]
        sinks[WebRTCClientWindow / Mp4ClientWindow<br/>MetricsOutputSink]
        base[StreamInferencePipeline<br/>DiffusionModel / Transformer]
        taehv[recipes/taehv<br/>Hy15TAEHVEncoder and Decoder]
    end

    subgraph adapter[integrations_v2/waypoint]
        app[app.py<br/>WaypointApplication]
        session[session.py<br/>WaypointSession and ModelLoop]
        events[control_events.py<br/>ControlEventAdapter]
    end

    subgraph waypoint[integrations/waypoint]
        config[config.py and spec.py<br/>checkpoint contract]
        pipeline[pipeline.py<br/>WaypointInferencePipeline]
        controls[controls.py / encoder.py<br/>WaypointControl]
        transformer[transformer/impl.py<br/>WaypointTransformer]
        network[transformer/network.py<br/>WaypointDiT]
        cache[transformer/cache.py<br/>WaypointKVCache]
        scheduler[scheduler.py<br/>fixed Euler schedule]
        codec[decoder.py<br/>WaypointTAEHVDecoder]
        checkpoint[checkpoint.py<br/>strict state-dict mapping]
    end

    registry --> app
    runner --> app
    api -. implemented by .-> app
    api -. implemented by .-> session
    runner --> session
    runner --> sinks
    app --> session
    app --> pipeline
    session --> events
    session --> pipeline
    pipeline --> controls
    pipeline --> transformer
    pipeline --> scheduler
    pipeline --> codec
    transformer --> network
    network --> cache
    transformer --> checkpoint
    config --> pipeline
    base -. specialized by .-> pipeline
    base -. specialized by .-> transformer
    taehv -. reused by .-> pipeline
    taehv -. specialized by .-> codec
```

The package boundary is intentional: `integrations/waypoint` is reusable model
inference code and imports no V2 runtime classes. `integrations_v2/waypoint` is
the thin application adapter and depends on both FlashDreams and the model
package.

## Class and ownership view

```mermaid
classDiagram
    direction LR

    class IApplication
    class ISession
    class IModelLoop
    class StreamInferencePipeline
    class Transformer
    class StreamingEncoder
    class Hy15TAEHVDecoder

    class WaypointApplication {
        -pipeline
        -pipeline_lock
        +init(args)
        +session_desc()
        +create_session(desc)
    }
    class WaypointSession {
        -seed_frames
        -controls
        -state
        +init()
        +close()
    }
    class WaypointModelLoop {
        +step(index, events)
        +reset()
        +close()
    }
    class WaypointModelState {
        +cache
        +rng_state
        +control_events
        +controls_generated
    }
    class WaypointInferencePipeline {
        +initialize_cache(seed_pixels)
        +generate(index, cache, control)
        +finalize(index, cache)
    }
    class WaypointControlEncoder
    class WaypointTransformer
    class WaypointDiT
    class WaypointKVCache
    class WaypointTAEHVDecoder

    IApplication <|-- WaypointApplication
    ISession <|-- WaypointSession
    IModelLoop <|-- WaypointModelLoop
    StreamInferencePipeline <|-- WaypointInferencePipeline
    StreamingEncoder <|-- WaypointControlEncoder
    Transformer <|-- WaypointTransformer
    Hy15TAEHVDecoder <|-- WaypointTAEHVDecoder

    WaypointApplication o-- WaypointInferencePipeline : shares weights
    WaypointApplication --> WaypointSession : creates
    WaypointSession *-- WaypointModelState : owns
    WaypointSession --> WaypointModelLoop : registers
    WaypointModelLoop --> WaypointInferencePipeline : drives
    WaypointInferencePipeline *-- WaypointControlEncoder
    WaypointInferencePipeline *-- WaypointTransformer
    WaypointInferencePipeline *-- WaypointTAEHVDecoder
    WaypointTransformer *-- WaypointDiT
    WaypointDiT --> WaypointKVCache : updates
```

The application owns expensive immutable modules. A session owns all mutable
rollout state, including the transformer K/V cache, TAEHV state, seed tensors,
RNG state, and control adapter. `StepResult` tensors are detached before
publication so runtime queues and encoders do not retain model autograd state.

## Initialization and action sequence

```mermaid
sequenceDiagram
    actor Client
    participant Runtime as V2 runtime
    participant App as WaypointApplication
    participant Session as WaypointSession
    participant Loop as WaypointModelLoop
    participant Pipeline as WaypointInferencePipeline
    participant DiT as WaypointTransformer / DiT
    participant Cache as K/V + TAEHV caches
    participant Sink as WebRTC or MP4 sink

    Client->>Runtime: slug, runtime args, Waypoint args
    Runtime->>App: init(args)
    Runtime->>App: create_session(session_desc)
    App->>App: lazily load one BF16 model
    App-->>Runtime: new session with shared model + lock
    Runtime->>Session: init()
    Session->>Pipeline: initialize_cache(four seed frames)
    Pipeline->>Cache: TAEHV encode and prime decoder state
    Pipeline->>DiT: sigma-zero seed flow
    DiT->>Cache: commit action 0 K/V

    alt step 0
        Runtime->>Loop: step(0, events)
        Loop-->>Runtime: detached seed StepResult (4 frames)
    else generated action N
        Runtime->>Loop: step(N, ordered input events)
        Loop->>Pipeline: generate(N, cache, control)
        loop sigmas 1.0 to 0.3
            Pipeline->>DiT: predict provisional flow
            DiT->>Cache: replace provisional action-N K/V
        end
        Pipeline->>Cache: decode clean latent to 4 RGB frames
        Loop->>Pipeline: finalize(N, cache)
        Pipeline->>DiT: sigma-zero clean-state evaluation
        DiT->>Cache: commit action-N K/V
        Loop-->>Runtime: detached TCHW StepResult
    end

    Runtime->>Sink: present or encode four frames
    opt browser reset
        Client->>Runtime: reset event
        Runtime->>Loop: reset()
        Loop->>Pipeline: rebuild cache from retained seed
    end
```

The four denoise evaluations may overwrite only the current provisional slot.
`finalize` performs the separate sigma-zero evaluation that commits clean K/V
state for the next action. Skipping or reordering that transition changes the
autoregressive world state.

## Tensor and control contracts

| Boundary | Contract |
|---|---|
| Display seed | `[4, 3, 512, 1024]` TCHW float, normalized to `[-1, 1]` |
| Codec seed | `[B, 4, 3, 512, 1024]` float in `[0, 1]` |
| One model action | `[B, 1, 32, 32, 64]` latent before 2x2 patchification |
| DiT token stream | 512 tokens per action, width 2048 |
| Public control | 256-way multi-hot buttons, `mouse_dx`, `mouse_dy`, ternary wheel |
| Model result | `[4, 3, 512, 1024]` detached TCHW float in `[-1, 1]` |

The upstream runtime can resize a 1280x720 client image to the model's
1024x512 codec canvas and resize decoded frames back to 1280x720. FlashDreams
deliberately exposes the native 1024x512 canvas and performs no post-generation
spatial resample.

## Cache policy

Each action has 512 spatial tokens. Most transformer blocks retain the latest
16 actions densely. Global blocks occur at indices 3, 7, 11, 15, 19, and 23;
they span 128 actions but pin only every eighth historical action plus the
current action. On CUDA, fixed-capacity ring storage and a compiled
`FlexAttention` block mask avoid reallocating or concatenating history. CPU
tests use a compact dictionary representation as the readable reference.

During denoising the cache is frozen: repeated evaluations replace a tail slot
without advancing history. During seed establishment and `finalize`, it is
unfrozen so the clean action is committed exactly once.

## Control files

Pass a JSON action timeline with `--controls-file`. Every field within an
action is optional:

```json
{
  "schema_version": 1,
  "actions": [
    {"buttons": [32], "mouse_dx": 0.1, "mouse_dy": 0.0, "scroll_wheel": 0},
    {},
    {"buttons": [1, 32]}
  ]
}
```

`buttons` must contain IDs in `[0, 256)`; mouse values must be finite; wheel
is `-1`, `0`, or `1`. See
[ADR-1](../../integrations_v2/waypoint/ADR-1-control-events.md) for browser-event
mapping and reset/focus semantics.

## Review boundaries and known limitations

- One V2 application may create multiple sessions, but shared model execution
  is serialized. The current WebRTC server accepts one browser client.
- Reset is deterministic for a fixed seed and control sequence and rebuilds the
  cache without temporarily retaining two full GPU caches.
- The session advertises 60 FPS for playback pacing. Generation throughput is
  hardware- and stack-dependent; it is not guaranteed to sustain that rate.
- The integration preserves upstream model behavior, including possible
  long-rollout drift, unstable geometry, inconsistent objects, and implausible
  motion. It is not a physically accurate or safety-critical simulator.
- Consolidating this application with the shared Cam2V app is deferred until
  Cam2V exposes Waypoint's raw mouse/button/wheel control contract.

## Validation

CPU tests cover checkpoint contracts, controls, lifecycle, reset, detached
outputs, MP4 frame accounting, and session isolation. CUDA tests cover local
and global fixed-cache attention through ring wraparound. Published-weight
validation covers official-reference parity, 15 distinct scenes at 40 actions,
one 118-action rollout, native-resolution MP4 output, and steady-state
performance on an RTX PRO 6000 Blackwell.

See [the V2 validation record](../../integrations_v2/waypoint/VALIDATION.md) for
the exact revisions, hashes, commands, metrics, and acceptance evidence.
