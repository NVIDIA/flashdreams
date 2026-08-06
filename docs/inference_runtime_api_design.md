<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlashDreams Inference Runtime API Design Proposal

Date: July 30, 2026

## Summary

This proposal defines a standard inference runtime API for FlashDreams
integrations. The goal is to make world-model integrations easier to build,
benchmark, and run without forcing every model into the same input shape or
optimization stack.

The proposed API separates the pieces that are currently mixed together in
integration-specific runner code:

- `InferenceConfig`: how the model and inference stack should run;
- `UserInputs`: controls or events from an app, replay trace, or benchmark;
- `InferenceInput`: prompts, frames, videos, trajectories, maps, scene data, and
  other values required by a specific model;
- input mapping: model/application-specific conversion from user-facing inputs
  into model-facing inputs;
- runtime/session execution: model setup, warmup, per-rollout state, and
  stepping;
- output targets: WebRTC, native display, MP4, benchmark artifacts, or headless
  runs;
- metrics/profiling: timings, memory, traces, NVTX ranges, and benchmark
  outputs.

Current T2/T3 implementation notes are in
`docs/inference_runtime_inputs_implementation.md`.

The supported-model input inventory used to revisit T2/T3 is in
`docs/inference_runtime_supported_inputs_inventory.md`.

The API should standardize the envelope and lifecycle. It should not pretend
that all world models have the same inputs, that all models use the same
optimization stack, or that a raw checkpoint can fully describe how to run the
model.

## Current Implementation Plan

Implementation should happen on an experimental integration branch. PRs for this
work should target that branch until the API shape, LingBot migration, and
OmniDreams migration are all working well enough to merge to `main` together.

The experimental branch can temporarily break or simplify command-line options
while the demos are being moved to the new API. The required outcome is that the
LingBot and OmniDreams demos still run through the new runtime path, and that
benchmark tooling can confirm they are at least broadly healthy before the
branch is merged back to `main`.

Initial scope:

- define the minimal runtime API envelope;
- migrate LingBot and OmniDreams to use it;
- support selectable output modes such as MP4, JPEG/MJPEG stream, WebRTC, and
  headless/null where appropriate;
- use or update benchmark tooling to verify the migrated demos;
- defer broader model migrations, hosted execution, full autotune, and polished
  metrics until the first branch proves the API shape.

## Task Tracker

| ID | Status | Workstream | Can run in parallel? | Depends on | Done when |
| --- | --- | --- | --- | --- | --- |
| T0 | Complete | Create experimental branch and contribution rules. | No, this starts the work. | None. | Branch exists, PR target is agreed, and main merge criteria are written down. |
| T1 | Complete | Minimal API envelope and naming. | Partly. | T0. | `InferenceConfig`, `UserInputs`, `InferenceInput`, runtime/session, output target, and mapping boundaries are defined well enough for demos to use. |
| T2 | Complete | Event-based `UserInputs`. | Yes, after T1 direction is agreed. | T1. | User inputs are primarily timestamped events; replay traces and derived snapshots are supported where needed. |
| T3 | Complete | `CanonicalInputs`, `InferenceInput`, schemas, and mapping boundary. | Yes, after T1 direction is agreed. | T1. | Models can declare required global/per-step inputs, and mappings can convert canonical inputs into inference inputs. |
| T4 | Planned | `ModelRunner`, `InferenceRuntime`, and `InferenceSession` skeleton. | Partly. | T1. | A minimal standard loop can initialize a runtime, run at least one sequential session, and close cleanly. |
| T5 | Planned | Output mode selection. | Yes, after the result/output shape is agreed. | T1, T4. | A run can choose output behavior such as MP4, JPEG/MJPEG stream, WebRTC, benchmark artifact, or headless/null without changing model code. |
| T6 | Planned | LingBot migration. | Yes, once T2-T4 have a usable skeleton. | T2, T3, T4. | LingBot runs through the new API path with its event inputs mapped into model inputs. |
| T7 | Planned | OmniDreams migration. | Yes, once T2-T4 have a usable skeleton. | T2, T3, T4. | OmniDreams runs through the new API path with its model-specific inputs and mapping preserved. |
| T8 | Planned | Benchmark/smoke verification for LingBot and OmniDreams. | Preparation can run early; final gate is late. | T5, T6, T7. | Existing or updated benchmark tooling can run both migrated demos and produce enough evidence that they still work. |
| T9 | Planned | Metrics and profiling normalization for the branch. | Yes, but final integration is late. | T4, T5, T8. | Basic canonical metrics are emitted for migrated demos; deeper metrics can remain follow-up work. |
| T10 | Planned | CLI compatibility and migration cleanup. | Yes, after demo migrations start. | T6, T7. | Required demo commands are restored or replaced, temporary hacks are removed, and user-facing docs/notes match the branch behavior. |
| T11 | Planned | Stabilize and merge experimental branch to `main`. | No, final integration step. | T6-T10. | LingBot and OmniDreams pass agreed smoke/benchmark checks, review feedback is addressed, and the branch can merge as one API transition. |

Suggested parallel split:

- one person owns T4 and keeps it aligned with the completed T1 envelope,
  because the standard loop is now the critical path;
- one person owns T2/T3, because event inputs, schemas, and mapping need to
  stay coherent;
- one person owns T5/T8/T9, because outputs, benchmarks, and metrics are tightly
  related;
- LingBot and OmniDreams can be assigned separately once the skeleton is usable;
- one person should track branch health, CLI compatibility, and merge readiness.

## Architecture

```text
Optional discovery for CLI, benchmark, hosted, or installed-package flows:
  Model/preset registry
    -> adapter/preset/default setup/scenario metadata
    -> contributes defaults to the app-supplied run setup

Main runtime flow:
App / integration / benchmark / transport
  chooses how the run is driven and where output goes
  supplies run setup:
    InferenceConfig + UserInputs + InferenceInput + output/metrics options
  |
  v
ModelRunner / standard loop
  orchestrates validation, lifecycle, stepping, output, and metrics
  uses input mapping to:
    validate that user/app inputs can drive the model
    build global and per-step InferenceInput during the run
  |
  v
InferenceRuntime
  reusable heavyweight lifecycle: distributed init, model load, compile, warmup
  load once; create sessions sequentially unless the backend supports concurrency
  |
  v
InferenceSession
  one rollout/stream: global conditioning, cache/state, current step, reset
  keeps per-run state from leaking across prompts, clients, or benchmark repeats
  |
  v
Model implementation / inference pipeline
  hot path: encode -> model step -> decode -> cache/finalize
  |
  v
Output target
  WebRTC | native window | MP4 | benchmark | headless/null
  |
  v
Metrics / artifacts / logs / reports / traces
```

## Example Sequential Session Flow

The runtime/session split is primarily about reusing expensive model setup while
keeping each rollout's state isolated. The default mental model should be
sequential sessions, not required concurrent sessions.

```text
ModelRunner / standard loop
  |
  v
Create InferenceRuntime from InferenceConfig
  load checkpoint/model
  initialize distributed/backend state
  compile/capture/warm up if configured
  |
  v
Start InferenceSession A
  global conditioning: prompt/frame/scene/etc.
  per-session state: cache, current step, reset state
  step 0 -> step 1 -> ... -> done
  outputs -> Output target
  metrics -> Metrics recorder
  close session A
  |
  v
Start InferenceSession B
  new global conditioning or replay scenario
  independent cache/state
  step 0 -> step 1 -> ... -> done
  outputs -> Output target
  metrics -> Metrics recorder
  close session B
  |
  v
Close InferenceRuntime
  release model/backend resources
```

For v0, an `InferenceRuntime` may support only one active session at a time.
Concurrent sessions should be treated as an optional backend/model capability,
not a baseline API requirement.

`StreamInferencePipeline` should remain an important local implementation path
for models that already use it, but it should not be treated as the only
possible model boundary. A session may call `StreamInferencePipeline`, another
local model implementation, a Dynamo-like backend, or a hosted service.

## System Components

| Component | Role | Boundary |
| --- | --- | --- |
| Model/preset registry | Lists what can run: model/preset slugs, scenarios, capabilities, resource hints, and supported output modes. | Must remain cheap to query and must not load checkpoints. |
| App / integration / benchmark / transport | Owns the user-facing mode: CLI, native integration, WebRTC, benchmark, hosted request, or replay. | Supplies run setup, user inputs, model inputs, and output target selection. |
| User input library | Normalizes live or replayed controls into FlashDreams-supported user input events/windows. | Shared primitives for keyboard, reset, prompt/image selection, traces, and future scalar controls. |
| Input mapping | Converts user/app inputs plus global conditioning into the model-specific inputs needed by the session. | A model adapter may provide a default mapping; runtimes, applications, benchmarks, and replay tools may override it without changing the model step. |
| ModelRunner / standard loop | Orchestrates one run from setup through runtime initialization, stepping, output, metrics, and teardown. | Shared orchestration layer used by CLIs, benchmarks, MP4 runs, and simple realtime flows. |
| InferenceRuntime | Owns heavyweight lifecycle: distributed init, model construction, checkpoint loading, compile/capture, warmup, hosted-service connection, and teardown. | Long-lived reusable runtime created from `InferenceConfig`; lets FlashDreams load/warm once and create sessions sequentially unless the backend supports concurrency. |
| InferenceSession | Owns one rollout or stream: global conditioning, cache state, current step, reset behavior, step requirements, and step execution. | Per-rollout interface consumed by the standard loop; keeps state isolated across prompts, browser clients, replay scenarios, or benchmark repeats. |
| Model implementation / inference pipeline | Implements encode, model step, decode, cache updates, and model-specific optimizations. | FlashDreams wraps this boundary; it should not replace every model implementation. |
| Output target | Consumes generated outputs and handles presentation or persistence. | Separate from model execution so the same session can feed WebRTC, MP4, benchmark, or headless output. |
| Metrics, artifacts, and profiling | Records timings, memory, quality data, logs, reports, traces, and optional NVTX ranges. | Shared observation layer for local runs, benchmarks, CI smoke, and hosted runs. |

## API Layers

FlashDreams should expose layered APIs rather than a single all-or-nothing
interface:

```text
High-level runtime API
  run setup -> standard loop -> output targets -> metrics/artifacts

Adapter/runtime API
  model adapter -> InferenceRuntime -> InferenceSession

Low-level inference API
  StreamInferencePipeline -> encoders/decoders -> cache/perf/profiling helpers
```

| Layer | Intended user | Provides |
| --- | --- | --- |
| High-level runtime API | Users who want FlashDreams to own the run loop. | Run setup, input mapping, runtime/session lifecycle, output targets, metrics, profiling, and benchmark artifacts. |
| Adapter/runtime API | Model owners who want their model to plug into the standard loop. | Model adapter, input requirements, runtime/session implementation, and model-specific mapping or validation. |
| Low-level inference API | Users who want to own their own loop while reusing FlashDreams building blocks. | `StreamInferencePipeline`, encoders, decoders, cache helpers, profiling tools, and optimization utilities. |

These layers should remain compatible. The new runtime API sits above the
existing lower-level pieces; it does not replace them.

## Goals

- Make FlashDreams easier to use for new world-model integrations.
- Keep model-specific input semantics explicit instead of hiding them in runner
  code.
- Avoid a single monolithic inference stack; different models should be able to
  validate and use different optimization features.
- Separate model execution from presentation and persistence.
- Support both live input and deterministic replay through the same
  runtime/session boundary.
- Make metrics, benchmark artifacts, and profiling first-class without forcing
  profiling overhead into normal runs.
- Preserve room for local single-GPU, local distributed, Dynamo-like, and hosted
  execution.

## Non-Goals

- Do not infer arbitrary model semantics from a raw checkpoint.
- Do not require every model to use the same encoder, decoder, scheduler,
  control representation, transport, or optimization set.
- Do not make WebRTC or native display part of the model API.
- Do not make autotuning part of normal inference startup.
- Do not require users to use the high-level standard loop when they only need
  lower-level inference building blocks.
- Do not require every existing integration to migrate in one large change.

## API Placement

The new API should sit above the existing `flashdreams.infra` layer. Existing
pipelines, encoders, decoders, runner configs, realtime input helpers, WebRTC
code, and quality/benchmark utilities should be reused where possible.

The exact package layout and class definitions can be decided during
implementation. This document should define responsibilities and boundaries, not
the final Python shape.

## InferenceConfig

`InferenceConfig` describes how to run the model/runtime. It should cover:

- model or preset identity;
- checkpoint or model asset selection;
- execution backend, such as local single GPU, local multi-GPU, Dynamo-like, or
  hosted/external execution;
- device placement, precision, and resource hints;
- optimization choices such as compile, CUDA graph capture, attention backend,
  cache policy, overlap, prefetch, and native extensions;
- runtime-affecting profiling or tracing options.

It should not contain prompts, keyboard state, browser settings, MP4 paths,
benchmark output directories, or other app/output settings. Those belong in the
run setup around `InferenceConfig`.

Existing `StreamInferencePipelineConfig` and `InstantiateConfig` style configs
can remain valid model references behind this layer. The model adapter should
validate which execution and optimization choices are supported. Unsupported
choices should fail clearly or be explicitly handled only when the user selected
an automatic mode.

## UserInputs

`UserInputs` describes user-facing controls produced by a live UI, browser,
native app, replay trace, synthetic benchmark driver, or no-op source.

User inputs should primarily be represented as timestamped events. This gives
live apps, replay traces, and benchmarks the same basic shape, and lets
FlashDreams route, drain, or window those events when a model session asks for
the next chunk of inputs. Resampling and interpolation should remain
input-specific mapping or helper behavior, because controls such as rotations,
poses, or controller state may need semantics that generic runtime code cannot
infer safely.

Initial supported user input types should stay close to what FlashDreams already
uses:

- keyboard keydown/keyup events;
- reset requests;
- prompt or image selection/update events;
- future scalar controls such as throttle, brake, steer, or camera axes once an
  integration needs them.

Snapshot-style inputs, such as current key state, can still be supported when
useful. They should be treated as a derived or compatibility form rather than
the primary user-input abstraction.

User inputs are not model inputs. A keyboard event does not have one universal
meaning. One model may map it to pose segments, another to steering commands,
and another may ignore it.

## CanonicalInputs And InferenceInput

Inputs move through three layers:

```text
UserInputs  ->  CanonicalInputs  ->  InferenceInput
   raw           canonicalized          encoded
```

Raw device events for live control are canonicalized into device-independent
modalities before application or mapping logic consumes them, so adding a
keyboard, gamepad, or wheel is a converter registration rather than an
application change. Global conditioning is application-owned and reaches
`InferenceInput` directly; it does not pass through live device canonicalization.
`InferenceInput` is what an `InferenceSession` actually receives.

`CanonicalInputs` describes device-independent live control for one requested
input window. `InferenceInput` describes the data the model or inference
pipeline actually requires, split into two conditioning slots:

- global conditioning: values that condition the whole rollout;
- per-step conditioning: values needed for one generated chunk or frame window.

Examples of global conditioning include prompt, negative prompt, conditioning
frame, input video, scene id, HD map asset, camera calibration, initial camera
pose, seed, or model-specific fields.

Global conditioning establishes session-global model state when a session
starts or resets. During an active rollout, a non-empty global-conditioning
payload passed to `InferenceSession.step()` asks the session to update that
state when the model supports it. Reset remains a separate explicit session
method.

Examples of per-step conditioning include frame timestamps, pose segments,
camera trajectory chunks, rendered HD map frames, conditioning video windows,
control tensors, event markers, or model-specific fields.

Inference input payloads should use semantic names, not only modality names. For
example, a first frame and an HD map frame should be distinct inputs even if
both are image-like values.

Model input names, input modalities, and schema metadata should be open-ended.
Supported integrations such as SANA-WM, LingBot, Omnidreams, and future
external adapters may need different semantic fields. Adding a new model should
usually mean adding adapter-owned schema declarations and mappings, not changing
a central FlashDreams enum.

Consumption cadence is a separate hint from input scope. A field may be
provided through global conditioning because it is session-global state, while
the adapter consumes or slices it during every step. That can be recorded as
`frequency_consumed` metadata without changing whether the field belongs in
`global_conditioning_fields` or `step_fields`.

For interactive runs, most `InferenceInput` values will be app-owned global
conditioning plus per-step inputs produced by input mapping. For MP4 generation
and benchmarking, the API should also support fixed per-step model inputs so
runs can be deterministic.

## Schemas

The API should support lightweight `UserInputSchema`, `CanonicalInputSchema`,
and `InferenceInputSchema`
metadata.

These schemas are not meant to be a rich type system or a replacement for
model-specific validation. They should be just enough to answer:

- what can this app, transport, trace, or benchmark source provide?
- what does this model require before session start and at each step?
- can this event source drive this model with the selected mapping?

The purpose is to fail early before expensive model initialization, produce
clearer errors, make fixed scenarios easier to validate, and avoid ambiguous
dict payloads where keys only describe modality.

Schema objects may carry open-ended metadata for query-time hints such as
coordinate frame, units, rough shape summary, accepted file suffixes, schema
URI, model family, or source/transport details. Metadata should help humans and
adapter selection code, but compatibility should still be based on the declared
event capabilities, semantic model fields, input modalities, and schema phases.
Consumption-cadence hints are descriptive and adapter-owned.

For simple CLI text-to-video or image-to-video runs, `UserInputSchema` can be
trivial or omitted because there may be no live controls. `InferenceInputSchema` is
more important because each supported model still needs to declare the
model-facing values it expects.

## Model Requirements

A raw checkpoint should not be treated as self-describing. It may imply tensor
shapes or architecture details, but it usually does not fully define:

- required semantic inputs;
- initial versus per-step inputs;
- units for timestamps, poses, or calibration values;
- how user controls become model controls;
- preprocessing, encoder, decoder, mask, prompt, or cache rules.

Therefore, a FlashDreams-supported model should have an adapter or integration
layer that declares its model input requirements, declares any user inputs it can
map by default, and prepares inputs for the underlying model implementation.

Users running an existing FlashDreams-supported model should not need to write
that adapter. Developers bringing a new world model to FlashDreams should expect
to provide one.

## External Model Usage

Users should be able to run their own models without adding those models to the
FlashDreams repository. The flow depends on which API layer they use:

```text
High-level runtime API
  user supplies or installs model adapter
  FlashDreams owns standard loop, outputs, metrics, benchmarks

Adapter/runtime API
  model owner implements adapter/runtime/session
  adapter can be passed directly or registered by an installed package

Low-level inference API
  user owns loop and lifecycle
  user reuses pipeline, encoder/decoder, cache, profiling, or optimization tools
```

| Flow | Registry needed? | Who provides model-specific code? | Result |
| --- | --- | --- | --- |
| Direct Python | No. | User or model owner passes an adapter/setup directly. | FlashDreams can run the standard loop without the model living in the repo. |
| Installed package | Yes, for discovery. | External or internal package registers adapters/presets. | CLIs, benchmarks, and hosted schedulers can discover the model cheaply. |
| Low-level only | No. | User owns the loop and calls lower-level FlashDreams pieces directly. | Useful when the user wants optimizations or pipeline helpers but not the standard loop. |

The model adapter is a role/boundary, not necessarily a concrete class. It is
the model-specific code that declares input requirements, validates supported
configs, creates the runtime/session, and connects FlashDreams to the actual
model implementation.

The registry should not be treated as a central FlashDreams-owned catalog of all
possible models. It is a discovery mechanism for installed adapters. Built-in
public integrations, internal GitLab-only integrations, and third-party packages
can all participate through the same mechanism.

FlashDreams should not claim to run an arbitrary checkpoint with no adapter
unless the checkpoint already matches a supported generic adapter.

## Input Mapping

Input mapping is required whenever `UserInputs` need to become per-step
`InferenceInput`. In the T1 envelope this boundary is represented by a separate
`InputMapping` protocol. A model adapter may provide the default mapper because
it knows how its supported user controls affect model-facing inputs. Applications,
benchmarks, replay tools, or hosted runtimes may replace that mapper when they
need a different wire surface or aggregation policy.

The selected mapping may be a single mapper or a composed set of mappers, so one
run can combine separate prompt, first-frame, and live-control mappings instead
of routing everything through one object.

There are two separate moments to keep clear:

- before runtime initialization, FlashDreams should select the mapping or mapper
  set and check obvious compatibility between the app event source and the
  model;
- during the standard loop, the runtime or runner passes app-owned global
  `InferenceInput` through the selected mapping before session start, then
  queues and timestamps user events, canonicalizes the session-requested window,
  and uses the selected mapping to build per-step `InferenceInput`.

This keeps the Reactor-style contract intact: the model-side integration can
declare user inputs, declare model inputs, and provide a default mapping, while
the runtime owns transport, event validation, timestamping, input queue/window
selection, output delivery, and optional overrides.

`StepRequest` and `StepResult` are per-step runtime messages, not declarative
schemas. `InferenceSession.next_step_request()` returns a `StepRequest` to say
which step is next, which user-input time window to map, and whether this step
has any narrower `InferenceInputSchema` than the session default. The runner or
application then builds an `InferenceInput` and calls `InferenceSession.step()`,
which returns a `StepResult` carrying the generated output, output timing,
metrics, and step metadata.

Examples:

- T2V mapping validates a prompt and creates no per-step control inputs.
- I2V mapping validates a prompt plus first frame and creates no live controls.
- A keyboard-driven integration maps key events or event windows into pose
  segments or steering controls.
- OmniDreams-like integrations may map driving commands into camera poses, HD
  map frames, and dynamic actor state.
- Benchmark mapping can read fixed event traces and produce identical step
  inputs each run.

The compatibility check should be treated as early validation, not a guarantee
that the run will succeed. It can catch obvious mismatches, but the model
adapter/runtime still owns deep tensor validation and model semantics.

## Runtime And Standard Loop

The standard loop should be shared by CLI generation, headless playback, MP4
generation, benchmarks, and simple realtime applications.

A run should:

1. Discover the model or preset without loading checkpoints.
2. Resolve inference config, user inputs, model inputs, output target, metrics,
   profiling, and optional scenario setup.
3. Validate that the event source and mapping can drive the selected model.
4. Initialize the runtime.
5. Start a session from global conditioning inputs.
6. For each step, ask the session what it needs, gather live or fixed inputs,
   build step model inputs, run the session step, route outputs, and record
   metrics.
7. Finalize output artifacts, metrics, logs, reports, and traces.

Realtime transports may need an async variant, backpressure, and explicit flow
control, but the conceptual boundary should remain the same: event/input source,
input mapping, session, output target, metrics.

The session should expose what it needs for the next step rather than requiring
the app or output layer to guess. This matters because AR step 0 can differ from
steady-state steps, and encoder/decoder temporal compression can produce
different input and output frame windows.

Input and output timing should share a session timeline even when raw capture
rates and presentation rates differ. A session can request a user-input window
for mapping, then return an output window or equivalent metadata so an output
target can present the generated chunk at the intended cadence.

## Output Targets

Output handling should be separate from model execution. The model session
returns generated outputs and metadata; the output target decides what to do
with them.

Expected output targets include:

- WebRTC streaming;
- native window display;
- MJPEG or lightweight remote preview;
- MP4 writing;
- benchmark artifact writing;
- headless playback;
- null output for pure throughput measurements.

Display and transport can still affect measured performance through copies,
encoding, queueing, backpressure, and presentation timing. Those costs should be
measured as output-target or end-to-end metrics instead of being mixed into core
model-stage timings.

## Fixed Inputs, Benchmarks

The API should support fixed runs as a first-class case. This is needed for MP4
generation, benchmarks, regression testing, and autotune.

Two replay levels should be supported:

- user-event replay: records timestamped key events, prompt or image
  selection/update events, reset events, and timing, then runs normal input
  mapping;
- model-input replay: records or defines already-mapped per-step model inputs
  for stricter model-level regression tests.

User-event replay tests more of the application stack. Model-input replay is
better for isolating model runtime performance and reproducibility.

## Metrics And Profiling

Metrics should have a small canonical baseline plus optional extras.

The baseline should cover:

- lifecycle timing: startup, load, warmup, first-step latency;
- model-stage timing: encode, model step, decode, finalize/cache update;
- memory: allocated, reserved, peak, and per-rank where applicable;
- throughput: frames per second, chunks per second, real-time factor.

Realtime runs may add input-to-present latency, jitter, missed deadlines, queue
depth, dropped frames, WebRTC stats, encoder bitrate, and client stats.
Benchmark runs may add quality metrics, logs, MP4/image previews, and reports.

Persisted timing metrics should use seconds as the canonical unit because
seconds compose cleanly across Python timers, traces, and long-running
durations. Reports and UIs can display milliseconds for short latencies.

Profiling should be optional and controlled separately from normal metrics.
NVTX ranges should be supported for Nsight profiling, but profiling should not
be required for normal inference or benchmark runs.

## Autotune

Autotune should be a separate harness that evaluates candidate
`InferenceConfig` variants against fixed scenarios. It should not be part of
normal startup.

Autotune may search over compile, CUDA graph capture, attention backend,
precision, cache policy, overlap, prefetch, native extensions, and chunk size
when the model supports those knobs.

Results are only valid for a specific model, checkpoint, hardware, driver,
FlashDreams commit, and scenario. First-run compile/capture cost should be
separated from steady-state metrics. Agent assistance could help propose search
spaces or summarize results, but the measured selection process should be
deterministic code.

## Distributed And Hosted Execution

The API should leave room for local single-GPU, local multi-GPU, Dynamo-like
execution, and hosted execution such as a Reactor-style platform.

At this stage, the proposal should not define Reactor- or Dynamo-specific
contracts in detail. It should preserve the right boundary: execution backend
selection belongs in `InferenceConfig`, while backend-specific scheduling,
authentication, asset access, output streaming, artifact handling, and failure
behavior belong behind the runtime/backend implementation.

The practical order should be local first, then local distributed, then
hosted/distributed backends once concrete backend owners can validate the
requirements.

## Existing Code And Migration

The new API should reuse existing code instead of replacing everything:

- keep `flashdreams.infra.pipeline` as the common local encode/model/decode
  implementation path;
- keep existing encoder and decoder contracts and reuse temporal size helpers;
- keep existing runner configs and CLI compatibility during migration;
- reuse `KeyboardResampler` and realtime input helpers behind the new input
  boundary;
- treat WebRTC as a transport/output adapter and bridge it gradually;
- reuse existing quality and benchmark utilities where applicable;
- keep internal-only integrations registered only in the GitLab/internal
  workspace.

The task tracker near the start of this document is the source of truth for the
first implementation branch. The first milestone is intentionally narrower than
the full design: prove the API with LingBot and OmniDreams, selectable output
modes, and enough benchmark/smoke coverage to merge the experimental branch
back to `main` safely.

## Design Risks

- `InferenceConfig` could become too broad if prompts, controls, output paths,
  browser settings, and benchmark settings are added to it. Keep it focused on
  model/runtime execution.
- Dict-like model inputs are flexible but can fail late. Keep dict payloads for
  flexibility, but require lightweight schemas and adapter validation for
  supported models.
- Schemas could become too heavy. Keep them minimal and role-oriented.
- User inputs are not model inputs. Keep input mapping explicit and
  model/application-owned.
- Per-frame, per-chunk, and AR-step clocks are easy to confuse. The session
  should expose step requirements instead of making app code guess.
- Output separation is necessary but not free. Measure output and transport
  costs separately from core model timings.
- Hosted/distributed execution is still under-specified. Keep the API boundary
  open until backend owners validate concrete requirements.
- Existing WebRTC behavior is nontrivial. Bridge it gradually to avoid
  regressions.
- Public/internal boundaries must remain clean. Internal adapters, slugs, and
  scenarios should not leak into the public repo.

## Decisions Made In T1

Task T1 settles the initial package and naming envelope without committing to a
registry, standard loop, concrete output modes, or model migrations:

- The experimental API lives under `flashdreams.runtime`.
- The model-specific integration boundary is named `ModelAdapter`.
- Heavyweight lifecycle is split into `InferenceRuntime` and
  `InferenceSession`.
- Step data carriers are named `StepRequest` and `StepResult`. They are runtime
  messages around one call to `InferenceSession.step()`, not schema
  declarations; a session returns `None` from `next_step_request()` when the
  rollout is complete.
- Raw inputs use `UserInputs`, canonicalized inputs use `CanonicalInputs`, and
  model-facing inputs use `InferenceInput`.
  Both remain lightweight payload envelopes with shallow read-only mappings.
- `UserInputSchema`, `CanonicalInputSchema`, and `InferenceInputSchema` stay
  intentionally small: they
  declare supported event types and required named fields for early validation,
  not a full type system.
- Input mapping is represented by a separate `InputMapping` protocol. Model
  adapters may provide a default mapping; runtimes and applications may override
  it while preserving the `CanonicalInputs` to `InferenceInput` boundary. Simple
  fixed-input runs can use `IdentityInputMapping`.
- Output handling is represented by `OutputTarget`; `NullOutputTarget` is the
  initial headless implementation.
- Metrics collection is represented by `MetricsRecorder`; timing samples use
  seconds as the canonical unit.
- The minimum v0 user input shape is timestamped `UserInputEvent` records plus
  optional snapshot data. Concrete event-type catalogs are left to T2 and demo
  migrations.

## Remaining Decisions

- What direct-Python API should let users pass an external adapter without
  registering it?
- What package registration mechanism should third-party and internal adapters
  use for CLI discovery and benchmarks?
- What is the first public model to migrate?
- What metrics are required for every benchmark run?
- What metadata must be discoverable without loading checkpoints?
- What requirements do Dynamo/Reactor-style backends need before we commit to
  hosted execution details?

The document currently uses "integration" for model-specific packages and app
entrypoints. If the team prefers "model" as the public term, that can be changed
later without changing the architecture.

## Recommendation

Proceed with the proposed split:

- `InferenceConfig` for model/runtime execution;
- `UserInputs` for app-facing controls and replay traces;
- `CanonicalInputs` for device-independent application-facing inputs;
- `InferenceInput` for model-facing global and per-step conditioning;
- input mapping for model/application-specific conversion;
- runtime/session boundaries for lifecycle and stepping;
- output targets for display, streaming, files, and benchmarks;
- shared metrics and optional profiling.

The main constraint is that arbitrary world-model inputs cannot be standardized
away. FlashDreams can provide the shared envelope, loop, metrics, replay, and
output tools, but each supported model still needs an adapter that declares and
validates its own input contract.
