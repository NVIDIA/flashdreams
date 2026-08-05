<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Supported Model Input Inventory

This note inventories the inputs used by the currently supported FlashDreams
runners and interactive runtimes, plus the SANA-WM input surface on `main`, then
records the T2/T3 API implications. It is intentionally about input contracts,
not tensor shape validation or model quality.

## Inventory

WAN 2.1 T2V, Self-Forcing WAN 2.1 T2V, Causal-Forcing T2V,
FastVideo Causal WAN 2.2 T2V, and Cosmos Predict2 T2V:

- Source/app inputs: prompt text or prompt text file, pixel height/width, and
  fps or block count depending on runner.
- Model-facing initial inputs: prompt text plus latent/output height and width
  derived from run config.
- Model-facing step/update inputs: no live controls; AR loop steps with fixed
  session state.

WAN 2.1 I2V, Causal-Forcing I2V, and Cosmos Predict2 I2V:

- Source/app inputs: prompt text or prompt file, first-frame image path or URL,
  and pixel height/width.
- Model-facing initial inputs: prompt text and decoded first-frame tensor.
- Model-facing step/update inputs: no live controls.

FlashVSR:

- Source/app inputs: input video path or URL, chunk size, crop region, sparse
  ratio, and optional output FPS.
- Model-facing initial inputs: no explicit prompt at runner time; the prompt
  tensor is configured in the pipeline. Input video dimensions affect
  per-video runtime/pipeline setup.
- Model-facing step/update inputs: video chunks passed to
  `pipeline.generate(input=clip)`.

LingBot CLI:

- Source/app inputs: prompt or prompt path, first-frame image path, pose path,
  intrinsics path, total blocks, dimensions, and fps.
- Model-facing initial inputs: prompt text and first-frame tensor.
- Model-facing step/update inputs: `CamCtrlInput` with intrinsics, camera poses,
  and world scale.

LingBot WebRTC:

- Source/app inputs: session prompt, uploaded/remote/default first-frame image,
  keyboard events, reset requests, text-event catalog, and trigger events.
- Model-facing initial inputs: prompt text, first-frame tensor, base text
  embeddings, precomputed text-event embeddings, base intrinsics, and world
  scale.
- Model-facing step/update inputs: keyboard event windows become pose segments
  and camera trajectories. Text-event triggers can replace rollout text
  embeddings when the model supports it.

HY-WorldPlay WAN I2V:

- Source/app inputs: prompt or prompt path, first-frame image path or example
  image, pose string or pose JSON, memory-selection settings, dimensions, fps,
  and seed.
- Model-facing initial inputs: prompt text and first-frame tensor for cache
  initialization.
- Model-facing step/update inputs: pose data is bound for the rollout as action
  labels, view matrices, intrinsics, and memory-selection state before AR steps.

Omnidreams CLI:

- Source/app inputs: shared prompt or per-camera prompts, HDMap video paths,
  first-frame image/video paths, camera names, example-data UUID, and optional
  embedding save/load paths.
- Model-facing initial inputs: prompt list, first-frame tensor, view names; or
  precomputed text/image/negative-text embeddings.
- Model-facing step/update inputs: HDMap video chunks passed per AR step.

Omnidreams WebRTC:

- Source/app inputs: scene directory or scene UUID, scene variant, camera name,
  prompt/first-frame assets resolved from the scene, keyboard events, reset
  requests, and optional postprocess preset.
- Model-facing initial inputs: scene data, renderer, first-frame tensor, prompt,
  camera calibration/extrinsics, initial ego pose, and initial timestamp.
- Model-facing step/update inputs: keyboard event windows become ego poses,
  camera poses per view, and frame timestamps. The wrapper renders HDMap
  conditioning internally for each step.

Omnidreams interactive drive:

- Source/app inputs: scene bundle, keyboard events or wheel/controller samples,
  view-mode/reset/scene-exit controls, and vehicle/chunk config.
- Model-facing initial inputs: scene bundle, selected camera, prompt, initial
  RGB frame, initial rig pose, and initial timestamp.
- Model-facing step/update inputs: `DriverCommand` samples become trajectory
  chunks, rendered frames, and world-model conditioning.

Template recipe:

- Source/app inputs: synthetic runner config: batch size, height, width, context
  tokens, AR steps, and seed.
- Model-facing initial inputs: synthetic transformer context, optional negative
  context, height, and width.
- Model-facing step/update inputs: optional synthetic control tensor.

WAN 2.2 TI2V pipeline config:

- Source/app inputs: downstream runners use this rather than a standalone runner
  in this tree.
- Model-facing initial inputs: prompt text and first-frame image for TI2V-style
  cache initialization.
- Model-facing step/update inputs: downstream runners decide controls;
  HY-WorldPlay currently binds action/camera state around it.

SANA-WM bidirectional and streaming on `main`:

- Source/app inputs: first-frame image path, prompt or prompt path, optional
  negative prompt, camera trajectory path or action DSL, optional intrinsics
  path or derived intrinsics, frame count, fps, Stage-1 sampling knobs, seed,
  precision/refiner options, and streaming chunk/block settings.
- Model-facing initial inputs: decoder context such as prompt, fps,
  `save_stage1`, refiner seed, sink size, and streaming refiner window/block
  parameters.
- Model-facing step/update inputs: bidirectional passes one
  `SanaWMI2VConditioningRequest` into the single generation step. Streaming
  passes one `SanaWMStreamingI2VConditioningRequest` repeatedly; the
  conditioning encoder caches rollout-wide prompt, first-frame, camera, latent
  shape, and chunk-boundary state, then slices per AR chunk.
- Model-facing semantic fields include prompt, negative prompt, first frame,
  camera-to-world trajectory, intrinsics vec4 sequence, frame count, fps,
  sampling parameters, seed, and streaming chunking parameters.

## API Implications

The inventory changes the T2/T3 shape in four concrete ways.

First, a selected mapping is often a composition. A LingBot-like run needs prompt
mapping, first-frame mapping, and keyboard-to-camera mapping. Omnidreams may add
scene selection, camera selection, and HDMap mapping. The implementation should
support checking a set of mapping schemas as one compatibility surface, while
still allowing a single mapping object when that is simpler.

Second, `InferenceInputSchema` needs a lightweight lifecycle tag in addition to the
`initial` versus `step` phase. The phase answers when the value is needed at the
standard-loop level. The lifecycle tag distinguishes where the model adapter
uses it, such as:

- `runtime_config`: values that affect setup before model/runtime construction,
  such as FlashVSR input-video dimensions;
- `cache_init`: values passed when initializing or resetting a rollout cache,
  such as prompts, first frames, view names, and precomputed embeddings;
- `rollout_binding`: values bound after cache initialization but before AR
  steps, such as HY-WorldPlay action labels, camera tensors, and memory state;
- `step_input`: values consumed for one generated chunk, such as HDMap frames,
  camera trajectories, driver commands, video chunks, and timestamps;
- `session_update`: values that can update an active session when supported,
  such as LingBot text-event embedding swaps.

The lifecycle tag is metadata, not a new deep type system. If both a model field
and mapping output specify lifecycle, compatibility should require them to agree.
If either side omits it, matching stays permissive for simple schemas.

Third, `semantic_type` should be treated as a representation hint rather than a
universal semantic type. For example, `prompt` may arrive as inline text or a
path but become prompt text or text embeddings; the global conditioning frame
may arrive as a path, URL, bytes, or decoded tensor; camera motion may arrive as keys, pose JSON,
Numpy arrays, or integrated tensors. The semantic input name is still the main
contract.

Fourth, schema objects need open-ended metadata for future adapters. This lets a
SANA-WM-like adapter advertise that `camera_trajectory_c2w` uses an
`[F,4,4]` OpenCV camera-to-world sequence, or lets another model advertise a
schema URI, units, coordinate frame, accepted file suffixes, cardinality hints,
or update notes. Metadata should remain query information and should not become
the compatibility type system.

Fifth, `UserInputSchema` describes raw source capabilities, `CanonicalModality`
describes what an application consumes, and mapping schemas describe derived
model-facing semantics. A browser may provide `key_down`, `key_up`,
`prompt_set`, and `initial_frame_set` events. Those become canonical modalities
such as `driver_command` or `conditioning_prompt`; whether they can then drive
`steering`, `camera_trajectory`, or text embedding updates depends on the
selected mapping and model schema.

## Implemented T2/T3 Shape

The implementation that came out of this inventory is:

1. Keep `UserInputEvent` and `UserInputs` as the raw event API, sliced by a
   half-open `TimeWindow`. Static startup values remain timestamp-zero events.
2. Keep `UserInputSchema` lightweight and source-facing. `event_types` declares
   that an event type exists; `UserInputCapability` additionally pins the
   payload fields it carries.
3. Add a canonical layer between raw and encoded. `CanonicalModality` names a
   device-independent input and its conditioning phase; `InputCanonicalizer`
   registers per-device converters and produces `CanonicalInputs`. Applications
   and mappings consume canonical inputs and never read raw device events.
4. Split `InferenceInput` (formerly `ModelInputs`) into `global_conditioning`
   and `step`. A non-empty global slot mid-rollout is an update request, not a
   reset; `InputField.update_policy` declares whether the model can apply it.
5. Extend `InputField` with `update_policy`, `lifecycle`, and `metadata` so
   models can distinguish runtime config, cache initialization, rollout binding,
   per-step inputs, and supported active-session updates.
6. Keep `InputMappingSchema` as the canonical-to-encoded boundary, with
   mapping-set compatibility helpers for composed mappings.
7. Keep input names, semantic types, lifecycle labels, and metadata open-ended.
   Adding a new model should usually mean adding adapter-owned schema
   declarations and mappings, not changing the core input dataclasses.
8. Leave deep validation to model adapters, sessions, and mappings. The schema
   layer catches obvious source/mapping/model mismatches before expensive
   runtime initialization; it does not validate every tensor and coordinate
   convention.

See `docs/inference_runtime_inputs_implementation.md` for the resulting API.

## Extensibility Contract

The inventory above is not a vocabulary freeze. The core API does not contain a
closed enum of allowed input names. New adapters can introduce semantic field
names that match the model boundary they own.

Use these conventions when adding future model schemas:

- Prefer semantic names over modality names, such as `camera_trajectory_c2w`
  instead of `array`, or `hdmap_frames` instead of `image`.
- Use `semantic_type` for a coarse representation hint, such as `path`,
  `decoded_tensor`, `c2w_sequence`, `intrinsics_vec4_sequence`, or `embedding`.
- Use `lifecycle` to say where the adapter consumes the value, such as
  `runtime_config`, `cache_init`, `rollout_binding`, `step_input`, or
  `session_update`.
- Use `update_policy` to say when a value may change. `SESSION_START_ONLY` is
  the one reserved token, meaning the value cannot be swapped mid-rollout.
- Use `metadata` for query hints: units, coordinate frame, shape summary,
  accepted suffixes, schema URI, model family, value ranges, or cardinality.
- Keep deep validation in the adapter/mapping. The lightweight schemas answer
  whether the selected source and mapping can plausibly drive the model before
  expensive initialization.

## Representative Schema Sketches

These are not migration work for T4+, but they show that the current primitives
can describe the supported input surfaces. All use
`flashdreams.runtime.InferenceInputSchema` and `InputField`.

```python
lingbot_model = InferenceInputSchema(
    description="lingbot-world",
    global_fields=(
        InputField(name="prompt", lifecycle="cache_init"),
        InputField(name="global_conditioning_frame", lifecycle="cache_init"),
    ),
    step_fields=(
        InputField(name="camera_trajectory", lifecycle="step_input"),
        InputField(
            name="text_embeddings",
            required=False,
            update_policy="step_boundary",
            lifecycle="session_update",
        ),
    ),
)
```

```python
omnidreams_model = InferenceInputSchema(
    description="omnidreams",
    global_fields=(
        InputField(name="prompts", lifecycle="cache_init"),
        InputField(name="global_conditioning_frames", lifecycle="cache_init"),
        InputField(name="view_names", lifecycle="cache_init"),
        InputField(name="text_embeddings", required=False, lifecycle="cache_init"),
        InputField(name="image_embeddings", required=False, lifecycle="cache_init"),
    ),
    step_fields=(InputField(name="hdmap_frames", lifecycle="step_input"),),
)
```

```python
hy_worldplay_model = InferenceInputSchema(
    description="hy-worldplay",
    global_fields=(
        InputField(name="prompt", lifecycle="cache_init"),
        InputField(name="global_conditioning_frame", lifecycle="cache_init"),
        InputField(name="action_labels", lifecycle="rollout_binding"),
        InputField(name="camera_viewmats", lifecycle="rollout_binding"),
        InputField(name="camera_intrinsics", lifecycle="rollout_binding"),
        InputField(name="memory_config", lifecycle="rollout_binding"),
    ),
)
```

```python
sana_wm_model = InferenceInputSchema(
    description="sana-wm",
    global_fields=(
        InputField(name="prompt", lifecycle="cache_init"),
        InputField(name="negative_prompt", required=False, lifecycle="cache_init"),
        InputField(name="global_conditioning_frame", lifecycle="cache_init"),
        InputField(
            name="camera_trajectory_c2w",
            semantic_type="c2w_sequence",
            lifecycle="rollout_binding",
            metadata={"shape": "[F,4,4]", "coordinates": "opencv_c2w"},
        ),
        InputField(
            name="camera_intrinsics_vec4",
            required=False,
            semantic_type="intrinsics_vec4_sequence",
            lifecycle="rollout_binding",
            metadata={"shape": "[F,4]"},
        ),
    ),
)
```

SANA-WM's `stage1_sampling` and `streaming_chunking` are deliberately absent
above. They describe how to run the model rather than what conditions it, so
they belong in `InferenceConfig`, not in an input schema. Flagged here because
the runner currently threads them alongside the conditioning inputs.
