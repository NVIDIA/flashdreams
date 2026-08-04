<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Inference Runtime Inputs API Implementation Notes

This note documents the current T2/T3 implementation of the inference runtime
input contracts. It is meant as an evaluator's guide: what exists, how the
pieces fit together, what the compatibility query answers, and what is
intentionally still outside this layer.

Implementation lives in `flashdreams.inference`:

- `flashdreams/flashdreams/inference/__init__.py`
- `flashdreams/flashdreams/inference/inputs.py`
- `flashdreams/tests/test_inference_inputs.py`

The current supported-model input inventory that informed this revision is in
`docs/inference_runtime_supported_inputs_inventory.md`.

## What This Implements

The implementation covers the T2/T3 contract from
`docs/inference_runtime_api_design.md`:

- `UserInputs` are represented as timestamped events.
- `UserInputSchema` describes what an app, transport, replay trace, or
  benchmark source can provide.
- `ModelInputs` are semantic model-facing payloads split into initial and
  per-step inputs.
- `ModelInputSchema` describes what a model or session requires or optionally
  accepts.
- `InputMapper` is the conversion contract between user events and model inputs.
- Schema objects support open-ended `metadata` maps for lightweight hints that
  future model adapters can expose without changing the core API.
- `check_mapping_compatibility()` answers whether a selected source can drive a
  selected model through a selected mapper before expensive runtime setup.
- `check_mapping_set_compatibility()` answers the same question for a composed
  set of mapper schemas, such as prompt plus first-frame plus live-control
  mappings.

This does not migrate LingBot, OmniDreams, WebRTC, CLI runners, or the standard
runtime/session loop. Those are T4+ and migration tasks.

## User Inputs

`UserInputEvent` is the primary user-input representation. Every user-facing
input is modeled as an event in session time, including static startup values.

Examples:

- `prompt_set` at `timestamp_s=0.0`
- `initial_frame_set` at `timestamp_s=0.0`
- `scene_selected` at `timestamp_s=0.0`
- `key_down` / `key_up` during a session
- `controller_axis` during a session
- `camera_pose` events in a replay trace

`UserInputTrace` stores events in deterministic timestamp order and can slice a
`UserInputWindow` for one runtime step. Events with equal timestamps preserve
their original order. A `UserInputWindow` enforces its own invariant: every
event it holds must fall inside `[start_s, end_s]`, so a directly constructed
window cannot silently disagree with its bounds.

```python
from flashdreams.inference import UserInputEvent, UserInputTrace

trace = UserInputTrace.from_events(
    [
        UserInputEvent(
            timestamp_s=0.0,
            kind="prompt_set",
            payload={"prompt": "drive forward"},
        ),
        UserInputEvent(
            timestamp_s=0.5,
            kind="key_down",
            payload={"key": "w"},
        ),
    ]
)

step_window = trace.window(start_s=0.0, end_s=1.0)
```

Snapshots are intentionally not primary inputs. A mapper or runtime can derive
snapshots from a trace/window when a model wants snapshot-style controls.

## User Input Schema

`UserInputSchema` is lightweight metadata for the source side. It answers:

- Which event kinds can this source provide?
- Which payload fields are present on those events?
- Is this source live, replayed, fixed, or otherwise identified?

It does not assign model semantics. For example, `key_down` does not mean
steering or camera motion until a mapper says so.

```python
from flashdreams.inference import UserInputCapability, UserInputSchema

browser_schema = UserInputSchema(
    name="browser",
    source_kind="live",
    metadata={"transport": "webrtc"},
    capabilities=(
        UserInputCapability(
            event_kind="prompt_set",
            payload_kind="text",
            payload_fields=frozenset({"prompt"}),
        ),
        UserInputCapability(
            event_kind="key_down",
            payload_fields=frozenset({"key"}),
        ),
        UserInputCapability(
            event_kind="key_up",
            payload_fields=frozenset({"key"}),
        ),
    ),
)
```

The schema can validate that a source event has an event kind and payload fields
the source claims to provide. It does not validate tensor shape, image format,
camera coordinate system, or model-specific units.

Capabilities and schemas also carry optional `metadata`. This is for query-time
hints such as source type, UI widget, file suffixes, units, coordinate frame, or
schema URI. Metadata is intentionally not part of compatibility matching.
Because metadata is an open-ended pass-through, its keys are validated (they
must be non-empty strings) but never rewritten, so a key round-trips unchanged.

## Model Inputs

`ModelInputs` is the model-facing payload container. It separates values needed
to start or reset a rollout from values needed for one generated step/chunk.

```python
from flashdreams.inference import ModelInputs

inputs = ModelInputs(
    initial={"prompt": "drive forward", "first_frame": first_frame},
    step={"steering": 0.25},
)
```

`ModelInputSchema` declares the semantic fields a model or session expects. It
uses names like `prompt`, `first_frame`, `steering`, `camera_trajectory`, or
`hdmap_frames`, rather than generic modality-only keys.

```python
from flashdreams.inference import ModelInputField, ModelInputSchema

model_schema = ModelInputSchema(
    name="driving-model",
    fields=(
        ModelInputField(
            name="prompt",
            phase="initial",
            required=True,
            payload_kind="text",
            update_policy="step_boundary",
            lifecycle="cache_init",
        ),
        ModelInputField(
            name="steering",
            phase="step",
            required=True,
            lifecycle="step_input",
        ),
        ModelInputField(
            name="first_frame",
            phase="initial",
            required=False,
            lifecycle="cache_init",
        ),
        ModelInputField(
            name="camera_trajectory_c2w",
            phase="initial",
            required=False,
            payload_kind="c2w_sequence",
            lifecycle="rollout_binding",
            metadata={"shape": "[F,4,4]", "coordinates": "opencv_c2w"},
        ),
    ),
    metadata={"model_family": "example-driving-model"},
)
```

`update_policy` is deliberately plain metadata. It lets a model advertise facts
such as "prompt updates can happen at step boundaries" without making this
schema layer responsible for implementing or deeply validating that behavior.

`lifecycle` is also plain metadata. It lets a model distinguish initial values
used at different adapter moments, such as `runtime_config`, `cache_init`,
`rollout_binding`, `step_input`, or `session_update`. Compatibility requires
lifecycle agreement only when both the model field and mapper output specify a
lifecycle; otherwise simple schemas remain permissive.

Model input names, payload kinds, lifecycle labels, and metadata are
open-ended. They are not a FlashDreams-wide enum. A SANA-WM-like adapter can
declare fields such as `camera_trajectory_c2w`, `camera_intrinsics_vec4`,
`stage1_sampling`, or `streaming_chunking`; another model can declare different
semantic names. The adapter and mapper own the deep interpretation.

## Input Mappers

`InputMapper` is a protocol with three responsibilities:

- expose an `InputMapperSchema`;
- build initial `ModelInputs` from a `UserInputTrace`;
- build per-step `ModelInputs` from a `UserInputWindow`.

`InputMapperSchema` declares the mapper's compatibility surface:

- `consumes`: user event capabilities required by the mapper;
- `produces`: model input fields the mapper can produce.

Example: keyboard events can be mapped into steering for one model, camera
trajectory for another model, or ignored entirely. That meaning is owned by the
mapper, not by `UserInputEvent`.

```python
from flashdreams.inference import (
    InputMapperSchema,
    ModelInputField,
    UserInputCapability,
)

keyboard_to_steering_schema = InputMapperSchema(
    name="keyboard-to-steering",
    consumes=(
        UserInputCapability(
            event_kind="key_down",
            payload_fields=frozenset({"key"}),
        ),
        UserInputCapability(
            event_kind="key_up",
            payload_fields=frozenset({"key"}),
        ),
    ),
    produces=(
        ModelInputField(name="steering", phase="step"),
    ),
)
```

For fixed runs that already have model-facing inputs, `StaticInputMapper`
provides the no-live-controls path. It consumes no user events and returns the
configured initial/per-step `ModelInputs`.

```python
from flashdreams.inference import ModelInputs, StaticInputMapper

static_mapper = StaticInputMapper.from_inputs(
    inputs=ModelInputs(
        initial={"prompt": "fixed prompt"},
        step={"camera_trajectory": camera_poses},
    ),
    name="fixed-scenario",
)
```

A mapper schema is a hand-written declaration, so it can drift from what
`build_initial_inputs` / `build_step_inputs` actually return.
`undeclared_model_inputs()` reports payload keys a mapper produced but did not
declare, which keeps mapper tests honest about the compatibility surface.

Mapper schemas can also be combined for compatibility checks. This matches the
current supported model inventory: a run may select one mapping for prompt
events, another for first-frame events, and another for keyboard or controller
events.

```python
from flashdreams.inference import check_mapping_set_compatibility

compatibility = check_mapping_set_compatibility(
    source_schema=browser_schema,
    model_schema=lingbot_schema,
    mapper_schemas=(
        prompt_mapper.schema,
        first_frame_mapper.schema,
        keyboard_to_camera_mapper.schema,
    ),
)
```

## Compatibility Query

`check_mapping_compatibility()` is the central query for T2/T3. Given a source
schema, model schema, and mapper schema, it reports:

- whether the source can drive the model through this mapper;
- source capabilities the mapper needs but the source lacks;
- required model fields the mapper cannot produce;
- required model fields that are satisfied;
- optional model fields that can be enabled;
- mappers dropped because the source cannot feed them.

```python
from flashdreams.inference import check_mapping_compatibility

compatibility = check_mapping_compatibility(
    source_schema=browser_schema,
    model_schema=model_schema,
    mapper_schema=keyboard_to_steering_schema,
)

if not compatibility.can_drive:
    compatibility.raise_if_incompatible()
```

This is an early compatibility check. It is intended to fail before expensive
runtime initialization when the mismatch is obvious. It is not a guarantee that
the model run will succeed.

Use `check_mapping_set_compatibility()` when the selected mapping is composed
from multiple mapper schemas. It returns the same `MappingCompatibility` report
for the composed mapping.

Compatibility is evaluated per mapper rather than over a flattened bag of
capabilities, so each mapper keeps its own `consumes`/`produces` link. A mapper
the source cannot feed is dropped and reported in `unavailable_mapper_schemas`;
it costs only the model inputs that mapper produced. This means:

- a dropped mapper that produced only optional fields degrades the run instead
  of vetoing it, and those fields are correctly absent from
  `available_optional_model_fields`;
- a dropped mapper that was the only producer of a required field still blocks,
  and its unmet source capabilities are reported in
  `missing_source_capabilities`;
- `satisfied_required_model_fields` and `available_optional_model_fields` name
  only fields this source can really produce, not every field some mapper
  declared.

## What This Does Not Validate

The schemas intentionally avoid becoming a rich type system. The following
remain the responsibility of the model adapter, runtime, session, or mapper
implementation:

- tensor shape and dtype;
- image decode details;
- camera coordinate systems;
- pose and timestamp units;
- prompt-embedding swap mechanics;
- reset semantics;
- whether a specific model can actually apply an update policy at runtime;
- deep validation of scene, HD map, or actor-state data.

The schema layer should be enough to answer "can this source plausibly drive
this model through this mapper?" It should not replace model-owned validation.

## Current Tests

The focused CPU tests are in `flashdreams/tests/test_inference_inputs.py`.
They cover:

- deterministic event ordering and trace windowing;
- startup values represented as events;
- source capability declaration and basic event validation;
- required and optional model input declarations;
- prompt update metadata on a model field;
- lifecycle metadata on model fields and mapper outputs;
- open-ended schema metadata that stays queryable but does not constrain
  compatibility;
- missing required model inputs;
- missing source capabilities;
- optional model inputs becoming available only when mapper support exists;
- composed mapper-set compatibility;
- graceful degradation when an optional mapper cannot be fed, and blocking when
  a required one cannot;
- metadata merging when duplicate declarations collapse during mapper-set
  combination;
- event hashability, window bound enforcement, and metadata key round-tripping;
- mappers producing only model inputs their schema declares;
- SANA-WM-like model-specific inputs without importing the SANA integration;
- fake prompt, initial-frame, keyboard-to-steering, and camera-trajectory
  mappers;
- `StaticInputMapper` for fixed model-input scenarios.

Targeted validation command:

```bash
.venv/bin/pytest flashdreams/tests/test_inference_inputs.py -q
.venv/bin/ty check flashdreams/flashdreams/inference flashdreams/tests/test_inference_inputs.py
.venv/bin/python -m py_compile \
  flashdreams/flashdreams/inference/__init__.py \
  flashdreams/flashdreams/inference/inputs.py \
  flashdreams/tests/test_inference_inputs.py
```

At the time this note was written, these targeted checks passed.

## Evaluation Checklist

Use this checklist to judge whether the implementation matches the T2/T3 design:

- Are all `UserInputs` represented as timestamped events?
- Can a source declare what event capabilities it provides?
- Can a model declare required and optional initial/per-step inputs?
- Are model input names semantic rather than modality-only?
- Can new models add model-specific field names and metadata without changing
  core dataclasses?
- Can a mapper declare what it consumes and what it produces?
- Can multiple mapper schemas be checked as one selected mapping surface?
- Does compatibility checking report both missing source capabilities and
  missing required model inputs?
- Are optional model inputs reported separately from required inputs?
- Can model fields distinguish cache initialization, rollout binding, per-step
  inputs, and active-session update support without deep tensor validation?
- Is deep model/tensor validation kept out of the lightweight schema layer?
- Is fixed input/model-input replay possible without live user events?
