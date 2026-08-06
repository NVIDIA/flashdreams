<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Inference Runtime Inputs Implementation Notes

This note documents the input layers of the experimental runtime API: what
exists, how the pieces fit together, what the compatibility query answers, and
what is intentionally still outside this layer.

Implementation lives in `flashdreams.runtime`:

- `flashdreams/flashdreams/runtime/inputs.py` — the input types and schemas
- `flashdreams/flashdreams/runtime/canonical.py` — raw device to canonical
  modality conversion
- `flashdreams/flashdreams/runtime/mapping.py` — canonical to encoded mapping
  and compatibility
- `flashdreams/tests/test_runtime_canonical.py`
- `flashdreams/tests/test_runtime_input_mapping.py`
- `flashdreams/tests/test_inference_runtime_api.py` — the T1 envelope tests,
  including a reference loop that exercises all three layers

The supported-model input inventory that informed this work is in
`docs/inference_runtime_supported_inputs_inventory.md`.

## The Three Layers

```text
UserInputs  ──InputCanonicalizer──▶  CanonicalInputs  ──InputMapping──▶  InferenceInput
   raw                                canonicalized                        encoded
(device events)                    (device-independent)              (what the session gets)
```

| Layer | Type | Owner | Example |
| --- | --- | --- | --- |
| raw | `UserInputs` / `UserInputEvent` | transport, replay loader, benchmark driver | `key_down {"key": "w"}`, wheel axis reading |
| canonicalized | `CanonicalInputs` | device converters registered on `InputCanonicalizer` | `driver_command {throttle, brake, steer, ...}` |
| encoded | `InferenceInput` | the selected `InputMapping` | whatever the model's session consumes |

Applications and mappings consume `CanonicalInputs`. They never read raw device
events: `InputMapping.map_step_inputs` takes `canonical_inputs`, not
`user_inputs`, so this is enforced by the signature rather than by convention.
Adding a keyboard, gamepad, or wheel is an `InputCanonicalizer.register` call
that touches no application, mapping, or model code.

This path covers **live user control only**. Global conditioning is
application-owned data and reaches `InferenceInput` directly, without passing
through canonicalization or a device converter. Session start/reset establishes
that global conditioning. During an active rollout, a non-empty
`global_conditioning` payload passed to `step()` requests an update of the
session-global state when the model supports it.

## Conditioning Slots

The encoded layer splits model-facing inputs into two slots:

- **global conditioning** — session-global model state: prompt, conditioning
  frame, scene.
- **per-step conditioning** — needed to generate the next chunk or frame:
  steering, HD map frames, camera trajectory.

`InputPhase` is `Literal["global_conditioning", "step"]`. The phase names the
`InferenceInput` slot the caller provides.

`InputField.frequency_consumed` is independent query metadata. It says how the
adapter consumes a field internally, such as `once` or `per_step`; it does not
decide whether the caller provides the field through `global_conditioning` or
`step`.

## Global Conditioning Is Session-Global State

`InferenceInput.global_conditioning` carries session-scoped inputs. A runtime
passes those values to `InferenceRuntime.start_session()` or to
`InferenceSession.reset()` when the backend supports resetting a rollout.
During an active rollout, passing a non-empty `global_conditioning` payload to
`InferenceSession.step()` asks the session to update that session-global state.
The model/session owns whether that update is supported.

```python
from flashdreams.runtime import InferenceInput, InferenceInputSchema, InputField

schema = InferenceInputSchema(
    global_conditioning_fields=(
        InputField(name="prompt"),
        InputField(name="scene_id"),
    )
)
schema.require_global_conditioning(
    InferenceInput(global_conditioning={"prompt": "drive", "scene_id": "town_02"})
)

step_with_prompt_update = InferenceInput(
    global_conditioning={"prompt": "heavy rain"},
    step={"steering": 0.0},
)
```

Per-step conditioning is different: those values are supplied through
`InferenceInput.step` for each generated chunk or frame. Converters still emit
every window, because live control is level-triggered: a key held across a step
emits no events but still means full throttle.

## Raw Inputs

`UserInputEvent` carries `timestamp_s`, `event_type`, `payload`, `source`, and
`source_event_id`. `UserInputs` holds an ordered batch plus a `snapshot` and
`metadata`, and slices to a half-open `TimeWindow`:

```python
from flashdreams.runtime import TimeWindow, UserInputEvent, UserInputs

inputs = UserInputs(
    events=(
        UserInputEvent(timestamp_s=0.0, event_type="prompt_set",
                       payload={"prompt": "drive forward"}),
        UserInputEvent(timestamp_s=0.5, event_type="key_down", payload={"key": "w"}),
    )
)
step_window = inputs.window(TimeWindow(start_s=0.0, end_s=1.0))
```

`UserInputSchema` describes what a transport, replay trace, or benchmark driver
can provide. `event_types` declares only that an event type exists;
`UserInputCapability` additionally pins the payload fields it carries, so a
converter can require `key_down` events that actually have a `key`. A bare
`event_types` entry still satisfies any consumer needing no specific payload
fields, so schemas written before capabilities existed keep working.

## Canonical Modalities

A `CanonicalModality` is a device-independent input: a name and the payload
fields it guarantees. Converters implement `DeviceConverter`, declaring
what raw capabilities they consume and which modality they produce.

```python
from flashdreams.runtime import (
    DRIVER_COMMAND, InputCanonicalizer, KeyboardToDriverCommand, TimeWindow,
)

canonicalizer = InputCanonicalizer([KeyboardToDriverCommand()])
canonicalizer.register(WheelToDriverCommand())      # a wheel is one call

canonical = canonicalizer.canonicalize(
    user_inputs, window=TimeWindow(start_s=0.0, end_s=1.0), source_schema=browser
)
canonical.values["driver_command"]["throttle"]
```

`DRIVER_COMMAND` is the one shipped modality. `KeyboardToDriverCommand` reuses
`KeyboardState`/`normalize_key` from `flashdreams.serving.realtime.input` and
mirrors the semantics the Omnidreams interactive-drive keyboard backend already
has. Its key bindings are data (`DEFAULT_DRIVING_BINDINGS`), and the set of
tracked keys is derived from them, so a rebound layout cannot leave an action
unreachable.

`ScriptedModality` is the mock/replay converter. It consumes no raw
capabilities, so a benchmark or test can author a scenario at the canonical
level without knowing any device vocabulary:

```python
canonicalizer = InputCanonicalizer([
    ScriptedModality(modality=DRIVER_COMMAND, timeline=[(0.0, full_throttle)]),
])
canonicalizer.canonicalize(
    UserInputs(), window=step_window, source_schema=UserInputSchema()
)
```

Application code is identical between a real run and a scripted one.

Converters are stateful, so feed windows in session order and call
`InputCanonicalizer.reset()` at a rollout boundary. Replaying the same window
sequence reproduces the same `CanonicalInputs`.

When several devices produce the same modality, the highest-priority one that
returned a value wins; `CanonicalInputs.metadata["canonical_sources"]` records
which device supplied each. Every feedable converter still sees each window, so
a preempted device's state stays current and unplugging the higher-priority
device does not resume from stale state.

## Mapping And Compatibility

`InputMapping` is the canonical-to-encoded boundary. `InputMappingSchema` is its
declarative surface: `consumes` names canonical modalities;
`produces_global_conditioning` and `produces_step` name the `InferenceInput`
fields it can build.

`InputMapping.validate()` raises, which fails a run late and cannot say *which*
optional model input a source would enable or *which* missing modality makes a
required one unreachable. `check_mapping_compatibility` answers those before
expensive runtime initialization:

```python
from flashdreams.runtime import check_mapping_set_compatibility

compatibility = check_mapping_set_compatibility(
    canonical_schema=canonicalizer.canonical_schema(browser),
    inference_input_schema=adapter.inference_input_schema,
    mapping_schemas=(prompt_mapping, frame_mapping, steering_mapping),
)
if not compatibility.can_drive:
    compatibility.raise_if_incompatible()
```

`MappingCompatibility` reports `missing_modalities`,
`missing_required_model_fields`, `satisfied_required_model_fields`,
`available_optional_model_fields`, and `unavailable_mapping_schemas`.

Compatibility is evaluated per mapping rather than over a flattened bag, so each
mapping keeps its own consumes/produces link. A mapping the source cannot feed
is dropped and reported, costing only the inputs it produced. So a dropped
mapping that fed only optional fields degrades the run instead of vetoing it,
and those fields are correctly absent from `available_optional_model_fields`; a
dropped mapping that was the only producer of a required field still blocks.

Because a mapping consumes modalities rather than raw events, one mapping
written against `driver_command` works for a keyboard, a wheel, or any device
registered later, with no change to the mapping or the model schema.

`undeclared_inference_inputs()` reports payload keys a mapping produced but did
not declare, which keeps hand-written schemas honest as the code drifts.

`StepRequest` and `StepResult` sit around a single `InferenceSession.step()`
call. They are not schema declarations. A session returns `StepRequest` from
`next_step_request()` to name the next step, optionally provide a narrower
`InferenceInputSchema`, and request a `TimeWindow` of user inputs. The runner
then builds `InferenceInput` and calls `step()`, which returns a `StepResult`
for the output target and metrics recorder.

## What This Does Not Validate

The schemas intentionally avoid becoming a rich type system. These remain the
responsibility of the model adapter, runtime, session, or mapping:

- tensor shape and dtype, image decode details;
- camera coordinate systems, pose and timestamp units;
- prompt-embedding mechanics;
- whether a model can actually apply a requested global-conditioning update;
- enforcing consumption-cadence metadata;
- deep validation of scene, HD map, or actor-state data.

The layer answers "can this source plausibly drive this model through this
mapping?" It does not replace model-owned validation.

## Open Questions

Tracked against the runtime API discussion, not yet settled:

- **Alternative valid input combinations.** `InferenceInputSchema` has one flat
  required set, so "accepts `{prompt}` OR `{prompt, conditioning_frame}`" cannot
  be expressed. `MappingCompatibility.missing_required_model_fields` assumes a
  single required set too.
- **`step()` returning a future**, for models with a dependency on their own
  output. `InferenceSession.step()` is currently synchronous.
- **`Input System` ownership.** The diagrams show it pulling events, so the
  Application owns an input system. `InputCanonicalizer` is currently a pure
  function over a supplied window and owns no source. Whether it needs to grow
  one depends on the loop-ownership decision. Mock input and key binding are
  handled (`ScriptedModality`, `DEFAULT_DRIVING_BINDINGS`).

## Owned Elsewhere

Named here only so the boundary is explicit; these are not gaps in the input
layer:

- **`FrameStream`**, which the architecture diagrams place between
  `InferenceSession` and `Output Target`. The code writes `StepResult` straight
  to `OutputTarget.write()`. Output shape is T5.
- **Declared output modalities**, so an output target or quality-eval can state
  what it requires and be matched the way inputs now are. T5/T8.
- **`Application`**, the class that has-a input system, input map, global
  conditioning, session, and output target. T4.
- **Loop ownership** — whether the application or the runtime/session drives the
  main event loop, and whether inputs are queued and batched.

## Validation

```bash
.venv/bin/pytest flashdreams/tests/test_runtime_canonical.py \
  flashdreams/tests/test_runtime_input_mapping.py \
  flashdreams/tests/test_inference_runtime_api.py -q
.venv/bin/ty check flashdreams/flashdreams/runtime
```

At the time of writing these pass: 87 tests, and `ty` is clean.
