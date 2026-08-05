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
through canonicalization or a device converter. An application that wants a
trigger key to swap the prompt reads that as ordinary canonical control input
and updates its own global conditioning in response.

## Conditioning Slots

Both the canonical and encoded layers split into two slots, and the split means
the same thing at each:

- **global conditioning** — conditions the whole rollout: prompt, conditioning
  frame, scene. Normally supplied at session start.
- **per-step conditioning** — needed to generate the next chunk or frame:
  steering, HD map frames, camera trajectory.

`InputPhase` is `Literal["global", "step"]`. The axis names *which slot*, not
*when the value may arrive* — see the next section.

## Global Conditioning Updates Are Not Resets

A non-empty global slot on a mid-rollout `InferenceInput` is an **update
request**. The session should apply it when the model supports doing so.
Resetting rollout state is a separate, explicit `InferenceSession.reset()` call.
The motivating case is changing prompt and conditioning frame mid-run to change
the weather in an Omnidreams rollout.

```python
from flashdreams.runtime import InferenceInput

steady_state = InferenceInput(step={"steering": 0.25})
assert not steady_state.requests_global_update

changed_weather = steady_state.with_global_update({"prompt": "heavy rain"})
assert changed_weather.requests_global_update
```

Because `with_step()` carries the global slot through unchanged, use
`without_global_update()` for the steady-state case; otherwise every step looks
like an update request.

Whether a value can actually be swapped mid-rollout is declared per field:

```python
from flashdreams.runtime import SESSION_START_ONLY, InferenceInputSchema, InputField

schema = InferenceInputSchema(
    global_fields=(
        InputField(name="prompt", update_policy="step_boundary"),
        InputField(name="scene_id", update_policy=SESSION_START_ONLY),
    )
)
schema.unsupported_global_updates(
    InferenceInput(global_conditioning={"prompt": "heavy rain", "scene_id": "town_02"})
)
# ("scene_id",)
```

`SESSION_START_ONLY` is the one reserved `update_policy` token. Everything else
in that vocabulary, and all of `lifecycle`, is open and adapter-owned; this layer
only carries it as queryable metadata.

Steady-state steps must leave the global slot empty; otherwise every step reads
as an update request. Converters emit every window, because live control is
level-triggered: a key held across a step emits no events but still means full
throttle.

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
declarative surface: `consumes` names canonical modalities; `produces_global`
and `produces_step` name the `InferenceInput` fields it can build.

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

## What This Does Not Validate

The schemas intentionally avoid becoming a rich type system. These remain the
responsibility of the model adapter, runtime, session, or mapping:

- tensor shape and dtype, image decode details;
- camera coordinate systems, pose and timestamp units;
- prompt-embedding swap mechanics;
- whether a model can actually apply a declared update policy at runtime;
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
