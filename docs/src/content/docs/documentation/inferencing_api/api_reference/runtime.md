---
title: 'Experimental Inference Runtime API Reference'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

`flashdreams.runtime` is the experimental inference API. Its lifecycle is
`ModelAdapter` → reusable `InferenceRuntime` → isolated
`InferenceSession`. See the [Demo API overview](../../demo_api/index.md) for
the application layer above it, or the
[Inferencing API overview](../index.md) for the surrounding model-side guides
and references.

## Minimal direct use

Integrations provide the adapter and its model-specific input schema.

```python

from flashdreams.runtime import (
    InferenceConfig,
    InferenceInput,
    ModelAdapter,
)

def run_one_step(adapter: ModelAdapter, global_conditioning, step):
    config = InferenceConfig(model_id=adapter.model_id)
    adapter.validate_config(config)
    runtime = adapter.create_runtime(config)
    try:
        initial_input = InferenceInput(global_conditioning=global_conditioning)
        session = runtime.start_session(initial_input)
        try:
            request = session.next_step_request()
            return None if request is None else session.step(InferenceInput(step=step))
        finally:
            session.close()
    finally:
        runtime.close()

```

Close sessions and runtimes even after errors. Session calls are sequential;
the caller owns concurrency unless an integration says otherwise.

## API Reference

### Runtime lifecycle

<a id="flashdreams.runtime.InferenceConfig"></a>
### `InferenceConfig`

```text
class InferenceConfig ( * , model_id: str , preset_id: str | None = None , checkpoint: str | ~pathlib.Path | None = None , backend: ~typing.Literal['local' , 'local-distributed' , 'external' , 'hosted'] = 'local' , device: str | None = None , precision: ~typing.Literal['auto' , 'fp32' , 'fp16' , 'bf16'] = 'auto' , compile: bool | None = None , cuda_graph: bool | None = None , attention_backend: str | None = None , cache_policy: str | None = None , seed: int | None = None , runtime_options: ~collections.abc.Mapping[str , ~typing.Any] = <factory> , resource_hints: ~collections.abc.Mapping[str , ~typing.Any] = <factory> )
```

Bases: `object`

Runtime settings that affect model execution.

Prompts, user controls, browser settings, output paths, and benchmark
directories intentionally live outside this object. The typed optimization
fields cover common cross-backend knobs; open-ended adapter-specific choices
can use `runtime_options`.

<a id="flashdreams.runtime.InferenceConfig.model_id"></a>
#### `model_id`

```text
model_id : str
```

Stable identity for the model adapter or runtime integration.
<a id="flashdreams.runtime.InferenceConfig.preset_id"></a>
#### `preset_id`

```text
preset_id : str | None
```

Optional preset identity under `model_id`.
<a id="flashdreams.runtime.InferenceConfig.checkpoint"></a>
#### `checkpoint`

```text
checkpoint : str | Path | None
```

Optional checkpoint or model-asset selector understood by the adapter.
<a id="flashdreams.runtime.InferenceConfig.backend"></a>
#### `backend`

```text
backend : Literal [ 'local' , 'local-distributed' , 'external' , 'hosted' ]
```

Execution placement and backend family for inference compute.
<a id="flashdreams.runtime.InferenceConfig.device"></a>
#### `device`

```text
device : str | None
```

Optional device selector such as `cuda` or `cuda:0`; `None` leaves placement to the adapter/backend.
<a id="flashdreams.runtime.InferenceConfig.precision"></a>
#### `precision`

```text
precision : Literal [ 'auto' , 'fp32' , 'fp16' , 'bf16' ]
```

Preferred compute precision.
<a id="flashdreams.runtime.InferenceConfig.compile"></a>
#### `compile`

```text
compile : bool | None
```

Optional - Whether model compilation is requested or disabled. None means left to the adapter to decide.
<a id="flashdreams.runtime.InferenceConfig.cuda_graph"></a>
#### `cuda_graph`

```text
cuda_graph : bool | None
```

Optional - Whether CUDA graph capture is requested or disabled. None means left to the adapter to decide.
<a id="flashdreams.runtime.InferenceConfig.attention_backend"></a>
#### `attention_backend`

```text
attention_backend : str | None
```

Optional attention implementation selector; `None` leaves the choice to the adapter.
<a id="flashdreams.runtime.InferenceConfig.cache_policy"></a>
#### `cache_policy`

```text
cache_policy : str | None
```

Optional cache policy selector; `None` leaves the choice to the adapter.
<a id="flashdreams.runtime.InferenceConfig.seed"></a>
#### `seed`

```text
seed : int | None
```

Optional seed used when resolving deterministic demo/runtime behavior.
<a id="flashdreams.runtime.InferenceConfig.runtime_options"></a>
#### `runtime_options`

```text
runtime_options : Mapping [ str , Any ]
```

Adapter/backend-specific runtime options.
<a id="flashdreams.runtime.InferenceConfig.resource_hints"></a>
#### `resource_hints`

```text
resource_hints : Mapping [ str , Any ]
```

Resource hints for launchers, schedulers, or hosted backends.

<a id="flashdreams.runtime.ModelAdapter"></a>
### `ModelAdapter`

```text
class ModelAdapter ( * args , ** kwargs )
```

Bases: `Protocol`

Model-specific boundary that declares defaults and creates runtimes.

Adapters declare model-facing input requirements, the canonical modalities
their default mapping consumes, and an optional default mapping between the
two. Runtime, application, or benchmark code may override that mapping while
preserving the same `CanonicalInputs` to `InferenceInput` boundary.

<a id="flashdreams.runtime.ModelAdapter.model_id"></a>
#### `model_id`

```text
property model_id : str
```

Stable identity for the model adapter or runtime integration.
<a id="flashdreams.runtime.ModelAdapter.inference_input_schema"></a>
#### `inference_input_schema`

```text
property inference_input_schema : InferenceInputSchema
```

Model-facing global conditioning and per-step input requirements.
<a id="flashdreams.runtime.ModelAdapter.canonical_input_schema"></a>
#### `canonical_input_schema`

```text
property canonical_input_schema : CanonicalInputSchema | None
```

Canonical modalities the adapter’s default mapping consumes.
<a id="flashdreams.runtime.ModelAdapter.default_input_mapping"></a>
#### `default_input_mapping`

```text
default_input_mapping ( ) → InputMapping | None
```

Return the model-provided default canonical-to-model mapping.
<a id="flashdreams.runtime.ModelAdapter.validate_config"></a>
#### `validate_config`

```text
validate_config ( config : InferenceConfig ) → None
```

Fail early for unsupported runtime settings.
<a id="flashdreams.runtime.ModelAdapter.create_runtime"></a>
#### `create_runtime`

```text
create_runtime ( config : InferenceConfig ) → InferenceRuntime
```

Initialize and return the heavyweight runtime.

<a id="flashdreams.runtime.InferenceRuntime"></a>
### `InferenceRuntime`

```text
class InferenceRuntime ( * args , ** kwargs )
```

Bases: `Protocol`

Heavyweight reusable runtime created from `InferenceConfig`.

<a id="flashdreams.runtime.InferenceRuntime.start_session"></a>
#### `start_session`

```text
start_session ( inputs : InferenceInput ) → InferenceSession
```

Create an isolated session from global conditioning inputs.
<a id="flashdreams.runtime.InferenceRuntime.close"></a>
#### `close`

```text
close ( ) → None
```

Release model/backend resources.

<a id="flashdreams.runtime.InferenceSession"></a>
### `InferenceSession`

```text
class InferenceSession ( * args , ** kwargs )
```

Bases: `Protocol`

One rollout or stream with isolated model/cache state.

<a id="flashdreams.runtime.InferenceSession.next_step_request"></a>
#### `next_step_request`

```text
next_step_request ( ) → StepRequest | None
```

Return the next step’s runtime request, or `None` when complete.
<a id="flashdreams.runtime.InferenceSession.step"></a>
#### `step`

```text
step ( inputs : InferenceInput ) → StepResult
```

Run one sequential inference step.
<a id="flashdreams.runtime.InferenceSession.reset"></a>
#### `reset`

```text
reset ( inputs : InferenceInput | None = None ) → None
```

Reset this session’s rollout state when the backend supports it.
<a id="flashdreams.runtime.InferenceSession.close"></a>
#### `close`

```text
close ( ) → None
```

Release per-session resources.

### Inputs and results

<a id="flashdreams.runtime.InferenceInput"></a>
### `InferenceInput`

```text
class InferenceInput ( * , global_conditioning: ~collections.abc.Mapping[str , ~typing.Any] = <factory> , step: ~collections.abc.Mapping[str , ~typing.Any] = <factory> , metadata: ~collections.abc.Mapping[str , ~typing.Any] = <factory> )
```

Bases: `object`

Encoded inputs for one `InferenceSession` call.

Two conditioning slots:

* `global_conditioning`: values that condition the whole rollout, such as
  the conditioning frame or prompt. Session start/reset establishes this
  state; a step call may carry a non-empty payload to request an update when
  the model supports it.
* `step`: values needed to generate the next chunk or frame.

<a id="flashdreams.runtime.InferenceInput.for_phase"></a>
#### `for_phase`

```text
for_phase ( phase : Literal [ 'global_conditioning' , 'step' ] ) → Mapping [ str , Any ]
```

Return the payload mapping for `phase`.

<a id="flashdreams.runtime.InferenceInputSchema"></a>
### `InferenceInputSchema`

```text
class InferenceInputSchema ( * , global_conditioning_fields : tuple [ InputField , ... ] = () , step_fields : tuple [ InputField , ... ] = () , description : str = '' )
```

Bases: `object`

Minimal metadata for global conditioning and per-step inputs.

<a id="flashdreams.runtime.InferenceInputSchema.global_conditioning_fields"></a>
#### `global_conditioning_fields`

```text
global_conditioning_fields : tuple [ InputField , ... ]
```

Model inputs carried in the global conditioning slot.
<a id="flashdreams.runtime.InferenceInputSchema.step_fields"></a>
#### `step_fields`

```text
step_fields : tuple [ InputField , ... ]
```

Model inputs required for one session step.
<a id="flashdreams.runtime.InferenceInputSchema.fields_for"></a>
#### `fields_for`

```text
fields_for ( phase : Literal [ 'global_conditioning' , 'step' ] ) → tuple [ InputField , ... ]
```

Return every declared field for `phase`.
<a id="flashdreams.runtime.InferenceInputSchema.required_fields"></a>
#### `required_fields`

```text
required_fields ( phase : Literal [ 'global_conditioning' , 'step' ] | None = None ) → tuple [ tuple [ Literal [ 'global_conditioning' , 'step' ] , InputField ] , ... ]
```

Return required fields as `(phase, field)`, optionally filtered.
<a id="flashdreams.runtime.InferenceInputSchema.optional_fields"></a>
#### `optional_fields`

```text
optional_fields ( phase : Literal [ 'global_conditioning' , 'step' ] | None = None ) → tuple [ tuple [ Literal [ 'global_conditioning' , 'step' ] , InputField ] , ... ]
```

Return optional fields as `(phase, field)`, optionally filtered.
<a id="flashdreams.runtime.InferenceInputSchema.field_for"></a>
#### `field_for`

```text
field_for ( * , name : str , phase : Literal [ 'global_conditioning' , 'step' ] ) → InputField | None
```

Return one declared field, if present.
<a id="flashdreams.runtime.InferenceInputSchema.missing_global_conditioning"></a>
#### `missing_global_conditioning`

```text
missing_global_conditioning ( inputs : InferenceInput ) → tuple [ str , ... ]
```

Return required global conditioning fields absent from `inputs`.
<a id="flashdreams.runtime.InferenceInputSchema.missing_step"></a>
#### `missing_step`

```text
missing_step ( inputs : InferenceInput ) → tuple [ str , ... ]
```

Return required per-step fields absent from `inputs`.
<a id="flashdreams.runtime.InferenceInputSchema.require_global_conditioning"></a>
#### `require_global_conditioning`

```text
require_global_conditioning ( inputs : InferenceInput ) → None
```

Raise if required global conditioning fields are absent.
<a id="flashdreams.runtime.InferenceInputSchema.require_step"></a>
#### `require_step`

```text
require_step ( inputs : InferenceInput ) → None
```

Raise if required per-step fields are absent.

<a id="flashdreams.runtime.StepRequest"></a>
### `StepRequest`

```text
class StepRequest ( * , step_index: int , inference_input_schema: ~flashdreams.runtime.inputs.InferenceInputSchema | None = None , user_input_window: ~flashdreams.infra.time.TimeWindow | None = None , metadata: ~collections.abc.Mapping[str , ~typing.Any] = <factory> )
```

Bases: `object`

Per-step runtime request emitted by an inference session.

This is not a schema declaration. `user_input_window` lets a runner drain
or slice timestamped user events for the current step before invoking the
selected `InputMapping`.

<a id="flashdreams.runtime.StepResult"></a>
### `StepResult`

```text
class StepResult ( * , step_index: int , output: ~typing.Any | None = None , frame_count: int = 0 , layout: ~typing.Literal['tchw' , 'btchw' , 'bcthw' , 'bvtchw'] | None = None , output_window: ~flashdreams.infra.time.TimeWindow | None = None , metadata: ~collections.abc.Mapping[str , ~typing.Any] = <factory> , metrics: ~collections.abc.Mapping[str , float | int] = <factory> )
```

Bases: `object`

Generated output and metadata returned by one inference step.

Video results use `from_video_chunk()`, which records a required tensor
layout and derives the frame count once. Non-video results may use the
regular constructor without a layout.

<a id="flashdreams.runtime.StepResult.from_video_chunk"></a>
#### `from_video_chunk`

```text
classmethod from_video_chunk ( * , step_index : int , video_chunk : Tensor , layout : Literal [ 'tchw' , 'btchw' , 'bcthw' , 'bvtchw' ] , output_window : TimeWindow | None = None , metadata : Mapping [ str , Any ] | None = None , metrics : Mapping [ str , float | int ] | None = None ) → StepResult
```

Build one layout-aware generated-video result.
<a id="flashdreams.runtime.StepResult.video_chunk"></a>
#### `video_chunk`

```text
property video_chunk : Tensor
```

Return the video tensor or fail if this is not a video result.
<a id="flashdreams.runtime.StepResult.lazy_rgb_frames"></a>
#### `lazy_rgb_frames`

```text
lazy_rgb_frames ( * , batch_index : int = 0 , view_index : int = 0 , record_cuda_event : bool = True ) → list [ LazyCudaFrame ]
```

Expose this video result as lazy per-frame RGB handles.
<a id="flashdreams.runtime.StepResult.video_hwc_uint8"></a>
#### `video_hwc_uint8`

```text
video_hwc_uint8 ( * , batch_index : int = 0 , view_index : int = 0 ) → Tensor
```

Return this video result as uint8 `[T,H,W,C]` on its device.

### Input mapping

<a id="flashdreams.runtime.CanonicalInputs"></a>
### `CanonicalInputs`

```text
class CanonicalInputs ( * , values: ~collections.abc.Mapping[str , ~typing.Any] = <factory> , metadata: ~collections.abc.Mapping[str , ~typing.Any] = <factory> )
```

Bases: `object`

Canonicalized user input for one step, keyed by modality name.

Values are level-triggered and normally present every step: a key held down
emits no events but still means full throttle. Global conditioning does not
appear here; it is application-owned and reaches `InferenceInput`
directly.

<a id="flashdreams.runtime.InputMapping"></a>
### `InputMapping`

```text
class InputMapping ( * args , ** kwargs )
```

Bases: `Protocol`

Convert user-facing inputs into model-facing inputs.

A mapping may be supplied by the model adapter as a default or by an
application/runtime override. Step mappings usually receive a timestamped
event window selected by the runner for the current model step or chunk.

<a id="flashdreams.runtime.InputMapping.validate"></a>
#### `validate`

```text
validate ( * , canonical_schema : CanonicalInputSchema | None = None , inference_input_schema : InferenceInputSchema | None = None ) → None
```

Fail early for obvious app, event-source, and model mismatches.
<a id="flashdreams.runtime.InputMapping.map_global_conditioning_inputs"></a>
#### `map_global_conditioning_inputs`

```text
map_global_conditioning_inputs ( * , canonical_inputs : CanonicalInputs , inference_input : InferenceInput ) → InferenceInput
```

Build global conditioning inputs for session start or reset.
<a id="flashdreams.runtime.InputMapping.map_step_inputs"></a>
#### `map_step_inputs`

```text
map_step_inputs ( * , canonical_inputs : CanonicalInputs , inference_input : InferenceInput , request : StepRequest ) → InferenceInput
```

Build model inputs for one session step from the current input window.

<a id="flashdreams.runtime.InputMappingSchema"></a>
### `InputMappingSchema`

```text
class InputMappingSchema ( * , name: str = 'input-mapping' , consumes: tuple[~flashdreams.runtime.inputs.CanonicalModality , ...] = () , produces_global_conditioning: tuple[~flashdreams.runtime.inputs.InputField , ...] = () , produces_step: tuple[~flashdreams.runtime.inputs.InputField , ...] = () , metadata: ~collections.abc.Mapping[str , ~typing.Any] = <factory> )
```

Bases: `object`

Declarative compatibility surface for one mapping.

`InputMapping.validate` fails a run late and opaquely: it raises, but it
cannot answer which optional model inputs a source would enable, or which
missing user capability is responsible for an unreachable model input. This
schema makes those questions answerable before runtime initialization.

<a id="flashdreams.runtime.InputMappingSchema.produces_for"></a>
#### `produces_for`

```text
produces_for ( phase : Literal [ 'global_conditioning' , 'step' ] ) → tuple [ InputField , ... ]
```

Return the fields this mapping produces for `phase`.
<a id="flashdreams.runtime.InputMappingSchema.can_produce"></a>
#### `can_produce`

```text
can_produce ( phase : Literal [ 'global_conditioning' , 'step' ] , required : InputField ) → bool
```

Return whether this mapping can produce `required` in `phase`.
