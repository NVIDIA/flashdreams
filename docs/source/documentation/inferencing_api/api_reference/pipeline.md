---
title: 'Pipeline API Reference'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

The `flashdreams.infra` package defines the swappable abstractions that
every integration plugs into: a config system, an encoder / diffusion-model /
decoder triple, and the streaming inference pipeline that drives them.

See the [Inferencing API overview](../index.md) for public runtime boundaries.

## Config

Infra component configs are mutable `InstantiateConfig` dataclasses
built via `config.setup()`. `PrintableConfig` provides readable
configuration dumps, while `derive_config` deep-copies an existing
config before applying nested overrides.

<a id="flashdreams.infra.config.PrintableConfig"></a>
### `PrintableConfig`

```text
class PrintableConfig
```

Bases: `object`

Config base class providing a multi-line `__str__` for human-readable dumps.

<a id="flashdreams.infra.config.InstantiateConfig"></a>
### `InstantiateConfig`

```text
class InstantiateConfig ( _target : type )
```

Bases: `PrintableConfig`

Config carrying a `_target` class plus its kwargs, instantiable via `setup`.

<a id="flashdreams.infra.config.InstantiateConfig.setup"></a>
#### `setup`

```text
setup ( ** kwargs : Any ) → Any
```

Instantiate the configured object.

<a id="flashdreams.infra.config.derive_config"></a>
### `derive_config`

```text
derive_config ( base_config : T , ** changes : Any ) → T
```

Deep-copy a base config and apply nested keyword overrides.

Nested `dict` values walk into both dataclass attributes and nested
dicts; leaf values overwrite directly. Raises `KeyError` on unknown
paths.

Example:

```
new_config = derive_config(
    base_config,
    tokenizer=WanVAEInterfaceConfig(checkpoint_path=...),
    dit=dict(len_t=3, checkpoint_path=...),
)
```

## Pipeline

The pipeline composes the model-side streaming inference stages. It
autoregressively generates one chunk at a time by running an optional encoder,
the diffusion model, and an optional decoder, with per-session caches for the
configured streaming components. Call `finalize()` after `generate()` to
advance the diffusion model's autoregressive cache.

<a id="flashdreams.infra.pipeline.StreamInferencePipelineConfig"></a>
### `StreamInferencePipelineConfig`

```text
class StreamInferencePipelineConfig ( * , _target: type[StreamInferencePipeline] = <factory> , name: str , diffusion_model: ~flashdreams.infra.diffusion.model.base.DiffusionModelConfig , decoder: ~flashdreams.infra.decoder.base.DecoderConfig | None = None , encoder: ~flashdreams.infra.encoder.base.EncoderConfig | None = None )
```

Bases: `InstantiateConfig`

Config for the streaming inference pipeline.

Set `encoder=None` when the pipeline has no per-AR-step control input
(pure T2V). Set `decoder=None` to return the clean latent directly
(training, latent-space evaluation, or pipelines that own decoding).

<a id="flashdreams.infra.pipeline.StreamInferencePipelineConfig.name"></a>
#### `name`

```text
name : str
```

Stable slug for this pipeline variant; the primary key of
`<NAME>_CONFIGS`. Runners mirror it as `runner_name` so
`flashdreams-run <slug>` resolves to this pipeline.
<a id="flashdreams.infra.pipeline.StreamInferencePipelineConfig.diffusion_model"></a>
#### `diffusion_model`

```text
diffusion_model : DiffusionModelConfig
```

Transformer + scheduler config.
<a id="flashdreams.infra.pipeline.StreamInferencePipelineConfig.decoder"></a>
#### `decoder`

```text
decoder : DecoderConfig | None = None
```

Optional output `StreamingDecoder` with a per-rollout cache,
called as `decoder(input, autoregressive_index, cache)`. Use
`None` to return the clean latent unchanged.
<a id="flashdreams.infra.pipeline.StreamInferencePipelineConfig.encoder"></a>
#### `encoder`

```text
encoder : EncoderConfig | None = None
```

Optional per-AR-step input encoder. Must be a
`StreamingEncoder`; one-shot encoders go on
`transformer.context_encoder` instead.
<a id="flashdreams.infra.pipeline.StreamInferencePipelineConfig.enable_sync_and_profile"></a>
#### `enable_sync_and_profile`

```text
property enable_sync_and_profile : bool
```

Whether to record synchronized per-stage CUDA timings.

Enabled process-wide with `FLASHDREAMS_SYNC_AND_PROFILE=1`

<a id="flashdreams.infra.pipeline.StreamInferencePipeline"></a>
### `StreamInferencePipeline`

```text
class StreamInferencePipeline ( config : StreamInferencePipelineConfig )
```

Bases: `Module`, `Generic`[`StreamingEncoderCacheT`, `TransformerCacheT`, `StreamingDecoderCacheT`]

End-to-end streaming inference pipeline.

Generic over the encoder, transformer, and decoder cache types. The
encoder’s input/output types are forwarded as `Any` so the
transformer’s `predict_flow` / `postprocess_clean_latent` overrides
own the typing on the `input` argument they receive.

Examples

cache = pipeline.initialize\_cache(transformer\_context={…})
output = pipeline.generate(0, cache, input=…)
pipeline.finalize(0, cache)
output = pipeline.generate(1, cache, input=…)
pipeline.finalize(1, cache) # optional for the last rollout

<a id="flashdreams.infra.pipeline.StreamInferencePipeline.initialize_cache"></a>
#### `initialize_cache`

```text
initialize_cache ( transformer_context : dict [ str , Any ] | None = None , encoder_context : dict [ str , Any ] | None = None , decoder_context : dict [ str , Any ] | None = None ) → StreamInferencePipelineCache [ StreamingEncoderCacheT , TransformerCacheT , StreamingDecoderCacheT ]
```

Build a fresh per-rollout cache.

Each `*_context` dict is forwarded as keyword arguments to the
corresponding component’s `initialize_autoregressive_cache`.

##### ``

```text
Parameters :
```

* **transformer\_context** – Per-rollout state for the transformer
  (e.g. `{"text_embeddings": ..., "image_embeddings": ...}`).
* **encoder\_context** – Per-rollout state for the encoder. Ignored
  when there is no encoder.
* **decoder\_context** – Per-rollout state for the decoder. Ignored
  when there is no decoder.
<a id="flashdreams.infra.pipeline.StreamInferencePipeline.generate"></a>
#### `generate`

```text
generate ( autoregressive_index : int , cache : StreamInferencePipelineCache [ StreamingEncoderCacheT , TransformerCacheT , StreamingDecoderCacheT ] , input : Any = None ) → Tensor
```

Generate one chunk for this AR step.

##### ``

```text
Parameters :
```

* **autoregressive\_index** – Must be `cache.autoregressive_index + 1`,
  or `0` for the first call after `initialize_cache`.
* **cache** – Per-rollout cache from `initialize_cache`.
* **input** – Raw input fed to the encoder. Required when an encoder
  is configured, must be `None` otherwise. Use
  `NullEncoderConfig` to pass an already-encoded tensor
  straight through.
<a id="flashdreams.infra.pipeline.StreamInferencePipeline.finalize"></a>
#### `finalize`

```text
finalize ( autoregressive_index : int , cache : StreamInferencePipelineCache [ StreamingEncoderCacheT , TransformerCacheT , StreamingDecoderCacheT ] ) → dict [ str , float ] | None
```

Advance the diffusion AR cache for the next AR step.

##### ``

```text
Parameters :
```

* **autoregressive\_index** – Must match the index passed to the most
  recent `generate` (asserted).
* **cache** – Same cache used by `generate`. Consumes
  `cache.final_state`.

<a id="flashdreams.infra.pipeline.StreamInferencePipelineCache"></a>
### `StreamInferencePipelineCache`

```text
class StreamInferencePipelineCache ( * , transformer_cache : TransformerCacheT , encoder_cache : StreamingEncoderCacheT | None = None , decoder_cache : StreamingDecoderCacheT | None = None , final_state : FinalState [ TransformerCacheT ] | None = None , autoregressive_index : int | None = None , event_profiler : EventProfiler | None = None )
```

Bases: `Generic`[`StreamingEncoderCacheT`, `TransformerCacheT`, `StreamingDecoderCacheT`]

Per-rollout cache held by the pipeline.

<a id="flashdreams.infra.pipeline.StreamInferencePipelineCache.transformer_cache"></a>
#### `transformer_cache`

```text
transformer_cache : TransformerCacheT
```

Long-lived transformer AR cache (always present).
<a id="flashdreams.infra.pipeline.StreamInferencePipelineCache.encoder_cache"></a>
#### `encoder_cache`

```text
encoder_cache : StreamingEncoderCacheT | None = None
```

Encoder AR cache; `None` iff the pipeline has no encoder.
<a id="flashdreams.infra.pipeline.StreamInferencePipelineCache.decoder_cache"></a>
#### `decoder_cache`

```text
decoder_cache : StreamingDecoderCacheT | None = None
```

Decoder AR cache; `None` iff the pipeline has no decoder.
<a id="flashdreams.infra.pipeline.StreamInferencePipelineCache.final_state"></a>
#### `final_state`

```text
final_state : FinalState [ TransformerCacheT ] | None = None
```

Diffusion-model state from the most recent `generate`, consumed
by `finalize`.
<a id="flashdreams.infra.pipeline.StreamInferencePipelineCache.autoregressive_index"></a>
#### `autoregressive_index`

```text
autoregressive_index : int | None = None
```

AR step index of the most recent `generate`.
<a id="flashdreams.infra.pipeline.StreamInferencePipelineCache.event_profiler"></a>
#### `event_profiler`

```text
event_profiler : EventProfiler | None = None
```

Per-step profiler, populated only when profiling is on.

## Diffusion model

Combines a transformer backbone with a denoising scheduler. The denoising
iteration is hidden inside
`~flashdreams.infra.diffusion.model.DiffusionModel.generate`, which
returns the clean latent and the final state needed by `finalize()` to
advance the autoregressive cache.

<a id="flashdreams.infra.diffusion.model.DiffusionModelConfig"></a>
### `DiffusionModelConfig`

```text
class DiffusionModelConfig ( * , _target: type[DiffusionModel] = <factory> , transformer: ~flashdreams.infra.diffusion.transformer.base.TransformerConfig , scheduler: ~flashdreams.infra.diffusion.scheduler.base.SchedulerConfig , seed: int | None = None , context_noise: int = 0 , noise_in_unpatchified_shape: bool = False )
```

Bases: `InstantiateConfig`

Config for the autoregressive diffusion model.

<a id="flashdreams.infra.diffusion.model.DiffusionModelConfig.transformer"></a>
#### `transformer`

```text
transformer : TransformerConfig
```

Flow-prediction network config.
<a id="flashdreams.infra.diffusion.model.DiffusionModelConfig.scheduler"></a>
#### `scheduler`

```text
scheduler : SchedulerConfig
```

Denoising-loop config.
<a id="flashdreams.infra.diffusion.model.DiffusionModelConfig.seed"></a>
#### `seed`

```text
seed : int | None = None
```

RNG seed for initial-noise draws and scheduler sampling.
`None` uses the global RNG.
<a id="flashdreams.infra.diffusion.model.DiffusionModelConfig.context_noise"></a>
#### `context_noise`

```text
context_noise : int = 0
```

Timestep used by `finalize` for the AR cache-update forward.
`0` skips `add_noise`.
<a id="flashdreams.infra.diffusion.model.DiffusionModelConfig.noise_in_unpatchified_shape"></a>
#### `noise_in_unpatchified_shape`

```text
noise_in_unpatchified_shape : bool = False
```

draw the initial noise in the unpatchified shape, then
patchify. Slower than the default patchified path; useful when matching
another implementation’s RNG sequence.

##### ``

```text
Type :
```

Debug-only

<a id="flashdreams.infra.diffusion.model.DiffusionModel"></a>
### `DiffusionModel`

```text
class DiffusionModel ( config : DiffusionModelConfig )
```

Bases: `Module`, `Generic`[`TransformerCacheT`]

Autoregressive diffusion model (scheduler + transformer).

Generic over the transformer’s AR cache type so user-facing typing on
`cache` is preserved end-to-end.

Examples

model = config.setup().to(“cuda”)
cache = model.transformer.initialize\_autoregressive\_cache(…)
clean, final\_state = model.generate(autoregressive\_index=0, cache=cache)
model.finalize(final\_state)

<a id="flashdreams.infra.diffusion.model.DiffusionModel.FinalState"></a>
#### `FinalState`

```text
class FinalState ( * , clean_latent : Tensor , autoregressive_index : int , cache : _FinalStateCacheT , input : Any | None = None )
```

Bases: `Generic`[`_FinalStateCacheT`]

State passed from `generate` to `finalize`.

<a id="flashdreams.infra.diffusion.model.DiffusionModel.FinalState.clean_latent"></a>
##### `clean_latent`

```text
clean_latent : Tensor
```

Patchified clean latent at the end of denoising.
<a id="flashdreams.infra.diffusion.model.DiffusionModel.FinalState.autoregressive_index"></a>
##### `autoregressive_index`

```text
autoregressive_index : int
```

AR step this state was produced at.
<a id="flashdreams.infra.diffusion.model.DiffusionModel.FinalState.cache"></a>
##### `cache`

```text
cache : _FinalStateCacheT
```

Long-lived AR cache used during generation.
<a id="flashdreams.infra.diffusion.model.DiffusionModel.FinalState.input"></a>
##### `input`

```text
input : Any = None
```

Patchified per-AR-step encoder output, or `None`.
<a id="flashdreams.infra.diffusion.model.DiffusionModel.rng"></a>
#### `rng`

```text
property rng : Generator | None
```

Per-model generator, lazily built on the current device.

Returns `None` when `config.seed` is `None`. Rebuilt the first
time the model’s device changes after a `.to(...)`. A device move
resets the RNG stream — fine for “construct on CPU, `.to(gpu)`
once” but mid-rollout device hops lose RNG state.
<a id="flashdreams.infra.diffusion.model.DiffusionModel.generate"></a>
#### `generate`

```text
generate ( autoregressive_index : int , cache : TransformerCacheT , input : Any | None = None ) → tuple [ Tensor , DiffusionModel.FinalState [ TransformerCacheT ] ]
```

Run the denoising loop for one AR step.

##### ``

```text
Parameters :
```

* **autoregressive\_index** – AR step index.
* **cache** – Long-lived AR cache, mutated in place.
* **input** – Optional per-AR-step encoder output. Patchified here and
  forwarded to `predict_flow` / `postprocess_clean_latent`,
  then stashed on the returned `FinalState` for `finalize`.
<a id="flashdreams.infra.diffusion.model.DiffusionModel.finalize"></a>
#### `finalize`

```text
finalize ( final_state : FinalState [ TransformerCacheT ] ) → None
```

Advance the AR cache using the clean latent from `generate`.

Re-noises the clean latent to `config.context_noise` and runs the
transformer’s `finalize_kv_cache` (one forward for vanilla
transformers, multiple for dual-network DiTs).

`context_noise == 0` skips `add_noise` (sigma=0 is identity) and
feeds the clean latent directly. This also dodges the requirement
for schedulers to support a `t=0` lookup (UniPC’s inference
schedule has no `t=0` entry).

## Transformer

<a id="flashdreams.infra.diffusion.transformer.TransformerConfig"></a>
### `TransformerConfig`

```text
class TransformerConfig ( * , _target: type[Transformer] = <factory> )
```

Bases: `InstantiateConfig`

Category base for every flow-prediction transformer config.

<a id="flashdreams.infra.diffusion.transformer.Transformer"></a>
### `Transformer`

```text
class Transformer ( config : TransformerConfig )
```

Bases: `Module`, `ABC`, `Generic`[`TransformerCacheT`]

Flow-prediction transformer, generic over its AR cache subclass.

Subclasses implement `predict_flow` and the patchify hooks. AR
transformers also subclass `TransformerAutoregressiveCache` and
override `initialize_autoregressive_cache`.

Example:

```
class MyTransformer(Transformer[MyCache]):
    def predict_flow(self, noisy_latent, timestep, cache, input=None):
        ...

    def initialize_autoregressive_cache(self, **context) -> MyCache:
        ...
```

<a id="flashdreams.infra.diffusion.transformer.Transformer.latent_shape"></a>
#### `latent_shape`

```text
abstract property latent_shape : tuple [ int , ... ]
```

Shape of the input/output latent tensor for this rank.

Includes batch dims. May depend on hierarchical context-parallel
group sizes (V/T/HW), so subclasses typically derive this from
`self.cp_groups` rather than from static config alone.
<a id="flashdreams.infra.diffusion.transformer.Transformer.predict_flow"></a>
#### `predict_flow`

```text
abstract predict_flow ( noisy_latent : Tensor , timestep : Tensor , cache : TransformerCacheT , input : Any | None = None ) → Tensor
```

Predict the flow at `timestep`.

##### ``

```text
Parameters :
```

* **noisy\_latent** – Patchified noisy latent for this denoising step.
* **timestep** – Scalar timestep tensor.
* **cache** – Per-rollout AR cache.
* **input** – Patchified encoder output for this AR step, or `None`
  when the pipeline has no encoder. Subclasses should narrow
  the type to their encoder’s output type.
<a id="flashdreams.infra.diffusion.transformer.Transformer.finalize_kv_cache"></a>
#### `finalize_kv_cache`

```text
finalize_kv_cache ( noisy_latent : Tensor , timestep : Tensor , cache : TransformerCacheT , input : Any | None = None ) → None
```

Advance the AR cache so it is ready for the next AR step.

Called by `DiffusionModel.finalize` after the denoising loop; the
flow is discarded — only the cache side effect matters. Default
runs a single `predict_flow` forward. Override for transformers
with multiple parallel networks that must stay in lock-step.

##### ``

```text
Parameters :
```

* **noisy\_latent** – Patchified latent at the AR-step’s context noise
  (or the clean latent when `context_noise == 0`).
* **timestep** – 0-d context-noise timestep tensor.
* **cache** – Per-rollout AR cache.
* **input** – Same patchified encoder output passed to `predict_flow`.
<a id="flashdreams.infra.diffusion.transformer.Transformer.initialize_autoregressive_cache"></a>
#### `initialize_autoregressive_cache`

```text
initialize_autoregressive_cache ( ** context : Any ) → TransformerCacheT
```

Build a fresh AR cache for a new rollout.

Default returns an empty cache, correct for non-AR transformers.
Subclasses with custom cache types must override this and may
declare typed per-rollout context (e.g. `text_embeddings`).

##### ``

```text
Parameters :
```

**context** – Per-rollout state forwarded as keyword arguments.
<a id="flashdreams.infra.diffusion.transformer.Transformer.initial_noise"></a>
#### `initial_noise`

```text
initial_noise ( * , latent_shape : tuple [ int , ... ] , rng : Generator | None , cache : TransformerCacheT , input : Any | None = None ) → Tensor
```

Draw the starting latent for one denoising loop.

Default is Gaussian noise in `latent_shape`. Model integrations can
override this to seed from encoder outputs, pin I2V frames, or attach
per-step state to the cache while still using `DiffusionModel`.

##### ``

```text
Parameters :
```

* **latent\_shape** – Shape requested by the diffusion model.
* **rng** – Per-model generator on `self.device`, or `None`.
* **cache** – Per-rollout AR cache.
* **input** – Same patchified encoder output passed to `predict_flow`.
<a id="flashdreams.infra.diffusion.transformer.Transformer.postprocess_clean_latent"></a>
#### `postprocess_clean_latent`

```text
postprocess_clean_latent ( clean_latent : Tensor , cache : TransformerCacheT , input : Any | None = None ) → Tensor
```

Optional postprocessing hook for the predicted clean latent.

Default is identity. Override to clamp or re-inject regions whose
clean value is known a priori (e.g. I2V first-frame pinning).
Called at the end of `DiffusionModel.generate`.

##### ``

```text
Parameters :
```

* **clean\_latent** – Patchified `x0` from the denoising loop.
* **cache** – Per-rollout AR cache.
* **input** – Same patchified encoder output passed to `predict_flow`.
<a id="flashdreams.infra.diffusion.transformer.Transformer.patchify_and_maybe_split_cp"></a>
#### `patchify_and_maybe_split_cp`

```text
abstract patchify_and_maybe_split_cp ( x : Any ) → Any
```

Patchify and (optionally) CP-split a noisy latent or encoder payload.

Tensors patchify and split. Structured payloads (e.g. an image-control
struct with `latent` + `mask`) patchify each tensor field and
return the same struct type. Implement as identity when neither
token packing nor CP sharding applies. Output preserves the input
Python type — only shapes change.
<a id="flashdreams.infra.diffusion.transformer.Transformer.unpatchify_and_maybe_gather_cp"></a>
#### `unpatchify_and_maybe_gather_cp`

```text
abstract unpatchify_and_maybe_gather_cp ( x : Tensor ) → Tensor
```

Inverse of `patchify_and_maybe_split_cp` for the network output.

<a id="flashdreams.infra.diffusion.transformer.TransformerAutoregressiveCache"></a>
### `TransformerAutoregressiveCache`

```text
class TransformerAutoregressiveCache
```

Bases: `object`

Cache that persists across an AR rollout.

Empty by default; safe to instantiate directly for non-AR transformers
(default `start` / `finalize` are no-ops). Subclass and add fields
plus AR bookkeeping for real per-rollout state.

Example:

```
cache.start(autoregressive_index)
# one or more denoising steps...
cache.finalize(autoregressive_index)
```

<a id="flashdreams.infra.diffusion.transformer.TransformerAutoregressiveCache.start"></a>
#### `start`

```text
start ( autoregressive_index : int ) → None
```

Mark the start of an AR step. Default is a no-op.

##### ``

```text
Parameters :
```

**autoregressive\_index** – Index of the AR step being started.
<a id="flashdreams.infra.diffusion.transformer.TransformerAutoregressiveCache.finalize"></a>
#### `finalize`

```text
finalize ( autoregressive_index : int ) → None
```

Finalize bookkeeping after use at this AR step. Default is a no-op.

##### ``

```text
Parameters :
```

**autoregressive\_index** – Index of the AR step just finalized.

## Schedulers

A scheduler owns the entire denoising loop. It is shape-agnostic: every
internal op is a broadcast against per-step scalar sigmas, so the same
scheduler works for any latent layout.

<a id="flashdreams.infra.diffusion.scheduler.Scheduler"></a>
### `Scheduler`

```text
class Scheduler ( config : SchedulerConfig )
```

Bases: `Module`, `ABC`

Denoising scheduler.

Owns the entire denoising loop. Callers see only `noise → clean`;
the loop shape (renoise / multistep / plain ODE) is private.

Concrete configs inherit `SchedulerConfig` and declare their own
`num_inference_steps` / `shift` fields (the base holds no
shared dataclass fields).

Examples

scheduler = config.setup()
clean = scheduler.sample(initial\_noise=noise, predict\_flow=predictor)
noisy = scheduler.add\_noise(clean\_input=clean, timestep=t)

<a id="flashdreams.infra.diffusion.scheduler.Scheduler.sample"></a>
#### `sample`

```text
abstract sample ( initial_noise : Tensor , predict_flow : FlowPredictor , rng : Generator | None = None ) → Tensor
```

Run the full denoising loop and return the clean latent.

Schedulers are shape-agnostic: every internal op broadcasts against
per-step scalar sigmas. In practice `initial_noise` is a video
latent `[B, C, T, H, W]`, conventionally treated as a sample at
`sigma=1`.

##### ``

```text
Parameters :
```

* **initial\_noise** – Gaussian noise on the caller’s device/dtype.
* **predict\_flow** – Per-step closure invoked
  `num_inference_steps` times.
* **rng** – Generator on the same device. Used by self-forcing renoise
  loops; pure ODE solvers ignore it.
<a id="flashdreams.infra.diffusion.scheduler.Scheduler.add_noise"></a>
#### `add_noise`

```text
abstract add_noise ( clean_input : Tensor , timestep : Tensor , rng : Generator | None = None ) → Tensor
```

Apply the forward corruption `x_t = (1 - sigma(t)) * x_0 + sigma(t) * eps`.

Timestep value semantics are scheduler-specific: all schedulers snap
to the nearest entry of their inference schedule.

<a id="flashdreams.infra.diffusion.scheduler.SchedulerConfig"></a>
### `SchedulerConfig`

```text
class SchedulerConfig ( * , _target: type[Scheduler] = <factory> )
```

Bases: `InstantiateConfig`

Category base for every denoising-scheduler config.

<a id="flashdreams.infra.diffusion.scheduler.FlowPredictor"></a>
### `FlowPredictor`

```text
class FlowPredictor ( * args , ** kwargs )
```

Bases: `Protocol`

Closure `(noisy_latent, timestep) -> predicted_flow`.

Built by `DiffusionModel.generate` by binding the per-AR-step `cache`
/ `input` to the transformer’s `predict_flow`. A scheduler invokes
it once per denoising iteration. The scheduler decides the timestep
dtype (UniPC uses int64, flow-match uses float).

<a id="flashdreams.infra.diffusion.scheduler.FlowMatchSchedulerConfig"></a>
### `FlowMatchSchedulerConfig`

```text
class FlowMatchSchedulerConfig ( * , _target: type[FlowMatchScheduler] = <factory> , num_inference_steps: int = 4 , shift: float = 8.0 , denoising_timesteps: list[int] = <factory> , warp_denoising_step: bool = True , num_train_timesteps: int = 1000 , sigma_max: float = 1.0 , sigma_min: float = 0.0 , extra_one_step: bool = True , timestep_dtype: ~torch.dtype = torch.float32 , enable_tqdm: bool = False )
```

Bases: `SchedulerConfig`

Config for the flow-matching scheduler.

<a id="flashdreams.infra.diffusion.scheduler.FlowMatchSchedulerConfig.num_inference_steps"></a>
#### `num_inference_steps`

```text
num_inference_steps : int = 4
```

Must equal `len(denoising_timesteps)`.
<a id="flashdreams.infra.diffusion.scheduler.FlowMatchSchedulerConfig.shift"></a>
#### `shift`

```text
shift : float = 8.0
```

Schedule warp factor.
<a id="flashdreams.infra.diffusion.scheduler.FlowMatchSchedulerConfig.denoising_timesteps"></a>
#### `denoising_timesteps`

```text
denoising_timesteps : list [ int ]
```

Per-step diffusion timesteps in `[0, num_train_timesteps]`.
<a id="flashdreams.infra.diffusion.scheduler.FlowMatchSchedulerConfig.warp_denoising_step"></a>
#### `warp_denoising_step`

```text
warp_denoising_step : bool = True
```

Map `denoising_timesteps` through the warped sigma schedule.
<a id="flashdreams.infra.diffusion.scheduler.FlowMatchSchedulerConfig.num_train_timesteps"></a>
#### `num_train_timesteps`

```text
num_train_timesteps : int = 1000
```

Length of the training sigma table.
<a id="flashdreams.infra.diffusion.scheduler.FlowMatchSchedulerConfig.sigma_max"></a>
#### `sigma_max`

```text
sigma_max : float = 1.0
```

Top of the linspace before warping; `1.0` matches DiffSynth, upstream
Wan / Lingbot ships `0.999`.
<a id="flashdreams.infra.diffusion.scheduler.FlowMatchSchedulerConfig.sigma_min"></a>
#### `sigma_min`

```text
sigma_min : float = 0.0
```

Bottom of the linspace before warping. Reserved for upstream parity;
only `0.0` is exercised.
<a id="flashdreams.infra.diffusion.scheduler.FlowMatchSchedulerConfig.extra_one_step"></a>
#### `extra_one_step`

```text
extra_one_step : bool = True
```

If `True`, build the schedule from
`linspace(sigma_max, sigma_min, N+1)[:-1]` (matches DiffSynth /
upstream Wan); `False` uses `N` points and is kept for non-Wan
recipes.
<a id="flashdreams.infra.diffusion.scheduler.FlowMatchSchedulerConfig.timestep_dtype"></a>
#### `timestep_dtype`

```text
timestep_dtype : dtype = torch.float32
```

Dtype of `denoising_step_list`. Set to an integer dtype (e.g.
`torch.int64`) when the network’s time embedding is sensitive to the
fractional part of the warped timestep — upstream Wan stores
`scheduler.timesteps` as `int64` and lets the embedding upcast to
`float64` internally.
<a id="flashdreams.infra.diffusion.scheduler.FlowMatchSchedulerConfig.enable_tqdm"></a>
#### `enable_tqdm`

```text
enable_tqdm : bool = False
```

Whether to enable tqdm progress bar.

<a id="flashdreams.infra.diffusion.scheduler.FlowMatchScheduler"></a>
### `FlowMatchScheduler`

```text
class FlowMatchScheduler ( config : FlowMatchSchedulerConfig )
```

Bases: `Scheduler`

Flow-matching scheduler with self-forcing renoise (DiffSynth-style).

Each iteration converts the predicted flow to an `x0` estimate, then
re-noises at the same sigma to feed the next iteration. The final
`x0` is returned:

```
x_t = initial_noise
for t in denoising_step_list:
    v = predict_flow(x_t, t)
    x0 = x_t - sigma(t) * v
    x_t = (1 - sigma(t)) * x0 + sigma(t) * eps
return x0
```

Example:

```
scheduler = FlowMatchSchedulerConfig(
    num_inference_steps=4,
    shift=8.0,
    denoising_timesteps=[1000, 750, 500, 250],
).setup().to("cuda")
clean = scheduler.sample(initial_noise=noise, predict_flow=fn)
```

Schedule buffers are pinned to fp32 even after `module.to(bf16)`;
integer timesteps like 1000 would otherwise round to 1024.

<a id="flashdreams.infra.diffusion.scheduler.FlowMatchScheduler.sample"></a>
#### `sample`

```text
sample ( initial_noise : Tensor , predict_flow : FlowPredictor , rng : Generator | None = None ) → Tensor
```

Run the self-forcing flow-match denoising loop.

Iteration 0 trusts `initial_noise` as the `sigma=1` sample;
later iterations re-noise the previous `x0` estimate to the new
sigma before the network forward. Schedule arithmetic auto-promotes
to fp32; the result is cast back to `initial_noise.dtype`.
<a id="flashdreams.infra.diffusion.scheduler.FlowMatchScheduler.add_noise"></a>
#### `add_noise`

```text
add_noise ( clean_input : Tensor , timestep : Tensor , rng : Generator | None = None ) → Tensor
```

Apply the forward corruption at an arbitrary timestep.

Snaps `timestep` to the nearest entry of the warped training table
and uses it as sigma in the standard lerp.

<a id="flashdreams.infra.diffusion.scheduler.FlowMatchEulerDiscreteSchedulerConfig"></a>
### `FlowMatchEulerDiscreteSchedulerConfig`

```text
class FlowMatchEulerDiscreteSchedulerConfig ( * , _target: type[FlowMatchEulerDiscreteScheduler] = <factory> , num_inference_steps: int = 4 , shift: float = 5.0 , num_train_timesteps: int = 1000 , fixed_timesteps: tuple[float , ...] | None = None , enable_tqdm: bool = False )
```

Bases: `SchedulerConfig`

Config for the flow-matching Euler-discrete scheduler.

Defaults match diffusers’ `FlowMatchEulerDiscreteScheduler`
with `num_train_timesteps=1000` and the standard
`linspace + shift` warp. Set `fixed_timesteps` to a length
`num_inference_steps + 1` tuple (terminal value typically `0.0`)
to pin an externally-derived distilled schedule.

<a id="flashdreams.infra.diffusion.scheduler.FlowMatchEulerDiscreteSchedulerConfig.num_inference_steps"></a>
#### `num_inference_steps`

```text
num_inference_steps : int = 4
```

Number of Euler steps. Matches the distilled WAN-5B 4-step path
out of the box; bump to 30-50 for non-distilled checkpoints.
<a id="flashdreams.infra.diffusion.scheduler.FlowMatchEulerDiscreteSchedulerConfig.shift"></a>
#### `shift`

```text
shift : float = 5.0
```

Schedule warp factor applied to the linspaced sigma grid. Ignored
when `fixed_timesteps` is set.
<a id="flashdreams.infra.diffusion.scheduler.FlowMatchEulerDiscreteSchedulerConfig.num_train_timesteps"></a>
#### `num_train_timesteps`

```text
num_train_timesteps : int = 1000
```

Length of the training sigma table. Used to convert between
integer timesteps and `[0, 1]` sigmas (`sigma = timestep / N`).
<a id="flashdreams.infra.diffusion.scheduler.FlowMatchEulerDiscreteSchedulerConfig.fixed_timesteps"></a>
#### `fixed_timesteps`

```text
fixed_timesteps : tuple [ float , ... ] | None = None
```

Override the derived schedule with a precomputed list of
timesteps. When set, must have length `num_inference_steps + 1`;
the trailing entry is typically `0.0` so the last Euler step
lands on the clean latent. `None` (default) derives the schedule
from `num_inference_steps` + `shift` via the standard
flow-matching linspace + warp.

The HY-WorldPlay distilled WAN-5B path pins this to
`(1000.0, 960.0, 888.8889, 727.2728, 0.0)`.
<a id="flashdreams.infra.diffusion.scheduler.FlowMatchEulerDiscreteSchedulerConfig.enable_tqdm"></a>
#### `enable_tqdm`

```text
enable_tqdm : bool = False
```

Whether to enable the tqdm progress bar inside `sample()`.

   Euler-discrete flow-matching scheduler.

<a id="flashdreams.infra.diffusion.scheduler.FlowMatchUniPCSchedulerConfig"></a>
### `FlowMatchUniPCSchedulerConfig`

```text
class FlowMatchUniPCSchedulerConfig ( * , _target: type[FlowMatchUniPCScheduler] = <factory> , num_inference_steps: int = 50 , shift: float = 5.0 , num_train_timesteps: int = 1000 , solver_order: int = 2 , use_kerras_sigma: bool = False , enable_tqdm: bool = False )
```

Bases: `SchedulerConfig`

Config for the flow-matching UniPC scheduler.

Defaults match the official Wan 2.1 inference integration (UniPC, BH2,
order 2, shift 5.0). Override `shift` per checkpoint as recommended
upstream (e.g. 3.0 for Wan 2.1 14B I2V 480P).

<a id="flashdreams.infra.diffusion.scheduler.FlowMatchUniPCSchedulerConfig.num_inference_steps"></a>
#### `num_inference_steps`

```text
num_inference_steps : int = 50
```

Number of UniPC denoising steps.
<a id="flashdreams.infra.diffusion.scheduler.FlowMatchUniPCSchedulerConfig.shift"></a>
#### `shift`

```text
shift : float = 5.0
```

Schedule warp factor.
<a id="flashdreams.infra.diffusion.scheduler.FlowMatchUniPCSchedulerConfig.num_train_timesteps"></a>
#### `num_train_timesteps`

```text
num_train_timesteps : int = 1000
```

Length of the training sigma table.
<a id="flashdreams.infra.diffusion.scheduler.FlowMatchUniPCSchedulerConfig.solver_order"></a>
#### `solver_order`

```text
solver_order : int = 2
```

UniPC solver order; only 2 is supported.
<a id="flashdreams.infra.diffusion.scheduler.FlowMatchUniPCSchedulerConfig.use_kerras_sigma"></a>
#### `use_kerras_sigma`

```text
use_kerras_sigma : bool = False
```

Whether to use the exact sigma used in edm sampler.
<a id="flashdreams.infra.diffusion.scheduler.FlowMatchUniPCSchedulerConfig.enable_tqdm"></a>
#### `enable_tqdm`

```text
enable_tqdm : bool = False
```

Whether to enable tqdm progress bar.

<a id="flashdreams.infra.diffusion.scheduler.FlowMatchUniPCScheduler"></a>
### `FlowMatchUniPCScheduler`

```text
class FlowMatchUniPCScheduler ( config : FlowMatchUniPCSchedulerConfig )
```

Bases: `Scheduler`

Order-2 UniPC predictor-corrector for flow-matching.

Specialized + pre-baked variant of the upstream Wan 2.1 UniPC solver.
Schedule buffers (sigmas + per-step coefficients) stay fp32 regardless
of `module.to(dtype)`.

Example:

```
scheduler = FlowMatchUniPCSchedulerConfig(
    num_inference_steps=50,
    shift=5.0,
).setup().to("cuda")
clean = scheduler.sample(initial_noise=noise, predict_flow=fn)
```

<a id="flashdreams.infra.diffusion.scheduler.FlowMatchUniPCScheduler.sample"></a>
#### `sample`

```text
sample ( initial_noise : Tensor , predict_flow : FlowPredictor , rng : Generator | None = None ) → Tensor
```

Run the order-2 UniPC predictor-corrector denoising loop.

Each iteration: network → flow → `x0` → corrector (skipped at
step 0) → predictor. All per-step coefficients are pre-baked at
construction; the loop is pure tensor ops. Internal arithmetic is
fp32; the result is cast back to `initial_noise.dtype`. `rng`
is unused (deterministic ODE) but accepted for interface conformance.
<a id="flashdreams.infra.diffusion.scheduler.FlowMatchUniPCScheduler.add_noise"></a>
#### `add_noise`

```text
add_noise ( clean_input : Tensor , timestep : Tensor , rng : Generator | None = None ) → Tensor
```

Apply the forward corruption at an arbitrary timestep.

Snaps `timestep` to the nearest entry of the inference schedule
on-device (no Python sync) and uses the matching sigma in the lerp.

## Encoder

Encoders turn raw conditioning (text prompts, reference images, per-AR-step
control inputs, …) into latent tensors. Two flavours:

- `Encoder` is stateless and one-shot. `forward(self, input)`.
  Used as `transformer.context_encoder` for text / CLIP-image / identity.
- `StreamingEncoder` is stateful and per-AR-step. ``forward(self,
  input, autoregressive_index, cache)`` with an
  `StreamingEncoderCache`. Used as `pipeline.encoder` for
  per-step control (HDMap, camera trajectory, I2V first-frame VAE).
- `StreamingVideoEncoder` extends `StreamingEncoder` with
  the contracts a streaming pixel-video encoder always needs: spatial /
  temporal compression ratios plus AR-step-aware temporal size mappers
  between pixel and latent space.

<a id="flashdreams.infra.encoder.EncoderConfig"></a>
### `EncoderConfig`

```text
class EncoderConfig ( * , _target: type[Encoder] = <factory> )
```

Bases: `InstantiateConfig`

Category base for every encoder config (stateless or streaming).

<a id="flashdreams.infra.encoder.Encoder"></a>
### `Encoder`

```text
class Encoder ( config : EncoderConfig )
```

Bases: `ABC`, `Module`

Stateless encoder.

`forward` is not pinned by the base. Encoders used as a
`context_encoder` (one-shot, called once inside
`Transformer.initialize_autoregressive_cache()`) must match the
slim call shape `forward(self, input)`.

For per-AR-step encoders that need a per-rollout cache, inherit
from `StreamingEncoder` instead.

<a id="flashdreams.infra.encoder.StreamingEncoder"></a>
### `StreamingEncoder`

```text
class StreamingEncoder ( config : EncoderConfig )
```

Bases: `ABC`, `Module`, `Generic`[`StreamingEncoderCacheT`]

Streaming encoder, generic over the per-rollout cache type.

`forward` is not pinned by the base. Streaming encoders called by
`StreamInferencePipeline` must match its call shape:
`forward(self, input, autoregressive_index=0, cache=None)`.

<a id="flashdreams.infra.encoder.StreamingEncoder.initialize_autoregressive_cache"></a>
#### `initialize_autoregressive_cache`

```text
abstract initialize_autoregressive_cache ( ** context : Any ) → StreamingEncoderCacheT
```

Build a fresh per-rollout cache.

Override to return the encoder’s concrete cache type.

<a id="flashdreams.infra.encoder.StreamingVideoEncoder"></a>
### `StreamingVideoEncoder`

```text
class StreamingVideoEncoder ( config : EncoderConfig )
```

Bases: `StreamingEncoder`[`StreamingEncoderCacheT`]

Streaming pixel-video encoder.

Pins down the contracts that every streaming pixel→latent video
encoder satisfies in addition to `StreamingEncoder`:

* Spatial and temporal compression ratios between the pixel and
  latent grids (constants of the architecture).
* AR-step-aware temporal size mappers, so a pipeline can size its
  inputs and outputs without knowing the encoder’s concrete
  temporal cache topology (causal first-frame padding, sliding
  windows, etc.).

Spatial scaling is trivially `side // spatial_compression_ratio`
in either direction; the AR-step-asymmetric piece is the temporal
size, which gets its own mapper. Typically AR 0 takes
`1 + (T_lat - 1) * r` pixel frames to produce `T_lat` latent
frames because of causal first-frame padding, while AR ≥ 1 takes
`T_lat * r` pixel frames.

<a id="flashdreams.infra.encoder.StreamingVideoEncoder.spatial_compression_ratio"></a>
#### `spatial_compression_ratio`

```text
abstract property spatial_compression_ratio : int
```

Pixel side ÷ latent side. Constant across AR steps.
<a id="flashdreams.infra.encoder.StreamingVideoEncoder.temporal_compression_ratio"></a>
#### `temporal_compression_ratio`

```text
abstract property temporal_compression_ratio : int
```

Pixel frames ÷ latent frames in steady state (AR ≥ 1).

AR 0 typically takes one extra (un-grouped) pixel frame to
produce its first latent frame because of causal first-frame
padding; that asymmetry lives inside
`get_output_temporal_size()` /
`get_input_temporal_size()`.
<a id="flashdreams.infra.encoder.StreamingVideoEncoder.get_output_temporal_size"></a>
#### `get_output_temporal_size`

```text
abstract get_output_temporal_size ( autoregressive_index : int , input_temporal_size : int ) → int
```

Latent frame count produced from `input_temporal_size` pixel frames.

##### ``

```text
Parameters :
```

* **autoregressive\_index** – AR step index (0-based).
* **input\_temporal\_size** – Number of pixel frames fed at this step.
<a id="flashdreams.infra.encoder.StreamingVideoEncoder.get_input_temporal_size"></a>
#### `get_input_temporal_size`

```text
abstract get_input_temporal_size ( autoregressive_index : int , output_temporal_size : int ) → int
```

Pixel frame count needed to produce `output_temporal_size` latents.

Inverse of `get_output_temporal_size()`. Implementations
should assert `output_temporal_size` is achievable at this AR
step (i.e. the corresponding pixel count comes out as a
positive integer).

##### ``

```text
Parameters :
```

* **autoregressive\_index** – AR step index (0-based).
* **output\_temporal\_size** – Desired number of latent frames.

<a id="flashdreams.infra.encoder.StreamingEncoderCache"></a>
### `StreamingEncoderCache`

```text
class StreamingEncoderCache
```

Bases: `object`

Per-rollout cache for `StreamingEncoder`.

Empty by default; subclass to add fields (e.g. last-frame latent,
cross-step accumulators).

<a id="flashdreams.infra.encoder.NullEncoderConfig"></a>
### `NullEncoderConfig`

```text
class NullEncoderConfig ( * , _target: type[NullEncoder] = <factory> )
```

Bases: `EncoderConfig`

Config for the identity encoder.

<a id="flashdreams.infra.encoder.NullEncoder"></a>
### `NullEncoder`

```text
class NullEncoder ( config : EncoderConfig )
```

Bases: `Encoder`

Identity encoder: returns its input unchanged.

Wire as the transformer’s `context_encoder` slot to pass already-
encoded tensors straight to the diffusion model.

Example:

```
config = TransformerConfig(
    context_encoder=NullEncoderConfig(),
    ...,
)
```

<a id="flashdreams.infra.encoder.NullEncoder.forward"></a>
#### `forward`

```text
forward ( input : Any ) → Any
```

Return `input` unchanged.

## Decoder

Decoders turn the latents emitted by the diffusion model back into pixel
frames. Single base class with two specialisations:

- `StreamingDecoder` is stateful. ``forward(self, input,
  autoregressive_index, cache)`` with a `StreamingDecoderCache`.
  Use for chunk-by-chunk streaming decoders (e.g. WAN VAE that maintains
  a temporal cache across AR steps); stateless decoders just return an
  empty `StreamingDecoderCache` from
  `StreamingDecoder.initialize_autoregressive_cache` and ignore
  `autoregressive_index` / `cache` in `forward`.
- `StreamingVideoDecoder` extends `StreamingDecoder` with
  the contracts a streaming pixel-video decoder always needs: spatial /
  temporal compression ratios plus AR-step-aware temporal size mappers
  between latent and pixel space.

<a id="flashdreams.infra.decoder.DecoderConfig"></a>
### `DecoderConfig`

```text
class DecoderConfig ( * , _target: type[StreamingDecoder] = <factory> )
```

Bases: `InstantiateConfig`

Category base for every decoder config.

<a id="flashdreams.infra.decoder.StreamingDecoder"></a>
### `StreamingDecoder`

```text
class StreamingDecoder ( config : DecoderConfig )
```

Bases: `ABC`, `Module`, `Generic`[`StreamingDecoderCacheT`]

Streaming decoder, generic over the per-rollout cache type.

`forward` is not pinned by the base. Streaming decoders called by
`StreamInferencePipeline` must match its call shape:
`forward(self, input, autoregressive_index=0, cache=None)`.

<a id="flashdreams.infra.decoder.StreamingDecoder.initialize_autoregressive_cache"></a>
#### `initialize_autoregressive_cache`

```text
abstract initialize_autoregressive_cache ( ** context : Any ) → StreamingDecoderCacheT
```

Build a fresh per-rollout cache.

Override to return the decoder’s concrete cache type.

<a id="flashdreams.infra.decoder.StreamingVideoDecoder"></a>
### `StreamingVideoDecoder`

```text
class StreamingVideoDecoder ( config : DecoderConfig )
```

Bases: `StreamingDecoder`[`StreamingDecoderCacheT`]

Streaming pixel-video decoder.

Pins down the contracts that every streaming latent→pixel video
decoder satisfies in addition to `StreamingDecoder`:

* Spatial and temporal compression ratios between the latent and
  pixel grids (constants of the architecture).
* AR-step-aware temporal size mappers, so a pipeline can size its
  inputs and outputs without knowing the decoder’s concrete
  temporal cache topology (causal first-frame padding, sliding
  windows, etc.).

Spatial scaling is trivially `side * spatial_compression_ratio`
in either direction; the AR-step-asymmetric piece is the temporal
size, which gets its own mapper. Typically AR 0 produces fewer
pixel frames per latent frame than AR ≥ 1 because of causal
first-frame padding.

<a id="flashdreams.infra.decoder.StreamingVideoDecoder.spatial_compression_ratio"></a>
#### `spatial_compression_ratio`

```text
abstract property spatial_compression_ratio : int
```

Pixel side ÷ latent side. Constant across AR steps.
<a id="flashdreams.infra.decoder.StreamingVideoDecoder.temporal_compression_ratio"></a>
#### `temporal_compression_ratio`

```text
abstract property temporal_compression_ratio : int
```

Pixel frames ÷ latent frames in steady state (AR ≥ 1).

AR 0 typically yields fewer pixel frames per latent frame
because of causal first-frame padding; that asymmetry lives
inside `get_output_temporal_size()` /
`get_input_temporal_size()`.
<a id="flashdreams.infra.decoder.StreamingVideoDecoder.get_output_temporal_size"></a>
#### `get_output_temporal_size`

```text
abstract get_output_temporal_size ( autoregressive_index : int , input_temporal_size : int ) → int
```

Pixel frame count produced by `input_temporal_size` latent frames.

##### ``

```text
Parameters :
```

* **autoregressive\_index** – AR step index (0-based).
* **input\_temporal\_size** – Number of latent frames fed at this step.
<a id="flashdreams.infra.decoder.StreamingVideoDecoder.get_input_temporal_size"></a>
#### `get_input_temporal_size`

```text
abstract get_input_temporal_size ( autoregressive_index : int , output_temporal_size : int ) → int
```

Latent frame count needed to produce `output_temporal_size` pixels.

Inverse of `get_output_temporal_size()`. Implementations
should assert `output_temporal_size` is achievable at this AR
step (i.e. divisible by the right ratio after subtracting any
causal padding).

##### ``

```text
Parameters :
```

* **autoregressive\_index** – AR step index (0-based).
* **output\_temporal\_size** – Desired number of pixel frames.

<a id="flashdreams.infra.decoder.StreamingDecoderCache"></a>
### `StreamingDecoderCache`

```text
class StreamingDecoderCache
```

Bases: `object`

Per-rollout cache for `StreamingDecoder`.

Empty by default; subclass to add fields (e.g. temporal feature
buffers carried across AR steps).
