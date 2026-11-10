---
name: flashdreams-integrate-model
description: Add or migrate a video or world model to FlashDreams V2, including recipe reuse, checkpoint conversion, pipeline configuration, and an adapter for an existing demo. Use when porting an external model or wiring its application entry point; creating a new model-independent demo and running an already supported preset are separate tasks.
license: Apache-2.0
metadata:
  author: "FlashDreams Team <flashdreams@nvidia.com>"
---

# Integrate a FlashDreams model

## Establish the target

Work in the user's FlashDreams checkout at the requested revision. If installed
separately, locate that checkout or obtain `https://github.com/NVIDIA/flashdreams.git`
when source setup is part of the request. Resolve paths below from the repo root.
For review-only tasks, use the supplied code and identify missing evidence rather
than creating a new integration.

Record the upstream code/checkpoint revisions, access and license requirements,
closest existing backbone, scheduler/steps/guidance, dtype, spatial and temporal
shape, conditioning, and streaming/cache semantics. Establish the upstream parity
command before changing numerics. A new modality or unsupported runtime contract
needs an explicit framework extension; do not hide it in video tensors or metrics.

## Place code at the correct boundary

```text
integrations_v2/<model>/
  __init__.py
  config.py
  impl/                    # model-specific code and checkpoint conversion
  apps/<demo>/adapter.py   # binding to a reusable app
  tests/
  pyproject.toml
```

Reuse `flashdreams.core`, `flashdreams.infra`, `flashdreams.recipes` and the closest
existing model implementation. Prefer config changes and small model-specific
subclasses over copying a network. `core` does not import `infra`; shared framework
code does not import integrations. A reusable `apps/<demo>/` package depends on
the framework and works with a stub model. The integration depends on that app,
never the reverse. Model and adapter tests belong in the integration's `tests/`.

The current workspace glob is `integrations_v2/*`. Mirror a current package's
setuptools discovery and workspace dependencies; do not restore the obsolete
`integrations/<model>` plus `integrations_v2/<application>` split.

## Configure the pipeline and prove weight loading

Define named, module-level `StreamInferencePipelineConfig` variants in `config.py`
and a dictionary keyed by their names. Use `derive_config` for small overrides.
Only subclass the pipeline when an actual contract differs.

- Put one-shot text/image context on the transformer's `context_encoder`.
- Put per-step controls on the pipeline's `StreamingEncoder` slot.
- Use the decoder contract appropriate to the output; pixel-video VAEs expose
  spatial/temporal compression needed to size I/O.
- Preserve checkpoint source, scheduler, denoising steps, cache/window/sink
  policy, guidance, precision, compile and CUDA-graph settings explicitly.
- Per-rollout state belongs in caches. Follow the chosen pipeline's
  `initialize_cache`, `generate`, and `finalize` contract, including finalization
  and reset after a rolling-window boundary.

Prefer native weights when they match the implementation. Validate the real
checkpoint transform using metadata or meta tensors before allocating GPU memory:
no missing keys, unexpected keys, shape mismatches, or two source keys mapping to
one destination. Count transformed keys as well as comparing sets so dictionary
overwrites cannot hide collisions. Cover representative upstream key names.
For `.pth`, use `weights_only=True`; do not disable safe loading just to make an
untrusted checkpoint load. Treat weight conversion and V2 application wiring as
separate correctness checks.

## Bind an existing demo

Inspect `integrations_v2/wan21/apps/minimal_t2v/adapter.py` and its manifest as a
small reference. Export a zero-argument `create_app()` returning `IApplication`;
the factory supplies pipeline configuration and defaults without loading weights.

```toml
[project.entry-points."flashdreams.applications_v2"]
"my-demo-my-model" = "my_model.apps.my_demo.adapter:create_app"
```

Declare the reusable app dependency, its workspace source for an in-tree package,
and include adapter modules in the built wheel. Out-of-tree packages declare
installable dependencies rather than workspace-only mappings. Do not register a
new V2 adapter in `flashdreams.runner_configs` or add a `RunnerConfig` wrapper.

Complete downloads, extraction, native compilation/loading and model construction
within `IApplication.init`; wait for any preparation workers before returning.
The preparation guard applies after `init` returns, including to workers started
there. Constructors and `session_desc()` must remain cheap: the runtime requests
the description before `init`. Session creation consumes prepared resources and
returns an uninitialized, isolated session; loops own mutable per-run state.
Missing prepared assets should produce a useful error rather than a download
from a model step.

## Validate and document

Start with metadata/config, checkpoint-remap and stub-adapter checks. Every pytest
test carries one of `ci_cpu`, `ci_gpu`, or `manual`. Once dependencies are present:

```bash
uv run --no-sync pytest integrations_v2/MODEL/tests -m ci_cpu
uv run --no-sync flashdreams-run-v2 PRESET -- --help
```

Replace `MODEL` and `PRESET` with the actual package and registered entry point.
Validate wheel contents and discovery in a clean environment before claiming the
integration is independently installable. Parsing help does not validate weights.
Real parity, generation, reset/window-boundary behavior and performance need the
selected model's GPU environment. Report what ran, what did not, and the upstream
reference, tolerances and artifacts used; do not infer parity from attractive video.

Update `docs/source/models/<model>.md` with requirements and runnable presets,
link demo controls, and update model navigation/gallery for a new model. Keep
third-party provenance and dependency notices consistent with the source repo's
contribution process.

## References

- [Create a model](https://flashdreams.org/main/documentation/inferencing_api/guides/create_model/)
- [Connect a model to a demo](https://flashdreams.org/main/documentation/demo_api/guides/integrate_model_adapter_for_a_demo/)
- [Inference pipeline](https://flashdreams.org/main/documentation/inferencing_api/guides/stream_inference_pipeline/)
