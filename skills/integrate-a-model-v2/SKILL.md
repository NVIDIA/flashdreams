---
name: integrate-a-model-v2
description: Guides end-to-end integration or migration of video and world models onto the FlashDreams V2 API, including recipe reuse, checkpoint remapping, IApplication/ISession/IModelLoop wiring, entry-point packaging, CPU contract tests, GPU rollout, and upstream parity. Use when adding an external model, rebasing an integration onto V2, or replacing legacy runner, runner_io, video_output, or model-owned serving code.
---

# Integrate a model with the FlashDreams V2 API

Build the smallest model-specific package that satisfies the current V2 contracts. Reuse
the closest pipeline recipe, prove checkpoint loading independently, and let the runtime
own orchestration, threading, input collection, presentation, and file/stream output.

Read `skills/flashdreams-integrations/SKILL.md` for model and pipeline architecture. Its
legacy runner sections describe V1; for application/runtime work, this skill and the V2
source on the target branch take precedence. Read `skills/integrate-a-model/SKILL.md` only
for checkpoint and model-core details that this skill does not cover. Match
`skills/python-docstring-style` when writing public Python APIs.

## Start from the target, not the source PR

Before editing, identify the exact merge base or target PR and inspect these files there:

- `flashdreams/flashdreams/api_v2/{application,session,loop}.py`
- `flashdreams/flashdreams/runtime_v2/{session_desc,step_result}.py`
- `flashdreams/flashdreams/t2v_v2/` for the reusable text-to-video adapter
- `integrations_v2/` for packaging and test examples

Do not infer the contract from an older checkout, a review document, or the V1 API. V2 is
still evolving. Record the target commit, chosen reference integration, published
checkpoint, and upstream parity implementation in the integration note.

When rebasing an existing PR, land conflict resolution before the V2 port when practical.
Keep the target branch's root dependency policy and lockfile, then add only scoped package
dependencies. Do not preserve a V1 shim unless dual-stack support is explicitly required.

## Non-negotiable boundaries

- Do not add new `Runner`, `RunnerConfig`, `flashdreams.runner_configs`,
  `flashdreams.infra.runner_io`, or `flashdreams.infra.video_output` dependencies for a
  V2-only integration.
- Do not create application-owned generation threads, presentation queues, MP4 writers,
  WebRTC servers, or input-polling loops. The V2 runtime owns those.
- Keep dependency direction `core -> infra -> recipes/integrations`. Framework packages
  never import a model integration; expose a generic hook for shared behavior.
- Keep shared, expensive model state on `IApplication`; keep cache, controls, rollout
  counters, and other per-run mutable state in the session's loops.
- Treat unsupported output modalities as framework work, not as tensors hidden in video,
  `metrics`, or `SessionDesc.metadata`.

## Choose the integration lane

| Lane | Use it when | Shape |
| --- | --- | --- |
| Shared T2V adapter | A prompt produces video blocks and live controls are unnecessary | Configure/subclass `T2VApplication`; reuse `T2VSession` and `T2VModelLoop` |
| Custom V2 application | The model consumes live controls, has unusual cache/finalize behavior, or needs custom UI composition | Implement `IApplication`, `ISession`, and `IModelLoop`; add `IUILoop` only when the default blitter is insufficient |
| Framework extension first | The model needs audio, actions, depth, typed outputs, or an input event absent from the target API | Design and test a model-neutral V2 contract and sinks first; then integrate the model |

Estimate the model port and the runtime extension separately. A novel backbone or new
modality is not a small adapter even when its factory is short.

## Phase 1: reuse the model core

Find the closest recipe under `integrations/` or `flashdreams/flashdreams/recipes/`.
Prefer config plus model-specific subclasses over a forked network. Write down:

1. Backbone, scheduler, denoising steps, guidance, dtype, and resolution.
2. Streaming versus one-shot behavior, frames per block, and first-block differences.
3. Static conditions such as prompt or first frame, and live control inputs.
4. Per-run state: KV caches, encoded prompt, RNG, conditioning history, reset semantics.
5. Shared state: checkpoint, compiled pipeline, tokenizer, immutable lookup tables.

Keep model deltas inside the model integration. Zero-initialize new residual conditioner
heads where an identity fallback is valid. If a base checkpoint lacks only those heads,
tolerate exactly those missing keys and remain strict for everything else.

## Phase 2: prove checkpoint loading

Prefer the native checkpoint when it matches the ported network. Before downloading full
weights, compare key counts and use safetensors headers plus a model built on `meta` to
prove that the real `state_dict_transform` is a key-and-shape bijection:

- no model keys missing after remap;
- no transformed checkpoint keys left over;
- no shape mismatches;
- representative real keys covered by focused spot checks.

For a `.pth`, load with `map_location="meta", weights_only=True`. For sharded
safetensors, read the index and construct meta tensors from each slice shape. Make this a
`ci_cpu` test. If changing checkpoint source, compare transformed tensors from both
sources and require exact equality or document the measured difference.

Checkpoint correctness is independent of V2 wiring. Prove it before debugging the
runtime; an uninitialized meta parameter is usually a key-name mismatch, not a threading
bug.

## Phase 3: package the V2 adapter

Keep reusable model code and the V2 surface separate:

```text
integrations/<model>/                 # recipe, weights, pipeline, model tests
integrations_v2/<application>/
|-- pyproject.toml                    # depends on flashdreams + model package
`-- <application>/
    |-- __init__.py
    |-- app.py                        # create_app and V2 types
    `-- tests/
```

Register a zero-argument factory in the V2 group:

```toml
[project.entry-points."flashdreams.applications_v2"]
"my-model" = "my_model.app:create_app"
```

The factory must return `flashdreams.api_v2.application.IApplication`. The registry can
import an unregistered module exposing `create_app()` during development, but the shipped
package must declare the entry point. The V2 adapter may depend on the model integration;
the model integration must not depend on the adapter.

### Standard text-to-video

Use `T2VApplicationDefaults` with the existing pipeline config. Populate it directly or
temporarily derive it from a runner config while V1 and V2 coexist. Subclass
`T2VApplication` only for actual model differences:

- `_configure_argument_parser` and `_apply_parsed_arguments` for static conditions;
- `_validate_total_blocks` for one-shot/bidirectional models;
- `_apply_compile_override` or `_apply_seed_override` for nonstandard config locations;
- `session_type` when cache construction or stepping differs.

Do not copy the shared T2V command line or loop into every integration.

### Custom or interactive model

Implement the target branch's lifecycle exactly:

- `IApplication.init(args)` parses application arguments and validates cheap startup
  state. Runtime arguments remain outside the application.
- `IApplication.session_desc()` must be cheap and is called before `init`; return the
  model's natural layout, size, rates, and presentation policy without loading weights.
- `IApplication.create_session(desc)` validates before expensive loading, loads shared
  model state once, and returns an uninitialized isolated session.
- `ISession.init()` constructs per-run state and registers exactly one `IModelLoop`.
  Register an `IUILoop` only for custom composition or controls.
- `IModelLoop.step(step_index, events)` returns a **list** of `StepResult`, one per video
  channel. Even a single channel is a one-element list.
- `IModelLoop.is_finished()` ends finite/file runs. `reset()` replaces state that must not
  leak across generations. `close()` releases loop-owned resources.
- Cleanup must tolerate partial initialization and must not hide the original failure.

The model loop runs on its own Python thread; the UI loop runs on the thread driving the
session. Each loop owns its state. Use `invoke_async(target_loop, operation)` for
cross-loop mutation; never mutate the other loop's state directly. Operations must
return `None` and execute before the target's next step. Let the runtime discard results
from an abandoned generation after reset.

## Phase 4: map inputs and outputs truthfully

Static prompt/image/config inputs belong to application arguments after the CLI's `--`.
Live input arrives as timestamp-ordered `UserInputEvents`. Convert event edges into model
controls with a pure, unit-tested mapper. Preserve held-key state across batches and use
only event payloads implemented by the target branch; some modality classes may be stubs.

For every video channel, `StepResult` must declare:

- the received `step_index`;
- a detached tensor whose dimensions match `output_layout`;
- `frame_count` equal to the tensor's temporal extent;
- numeric measurements only in `metrics`.

Current sinks interpret floating tensors as `[-1, 1]` and integer tensors as `[0, 255]`.
They accept the target's declared `VideoTensorLayout` values but require one sequence for
ordinary presentation. Do not perform MP4 conversion or CPU frame assembly in the model
loop.

On the PR #506 V2 contract, `StepResult` is video-only. A list of results represents video
channels, not arbitrary modalities. Audio samples, action vectors, depth semantics, and
other typed payloads require a model-neutral result/session/sink extension with CPU
contract tests before the model port can claim end-to-end support.

Choose presentation deliberately:

- equality, parity, and MP4 completeness: `BackpressureMode.BLOCK` plus
  `PresentationMode.ONLY_PRESENT_NEW`;
- latency-sensitive interaction: consider `DROP_OLDEST` and
  `ONLY_PRESENT_NEWEST`, and test the expected dropping/reuse behavior.

## Phase 5: test in increasing cost order

Every pytest test gets exactly one repository marker: `ci_cpu`, `ci_gpu`, or `manual`.

1. **CPU contract tests:** factory type, entry-point metadata, cheap `session_desc`,
   argument split, validation before model load, shared model across sessions, distinct
   per-session state, and cleanup on failure.
2. **CPU loop tests:** channel-list contract, frame count/shape/layout/range, step order,
   finite completion, reset, input mapping, metrics, and `invoke_async` where used.
3. **CPU stand-in end-to-end:** run through `run_session`; for T2V use
   `flashdreams.t2v_v2.testing` and its fake pipeline. If `ffmpeg` is available, write an
   MP4 and assert frame count plus non-empty, changing imagery.
4. **Checkpoint tests:** meta-tensor remap bijection and real-key spot checks; make large
   downloads opt-in.
5. **GPU smoke:** instantiate the real checkpoint, cover first and steady-state blocks,
   and produce valid output through the V2 runtime.
6. **Upstream parity:** same checkpoint, input, seed, precision, scheduler, and step count.
   Report the agreed metric and tolerance; visual review alone is not parity.
7. **Performance:** compare matched stacks, discard compile/autotune warmup, and separate
   model, decode, transfer, presentation, and encode time.

Use `flashdreams-run-v2 SLUG -- --help` to inspect application arguments. A representative
file run is:

```bash
uv run --project integrations_v2/<application> flashdreams-run-v2 \
    <slug> --output-path /tmp/<slug>.mp4 \
    -- --prompt "A cat surfing" --total-blocks 2 --no-compile
```

Do not run checkpoint downloads, CUDA generation, WebRTC, or large-model tests on a
CPU-only host unless explicitly requested. A config/import smoke does not prove a
checkpoint load, generation, output encoding, or parity.

## Migration sequence for an existing integration PR

Keep commits reviewable and independently diagnosable:

1. Rebase/transplant the source PR onto the target, resolving root manifests in favor of
   the target's policy.
2. Fix model correctness and checkpoint mapping without changing runtime behavior.
3. Extract a model/pipeline seam containing no CLI, writer, server, or worker thread.
4. Add the `integrations_v2/` adapter and entry point.
5. Add CPU contract and stand-in end-to-end tests.
6. Add any model-neutral V2 API extension as a separate prerequisite change.
7. Run the real GPU rollout, upstream parity, and matched performance checks.
8. Remove legacy runner/serving code only when compatibility is out of scope.

Avoid mixing dependency-policy rewrites, runtime refactors, correctness fixes, and parity
changes in one commit. Never hand-edit `uv.lock`; regenerate it after manifests settle.

## Done criteria

- [ ] The target V2 API commit and closest reference integration are recorded.
- [ ] Model code is isolated from V2 runtime orchestration and legacy runner I/O.
- [ ] Checkpoint keys and shapes form a complete bijection, or exceptions are proven.
- [ ] `create_app()` returns `IApplication` and is registered under
      `flashdreams.applications_v2`.
- [ ] Application, session, model-loop, optional UI-loop, reset, and cleanup ownership are
      tested with stand-ins on CPU.
- [ ] Every returned channel satisfies `StepResult` shape, layout, range, frame-count, and
      metrics rules.
- [ ] Deterministic/MP4 runs preserve every frame with safe presentation settings.
- [ ] Unsupported modalities have a landed generic V2 contract or are explicitly scoped
      out; they are not hidden in video tensors or metadata.
- [ ] A real GPU rollout succeeds and upstream parity meets the agreed tolerance.
- [ ] CI-pinned lint, type checks, and correctly marked tests pass.
