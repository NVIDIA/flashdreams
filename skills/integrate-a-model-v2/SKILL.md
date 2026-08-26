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
still evolving. Record the target commit, chosen reference integration, immutable
checkpoint revision, pinned upstream parity commit, and external-input hashes in the
integration note.

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

For stateful or multimodal ports, trace caches and conditioning through the lowest shared
compute boundary. Tests must verify the values actually consumed, not merely that an
argument passes through a wrapper. Cover every active CFG/guidance branch and assert the
model-defined ordering of committed history, current peer-modality state, and fresh state.
Pin a narrow dependency range when the port relies on private upstream APIs, and record
what must be retested before widening it.

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

Do not treat `integrations_v2/<application>` as an automatic replacement for
`integrations/<model>`. The former is normally the runtime adapter and the latter remains
the reusable model/checkpoint/pipeline package, as demonstrated by the dependency and
imports between them. Remove `integrations/<model>` only if those model responsibilities
were deliberately moved and the adapter is independently packageable afterward. Remove
legacy runner, writer, server, and CLI surfaces without deleting still-used model code.

If an installed integration claims browser streaming works from its declared environment,
its package must depend on `flashdreams[serving]`, not only `flashdreams`. Keep file-only
integrations on the narrower dependency when possible.

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

Clear held controls and pointer origins on focus loss and reset so input cannot stick.
Define what happens to events received while a seed, first frame, or other bootstrap step
is being presented. A step that returns bootstrap imagery without consuming its event
batch can silently lose the user's first action. Either carry those events into the first
generated action or keep input disabled until the model can consume it, and test a press
and release spanning that boundary.

Derive `SessionDesc` dimensions and per-step frame counts from the decoder's authoritative
shape mappings. When temporal expansion, causal overlap, compositing, or a presentation
crop makes that mapping ambiguous, confirm it with a real decoder probe before hardcoding
the values. Latent chunk geometry alone is not sufficient evidence.

For every video channel, `StepResult` must declare:

- the received `step_index`;
- a detached tensor whose dimensions match `output_layout`;
- `frame_count` equal to the tensor's temporal extent;
- numeric measurements only in `metrics`.

Name measurements with runtime-recognized unit suffixes such as `_s`, `_ms`, `_fps`,
`_bytes`, `_gib`, and `_count`, and test the normalized sink output.

Do not describe presentation queue, drop, or reset-discard counters as frames unless the
target contract explicitly counts frames. They commonly count chunks or model steps;
derive a frame total from each result's `frame_count` when reporting frame completeness.

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

### Interactive and WebRTC acceptance

For an application that claims interactive browser support, exercise the installed entry
point through the intended client/network path, not only a localhost HTTP smoke:

- Verify bind address, HTTP health, offer/answer signaling, UDP ICE, first video, and the
  controls data channel separately. An SSH `-L` tunnel forwards TCP signaling but does
  not prove the UDP media path. Document required bind/firewall behavior, and do not
  change host firewall policy without authorization.
- Exercise keyboard edges, held state, mouse motion/buttons/wheel, focus loss, and close in
  the real client. Exercise reset when the client exposes it; protocol-only acceptance does
  not prove that a supported reset is reachable by the user.
- Abort negotiation and close or disconnect the client. A failed negotiation must release
  any single-client reservation. Record the target's disconnect policy: on PR #506, close
  ends the run. If reconnect is claimed, prove it; otherwise prove a fresh run can start
  without a stale peer reservation.
- When reset is supported, it must rebuild per-run cache and RNG state, clear controls, and
  present the original seed/bootstrap result again. Compare that result and the first
  generated action with a file-mode run to separate codec/presentation corruption from
  model rollout drift.

Responsive controls prove event delivery, not visual correctness or seed fidelity.

## Phase 5: test in increasing cost order

Every pytest test gets exactly one repository marker: `ci_cpu`, `ci_gpu`, or `manual`.

1. **CPU contract tests:** factory type, entry-point metadata, cheap `session_desc`,
   argument split, validation before model load, shared model across sessions, distinct
   per-session state, and cleanup on failure.
2. **CPU loop tests:** channel-list contract, frame count/shape/layout/range, step order,
   finite completion, reset, input mapping, metrics, and `invoke_async` where used.
3. **CPU stand-in end-to-end:** run through `run_session`; for T2V use
   `flashdreams.t2v_v2.testing` and its fake pipeline. If `ffmpeg` is available, write an
   MP4 and assert the exact frame-count formula, including bootstrap/seed frames, plus
   non-empty, changing imagery.
4. **Checkpoint tests:** meta-tensor remap bijection and real-key spot checks; make large
   downloads opt-in.
5. **GPU rollout ladder:** load the full checkpoint with strict missing/unexpected-key
   checks and clean it up; run a minimal eager block, then multiple blocks with default
   guidance; exercise production V2 sinks; then test supported offload and compilation
   modes separately. Reuse one loaded application across fresh sessions with diverse
   available inputs. For stateful or streaming models, include at least one long practical
   supported horizon. Check exact output counts, ordered finite metrics, nonblank/changing
   results, cache longevity, session isolation, latency drift, and peak-memory growth.
6. **Mode parity:** use matched inputs and seeds. Require exact resident/offload output
   only when the execution graph, kernels, and precision are unchanged; otherwise record
   explicit tolerances. Compare eager and compiled flows with maximum and mean error
   bounds. Initialize the selected CUDA context before resetting per-run peak statistics.
   After `close()`, verify application/loop resources are released while distinguishing
   live tensor allocations from memory retained by the CUDA allocator.
7. **Upstream parity:** use the same checkpoint, input, seed, precision, scheduler, and
   step count. Run the upstream production inference path, including required patches,
   compiled attention, or cache-finalization behavior; a debug/eager fallback may be
   approximate rather than ground truth. For stateful or multimodal models, compare the
   closest useful pre-decoder flow boundary available in both implementations across each
   active modality, guidance branch, scheduler transition, and cache commit. Instrument
   intermediate tensors to identify the first divergence before interpreting final error.
   Report agreed metrics and tolerances. Visual review or finite final pixels alone are
   not parity.
8. **Performance:** compare matched stacks, discard compile/autotune warmup, and separate
   model, decode, transfer, presentation, and encode time.

Keep a reproducible evidence note with immutable revisions, input hashes, hardware and
software versions, exact commands, acceptance thresholds, outcomes, and claim limits.

Exercise application help through the installed entry point. Outer runtime validation may
run before delegated application parsing, so supply any required valid runtime arguments
on the target branch rather than assuming `SLUG -- --help` always short-circuits:

```bash
flashdreams-run-v2 <slug> --output-path /tmp/<slug>-help-unused.mp4 -- --help
```

A representative file run is:

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
- [ ] The V2 adapter depends on, rather than duplicates or accidentally deletes, any
      reusable model package; browser-enabled packages include the serving extra.
- [ ] Checkpoint keys and shapes form a complete bijection, or exceptions are proven.
- [ ] `create_app()` returns `IApplication` and is registered under
      `flashdreams.applications_v2`.
- [ ] Application, session, model-loop, optional UI-loop, reset, and cleanup ownership are
      tested with stand-ins on CPU.
- [ ] Every returned channel satisfies `StepResult` shape, layout, range, frame-count, and
      metrics rules.
- [ ] Authoritative decoder mappings, with a real probe where needed, prove the declared
      dimensions and frame counts.
- [ ] Deterministic/MP4 runs preserve every frame with safe presentation settings.
- [ ] Bootstrap-step input ownership is explicit and tested; no live events disappear
      before the first generated action.
- [ ] Interactive claims are validated through a real client path, including UDP ICE,
      controls, focus/disconnect, and aborted negotiation; reset or reconnect is exercised
      when the integration exposes or claims it.
- [ ] Unsupported modalities have a landed generic V2 contract or are explicitly scoped
      out; they are not hidden in video tensors or metadata.
- [ ] The staged GPU rollout covers eager, multi-block, sink, and every supported offload
      or compiled path; mode and upstream parity meet recorded tolerances and cleanup is
      verified.
- [ ] Fresh-session rollouts demonstrate state isolation and exact output accounting;
      stateful or streaming apps also pass a practical long-horizon run with finite metrics,
      stable cache behavior, and bounded latency/memory drift.
- [ ] Reproduction evidence pins inputs, revisions, environment, commands, and claim scope.
- [ ] CI-pinned lint, type checks, and correctly marked tests pass.
