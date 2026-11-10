---
name: flashdreams-create-demo
description: Build or modify a model-independent FlashDreams V2 demo application with sessions, model/UI loops, interactive controls, and stub-based tests. Use for a new interaction or application lifecycle change. For a new model binding to an existing demo, use the model integration workflow instead.
license: Apache-2.0
metadata:
  author: "FlashDreams Team <flashdreams@nvidia.com>"
---

# Create a FlashDreams demo

## Reuse an interaction when possible

Locate the user's checkout and requested version of
`https://github.com/NVIDIA/flashdreams`. Paths below are relative to its root.
For design or code-review questions, work from the supplied context without
installing packages or running a model. Inspect `apps/` before building a new app:
an existing text, video, camera or action interface may need only a model adapter.

Use `apps/minimal_t2v/` as a compact reference and read the target revision's
`flashdreams/flashdreams/api_v2/{application,session,loop}.py` contracts. Reuse
the runtime's orchestration, windows, video writing and WebRTC implementation.
An app does not create its own generation/presentation threads or serving stack.

## Define an app contract, then its model binding

```text
apps/<demo>/
  pyproject.toml
  <demo>/
    __init__.py
    application.py
    session.py
    model_loop.py
    ui_loop.py          # only for custom controls or composition
  tests/
```

The app depends on `flashdreams` and accepts a model-neutral config or interface.
Keep concrete pipeline construction and model defaults in
`integrations_v2/<model>/apps/<demo>/adapter.py`. That adapter depends on the app
and registers a zero-argument factory under `flashdreams.applications_v2`.
App tests use stubs and must not import `integrations_v2`.

## Implement the lifecycle

| Boundary | Responsibility |
| --- | --- |
| Constructor, `session_desc()`, default mode selection | Cheap configuration and output description; these run before initialization. |
| `IApplication.init(args)` | Parse app arguments; complete downloads, extraction, compilation, native loading and shared model construction before returning. |
| `create_session(desc)` | Check supported dimensions/layout/rate and return an isolated, uninitialized session using prepared resources. |
| `ISession.init()` | Register one model loop; register a UI loop only when custom UI is needed. |
| `IModelLoop.step(index, events)` | Advance model state and return a list of `StepResult`, including an empty list when there is no output. |
| `reset()` / `close()` | Reset per-run caches and controls; release owned resources. |

Keep expensive immutable/shared resources on the application and mutable rollout
state on session loops. Per-prompt cache initialization can happen on the model
thread, but one-time downloads or native builds must not be deferred there.
Wait for preparation workers before `init` returns; starting them there does not
exempt later operations from the preparation guard. When preparation is missing,
fail with a clear error instead of silently fetching.

Return accurate frame counts and layouts with each `StepResult`; preserve channel
ordering for multi-output apps. The model thread owns generation state. UI changes
use `invoke_async` to schedule updates there; capture submitted values so later UI
edits cannot alter a queued request. Model-loop completion does not by itself
end the application. An interactive app can keep a persistent model loop waiting
for prompts, or finish each prompt's model loop and let the UI call
`request_new_session` for the next prompt. Reset mutable rollout state for new
prompts or create a fresh session.

## Check with a stub before using a model

Add `ci_cpu` tests alongside the app that exercise:

- construction and `session_desc()` without loading weights;
- argument parsing, preparation and validation of unsupported session descriptions;
- generated frame counts/layouts and `generate`/`finalize` ordering;
- prompt/control updates, empty output, reset and resource cleanup;
- session replacement without leaking previous rollout state.

Use the existing test helpers and fake pipelines where available. Example:

```bash
uv run --no-sync pytest apps/minimal_t2v/tests -m ci_cpu
uv run --no-sync flashdreams-run-v2 minimal-t2v-wan21 -- --help
```

Replace these paths/preset for the new app. Help is a parser check, not a rollout.
GPU validation is a separate step once a real adapter and environment are ready.
`--preload-application` initializes the application and normally attempts one
model-loop step. That check may produce no frames while waiting for input, so
verify actual generation separately. `--skip-preload-validation` skips the step
check while still performing resource preparation.
Preparation policy is `none|warn|error` through `FLASHDREAMS_PREPARATION_POLICY`;
preload defaults to `warn`. Inspect the reported preparation issues and fix their
source rather than treating skipped validation as a successful runtime test.

## Document the interaction

Put controls and application arguments in `docs/source/demos/<demo>.md` and
installation/presets on the model page. Add the demo to `demos/.nav.yml` and
`demos/index.md`. Runtime flags precede `--`; application flags follow it.
Build the docs when changing website content. Report stub checks separately from
real generation, browser/native-window checks, and performance measurements.

## References

- [Create a demo](https://flashdreams.org/main/documentation/demo_api/guides/create_demo/)
- [Application API](https://flashdreams.org/main/documentation/demo_api/api_reference/application/)
- [Interactive serving](https://flashdreams.org/main/documentation/demo_api/guides/interactive_serving/)
