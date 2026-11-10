---
name: flashdreams-create-demo
description: Build or modify a model-independent FlashDreams V2 demo application with sessions, model/UI loops, interactive controls, and stub-based tests. Use for a new interaction or application lifecycle change. For a new model binding to an existing demo, use the model integration workflow instead.
license: Apache-2.0
metadata:
  author: "FlashDreams Team <flashdreams@nvidia.com>"
---

# Create a FlashDreams demo

Use the user's checkout and requested revision. Paths below are relative to
that checkout; the published guide is available when this skill is installed
separately. For design reviews, inspect the proposal without running a model.

1. Inspect the [demo gallery](https://flashdreams.org/main/demos/) and `apps/`.
   If an existing interaction fits, add a model adapter rather than another app.
2. Follow `docs/source/documentation/demo_api/guides/create_demo.md`
   ([create a demo](https://flashdreams.org/main/documentation/demo_api/guides/create_demo/))
   and its `apps/minimal_t2v/` source examples. Read the linked API contracts when
   changing lifecycle or threading behavior; reuse the runtime's serving and
   presentation support.
3. Keep the app model-independent and testable with a stub. Model defaults and
   application factory registration belong to the integration's adapter, as
   described in the [adapter guide](https://flashdreams.org/main/documentation/demo_api/guides/integrate_model_adapter_for_a_demo/).
   Complete one-time preparation in application initialization; keep mutable
   rollout state in session loops and queue UI changes on the model thread.
4. Follow the guide's stub tests and documentation steps. Exercise lifecycle,
   reset/session replacement, controls and output contracts before a real model.
   See `docs/source/documentation/tools/testing.md`
   ([testing](https://flashdreams.org/main/documentation/tools/testing/)) for
   test placement and CPU/GPU markers.
5. Report stub checks separately from real generation and UI checks. Application
   help and preload alone do not prove that an interactive demo generates frames.
