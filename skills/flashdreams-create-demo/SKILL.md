---
name: flashdreams-create-demo
description: Build or modify a model-independent FlashDreams V2 demo application with sessions, model/UI loops, interactive controls, and stub-based tests. Use for a new interaction or application lifecycle change. For a new model binding to an existing demo, use the model integration workflow instead.
license: Apache-2.0
metadata:
  author: "FlashDreams Team <flashdreams@nvidia.com>"
---

# Create a FlashDreams demo

Resolve source paths from the user's checkout at the requested revision.
For design reviews, inspect the proposal without running a model.

1. Inspect the [demo gallery](https://flashdreams.org/main/demos/) and `apps/`.
   If an existing interaction fits, add a model adapter rather than another app.
2. Follow `docs/source/documentation/demo_api/guides/create_demo.md`
   ([create a demo](https://flashdreams.org/main/documentation/demo_api/guides/create_demo/))
   and its `apps/minimal_t2v/` examples. Consult the linked lifecycle/threading
   contracts and reuse the runtime's serving and presentation support.
   Keep the app model-independent; defaults and factory registration belong to
   the integration's [adapter](https://flashdreams.org/main/documentation/demo_api/guides/integrate_model_adapter_for_a_demo/).
3. Follow the guide's stub tests and documentation steps. Check lifecycle, reset,
   controls and output contracts before using a real model. See
   `docs/source/documentation/tools/testing.md`
   ([testing](https://flashdreams.org/main/documentation/tools/testing/)) for
   test placement and CPU/GPU markers. Report stub, generation and UI checks
   separately; help and preload alone do not prove that a demo generates frames.
