---
name: flashdreams-integrate-model
description: Add or migrate a video or world model to FlashDreams V2, including recipe reuse, checkpoint conversion, pipeline configuration, and an adapter for an existing demo. Use when porting an external model or wiring its application entry point; creating a new model-independent demo and running an already supported preset are separate tasks.
license: Apache-2.0
metadata:
  author: "FlashDreams Team <flashdreams@nvidia.com>"
---

# Integrate a FlashDreams model

Use the user's checkout at the requested revision. Paths below are relative to
that checkout; published links are available for separately installed skills.
For a review, inspect the supplied proposal and reference code without installing
GPU dependencies or creating an integration the user did not request.

1. Follow `docs/source/documentation/inferencing_api/guides/create_model.md`
   ([model guide](https://flashdreams.org/main/documentation/inferencing_api/guides/create_model/)).
   Establish the upstream revision and parity reference, then reuse the closest
   recipe. Keep model code, configuration, checkpoint conversion and tests in
   `integrations_v2/<model>/`; shared framework code must not import the model.
2. For a demo binding, follow
   `docs/source/documentation/demo_api/guides/integrate_model_adapter_for_a_demo.md`
   ([adapter guide](https://flashdreams.org/main/documentation/demo_api/guides/integrate_model_adapter_for_a_demo/)).
   The integration owns `apps/<demo>/adapter.py`, depends on the reusable app,
   and registers a zero-argument factory under `flashdreams.applications_v2`.
   Check the package's dependencies and wheel contents as well as its entry point.
   V2 bindings do not use the legacy runner registry.
3. Follow the application lifecycle in the
   [demo guide](https://flashdreams.org/main/documentation/demo_api/guides/create_demo/).
   Factories and descriptions stay cheap; one-time preparation must finish
   before `IApplication.init` returns. App tests use stubs, while model/adapter
   checks belong to the integration.
4. Validate the actual checkpoint transform on metadata or meta tensors before
   GPU loading, including duplicate destination keys as well as keys and shapes.
   Follow the model guide's CPU checks and upstream parity checks; a parser check
   or plausible video alone does not establish numerical parity.
5. Update the model page, registered presets and navigation as directed by the
   guides. Report completed checks, missing hardware-dependent evidence, and
   any dependency or attribution changes for contributor review.
