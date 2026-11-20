---
name: flashdreams-integrate-model
description: Add or migrate a video or world model to FlashDreams V2, including recipe reuse, checkpoint conversion, pipeline configuration, and an adapter for an existing demo. Use when porting an external model or wiring its application entry point; creating a new model-independent demo and running an already supported preset are separate tasks.
license: Apache-2.0
metadata:
  author: "FlashDreams Team <flashdreams@nvidia.com>"
---

# Integrate a FlashDreams model

Resolve source paths from the user's checkout at the requested revision.
For reviews, inspect the proposal and source without installing GPU dependencies
or implementing an integration.

1. Follow `docs/source/documentation/inferencing_api/guides/create_model.md`
   ([model guide](https://flashdreams.org/main/documentation/inferencing_api/guides/create_model/)).
   Establish the upstream revision and parity reference; reuse the closest recipe.
   Keep model-specific code and tests in `integrations_v2/<model>/`.
2. For a demo binding, follow
   `docs/source/documentation/demo_api/guides/integrate_model_adapter_for_a_demo.md`
   ([adapter guide](https://flashdreams.org/main/documentation/demo_api/guides/integrate_model_adapter_for_a_demo/)).
   The integration depends on the reusable app and registers a zero-argument
   factory under `flashdreams.applications_v2`. Check dependencies and wheel
   contents as well as registration. Shared framework and app code must not
   import the integration; V2 bindings do not use the legacy runner registry.
3. Follow the [demo lifecycle guide](https://flashdreams.org/main/documentation/demo_api/guides/create_demo/):
   keep factories and descriptions cheap and finish preparation before
   `IApplication.init` returns. Test apps with stubs and adapters in the integration.
4. Follow the model guide's validation and documentation steps. Check checkpoint
   transforms on metadata or meta tensors, including keys, shapes and duplicate
   destinations, before GPU loading. Report CPU checks and upstream parity
   evidence separately; parser success or plausible video does not prove parity.
   Flag missing checks and dependency or attribution changes for review.
