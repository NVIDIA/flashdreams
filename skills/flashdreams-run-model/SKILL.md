---
name: flashdreams-run-model
description: Select, install, launch, and troubleshoot an existing FlashDreams video or world-model demo using the V2 CLI. Use for model/demo presets, browser or file output, runtime versus application arguments, and first-run failures. Model porting, new demo development, and offline executable packaging are separate workflows.
license: Apache-2.0
metadata:
  author: "FlashDreams Team <flashdreams@nvidia.com>"
---

# Run a FlashDreams model

Resolve source paths from the user's checkout at the requested revision.
For command or review requests, inspect docs and manifests without installing
packages or launching inference.

1. Select a documented preset from `docs/source/models/` and `docs/source/demos/`
   ([models](https://flashdreams.org/main/models/),
   [demos](https://flashdreams.org/main/demos/)). Follow the model's requirements
   and installation instructions; verify the preset in `integrations_v2/<model>/pyproject.toml`.
2. Follow `docs/source/documentation/cli.md`
   ([CLI reference](https://flashdreams.org/main/documentation/cli/)):
   `flashdreams-run-v2 PRESET RUNTIME_ARGS -- APP_ARGS`. For raw T2V file output,
   pass the app's `--no-ui`; MP4 output alone does not disable UI.
   Choose an explicit output path and run limit for files.
3. In an installed environment, check runtime help and
   `flashdreams-run-v2 PRESET -- --help` for app arguments. Help does not test
   generation. `--preload-application` initializes the installed application and,
   by default, runs one model step; it can load weights and use the GPU.
   Use preloading only for requested execution or preparation checks.
4. For failures, follow `docs/source/troubleshooting.md`
   ([troubleshooting](https://flashdreams.org/main/troubleshooting/)). Check
   preset installation and argument placement before changing dependencies.
   Use the [profiling guide](https://flashdreams.org/main/documentation/demo_api/guides/profiling/)
   for performance investigations.
5. Report the command and checks performed. When execution is requested, inspect
   the output; distinguish parsing, preparation and generation. Use the CLI's
   backpressure/presentation controls when every generated frame must be retained.
