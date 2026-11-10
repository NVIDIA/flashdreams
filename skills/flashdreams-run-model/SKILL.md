---
name: flashdreams-run-model
description: Select, install, launch, and troubleshoot an existing FlashDreams video or world-model demo using the V2 CLI. Use for model/demo presets, browser or file output, runtime versus application arguments, and first-run failures. Model porting, new demo development, and offline executable packaging are separate workflows.
license: Apache-2.0
metadata:
  author: "FlashDreams Team <flashdreams@nvidia.com>"
---

# Run a FlashDreams model

Use the user's FlashDreams checkout and requested revision. Paths below are
relative to that checkout; the published links also work when this skill is
installed separately. For command or review requests, inspect the docs and
manifests without installing packages or launching inference.

1. Select a supported model/demo pair using `docs/source/models/` and
   `docs/source/demos/` ([models](https://flashdreams.org/main/models/),
   [demos](https://flashdreams.org/main/demos/)). Read the model's requirements,
   install command and preset; verify registration in its
   `integrations_v2/<model>/pyproject.toml`. Do not construct preset names by guesswork.
2. Build the command using `docs/source/documentation/cli.md`
   ([CLI reference](https://flashdreams.org/main/documentation/cli/)) and the
   selected demo's arguments. V2 uses `flashdreams-run-v2 PRESET RUNTIME_ARGS --
   APP_ARGS`. For raw T2V file output, use the app's `--no-ui`; choosing MP4 does
   not disable its UI. Choose an explicit output path and run limit for files.
3. In an already installed environment, use runtime help and
   `flashdreams-run-v2 PRESET -- --help` to check arguments. Application help is
   not a GPU dry run or generation test; preload can load weights and execute
   model code. Check the documented package/extras before changing dependencies.
4. For failures, follow `docs/source/troubleshooting.md`
   ([troubleshooting](https://flashdreams.org/main/troubleshooting/)). Check
   preset installation and argument placement before changing the environment.
   For performance investigations, use the maintained
   [profiling guide](https://flashdreams.org/main/documentation/demo_api/guides/profiling/).
5. Report the command and what was actually verified. When execution is requested,
   check the resulting file or interactive output; distinguish parser success,
   preparation, and generation. Use the CLI's backpressure/presentation controls
   when every generated frame must be retained.
