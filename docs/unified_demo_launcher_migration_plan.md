# Unified Demo Launcher Migration Plan

## Goal

Provide one memorable, extensible command for every FlashDreams launch:

```bash
flashdreams-run <runner-slug> [mode] [--manifest PATH] [overrides...]
```

`run` is the default mode. The initial shared modes are `run`, `mp4`, `null`,
`webrtc`, and `local-window`. A runner advertises only the modes it supports.

Examples:

```bash
flashdreams-run lingbot-world-fast webrtc --manifest lingbot-live.yaml
flashdreams-run lingbot-world-fast mp4 --manifest lingbot-replay.yaml
flashdreams-run omnidreams webrtc \
  --manifest omnidreams-drive.yaml
flashdreams-run omnidreams-perf local-window \
  --manifest example_world_model_perf.yaml
```

The runner slug remains the canonical model/preset selector. It is a stable,
fully specified public identity; a second model-name registry is not needed.

## Architecture decisions

- [x] Add a model-agnostic launch capability contract in `flashdreams`.
      The contract must not import from `integrations/`.
- [x] Make integrations register their launch capability through a config slot
      or plugin entry point, preserving `core -> infra -> integrations`.
- [x] Replace the current output-target `module + argv` translation and
      `runpy` handoff with direct, typed launch construction.
- [x] Keep the generic runner path as mode `run`; do not force batch-only
      integrations to implement demo modes.
- [x] Treat `local-window` as a first-class launch mode even when its backend
      remains model-specific.

## Manifest contract

- [x] Add a versioned `FlashDreamsLaunchManifest` rather than extending
      OmniDreams' `WorldModelManifest`, whose fields are intentionally
      OmniDreams-specific.
- [x] The manifest must contain `schema_version`, `runner`, and `mode`, with
      optional `runner_overrides`, `scenario`, and `output` mappings.
- [x] Resolve file paths relative to the manifest directory.
- [x] Reject unknown top-level fields and reject a manifest runner/mode that
      conflicts with explicit command-line selection.
- [x] Make the selected integration validate typed `scenario` and `output`
      sections; generic code must not learn model-specific fields.
- [x] Define and test precedence:

      ```text
      registered runner preset
        < manifest runner_overrides
        < manifest scenario/output settings
        < explicit CLI overrides
      ```

- [x] Support `--no-instantiate` by printing the resolved runner, mode,
      manifest path, scenario, and output without loading a model.

## Central CLI

- [x] Parse `flashdreams-run <runner-slug> [mode]` before building the
      mode-specific typed CLI.
- [x] Make `flashdreams-run <slug> --help` list supported modes and make
      `flashdreams-run <slug> <mode> --help` list only valid options.
- [x] Fail unsupported runner/mode pairs before CUDA initialization.
- [x] Remove the temporary `flashdreams-run --output <mode> <slug>`
      compatibility alias after central mode-dispatch tests pass.
- [x] Retain `torchrun --nproc_per_node=N --no-python flashdreams-run ...`
      as the multi-rank invocation form.

## Integration migration

### OmniDreams

- [x] Move `omnidreams-demo replay`, `omnidreams-demo webrtc`, and
      `interactive-drive` behind an OmniDreams launch capability.
- [x] Support `mp4`, `null`, `webrtc`, and `local-window`.
- [x] Accept existing `example_world_model*.yaml` files through either a
      compatibility reader or converted manifests.
- [x] Preserve scene, weather, camera, conditioning, performance, native
      acceleration, cache, VAE, seed, and post-process controls.
- [ ] Implement shared-runtime equivalents for every legacy WebRTC fallback
      (including multi-rank serving and debug HDMap streaming) before deletion.

### LingBot

- [x] Move `lingbot-demo replay|webrtc` behind a LingBot launch capability.
- [x] Support `mp4` and `webrtc`.
- [x] Cover example assets, prompt, first frame, poses, intrinsics, compile,
      warmup, WebRTC, and context-parallel settings in typed options/manifests.

### Other integrations

- [x] Expose `run` for every registered runner.
- [x] Add other modes only when an integration has a real implementation and
      typed adapter for it.

## Validation and documentation

- [x] Add CPU tests for discovery, help, invalid pairs, manifest validation,
      precedence, relative paths, and `--no-instantiate` output.
- [ ] Add GPU coverage for LingBot MP4/WebRTC and OmniDreams null/MP4/WebRTC/
      local-window, including multi-rank WebRTC where supported.
- [x] Rewrite the quickstart and model docs around the central command.
- [x] Publish one launch-manifest guide with schema, examples, precedence,
      overrides, and reproducibility guidance.
- [x] Replace every demo-launch command in README, docs, CI, and package help.

## Legacy removal gates

- [x] Remove `OutputTargetAdapter`, `OutputTargetSpec`, and the `runpy`
      output-launch bridge after direct launch parity is verified.
- [x] Remove the LingBot and OmniDreams output-target adapters.
- [x] Remove `lingbot-demo`, `omnidreams-demo`, direct WebRTC server launch
      commands, and `interactive-drive` as model-launch entry points.
- [x] Retain preparation, evaluation, and controller-configuration utilities;
      they are not model launch commands.
- [x] Remove compatibility aliases only after their central-command
      replacements have test coverage and documentation.

## Completion criteria

- [x] Every supported demo launches through `flashdreams-run <slug> <mode>`.
- [x] Every supported demo configuration is expressible as a manifest plus
      explicit overrides.
- [x] The central launcher does not invoke a second model CLI.
- [x] No legacy demo/server launch command remains documented or shipped.
- [ ] CPU and GPU tests cover every retained launch mode.
