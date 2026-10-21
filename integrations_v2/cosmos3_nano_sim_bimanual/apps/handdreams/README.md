<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Cosmos HandDreams adapter

Registers `handdreams` using the reusable
[HandDreams app](../../../../apps/handdreams/README.md).
This adapter owns model defaults/CLI validation, scene encoding, action encoding,
and rollout construction. Controller calibration and UI remain in the app;
inference remains in the Cosmos model pipeline.

## Single-command deployment

Follow the [HandDreams deployment guide](../../../../apps/handdreams/README.md#run-one-command-per-machine)
for the shared server, server-attached Quest, and laptop `remote-vr` commands.
The same server also accepts [browser-only IWER input](../../../../apps/handdreams/README.md#browser-only-virtual-controllers).
The app guide covers EULA acceptance, credentials, physical wake, browser
permissions, and calibration.
Both physical Quest routes use
[USB-local transport](../../../../apps/handdreams/README.md#quest-usb-transport)
with cached browser assets and local coturn. No transport selector or
Quest-reachable host address is needed; browser-only IWER remains available.

Launch directly through the standard V2 runtime:

```bash
uv run --package flashdreams-cosmos3-nano-sim-bimanual \
  flashdreams-run-v2 handdreams --mode webrtc \
  --isaac-teleop-accept-eula
```

The controller plugin prepares private TLS credentials; the adapter owns the
performance preset. `launch.py server` remains a compatibility wrapper, not a
required launch path. The remote laptop's `remote-vr` command is unchanged.

The server's `--mode webrtc` default serves the browser operator viewer.
Use `--mode native-window` to show the same UI in a local window on the model
host's graphical desktop, using FlashDreams' existing SlangPy/Vulkan support.
The app resolves the standard display names to its controller-enabled plugins;
the runtime still creates and manages the selected native/WebRTC display.
The default address is loopback for local use, including a local USB Quest
detected at startup. Set `--controller-host <DGX-LAN-IP>` before `--` to accept the unchanged laptop
`remote-vr` command or remote Isaac Teleop/IWER browsers with either output mode.
See [offline native-window startup](../../../../apps/handdreams/README.md#offline-native-window-run)
for a cached-only invocation without a host LAN connection.
`--quest-device SERIAL` selects an
explicit headset without disabling browser input. CloudXR starts before the
headset wake gate, so IWER stays available during the power-button wait or a
local-headset setup failure. Controller connection status appears in either UI.
Native output serves no operator browser URL or HTTP health endpoint; `--port`
only applies to WebRTC output. Both output modes show the Isaac Teleop/IWER link.

On CUDA, the adapter preset enables the existing block/chunk cache, overlapping
cache update, default-mode denoise compilation, valid-length attention, and CUDA
graphs with two graph warmups, alongside checkpoint, decoder, and chunk/window
defaults. Explicit application arguments can override them. For eager debugging,
pass `--no-compile-denoise --no-cuda-graph-denoise --no-valid-length-cache-attention`
after `--`. The generic model baseline is unchanged.

With `--device cpu`, compilation, overlapping cache updates, valid-length
attention, and CUDA graphs default off, with no automatic compile warmup.
Explicit optimization flags still take precedence and incompatible combinations
remain validation errors. This is a model-configuration/debugging option, not
CPU support for CloudXR, which still requires an NVIDIA GPU and Vulkan.

Use `--checkpoint /path/to/export` for a local export. The default checkpoint
alias uses the pinned Hugging Face export described in the [model guide](../../README.md).
Pass an existing `cosmos_interactive/` export directory to reuse it; no copying
or redownload is needed for a complete export. Model and scene arguments belong
after `--`, for example `--sample my-scene`.

Scene data is not included. Without a local scene override, initialization
downloads the three selected scenes from the original gated
[AgiBotWorld-Beta dataset](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta).
The model host uses **your own** `HF_TOKEN` or `hf auth login` credentials, just
like the other apps' Hugging Face downloads. Your account must have dataset
access. There are no embedded credentials or fallback copies of the data.

| Scene name | Public task / episode | Source video frame |
| --- | --- | --- |
| `fold-tshirt` | 570 / 816412 | 0 |
| `fold-shorts` | 599 / 822065 | 0 |
| `open-tissue-box` | 522 / 754102 | 0 |

The fetch list pins dataset revision
`2252b3fdc88dc67fa5d5e4b771e1b86f5e71c278` and exact archive members with
SHA-256 checksums. Only selected video/state/calibration members are transferred,
not the complete task archives. Task annotations use `hf_hub_download`;
selected archive members and converted scenes use the Hugging Face assets cache
(`HF_HOME` / `HF_ASSETS_CACHE`). Each prepared catalog records its source and
output hashes in `provenance.json`. A complete cache works offline.
Cached source files can also rebuild the converted scenes without credentials.
Scene preparation automatically converts raw actuator positions using each
episode's gripper hardware: Robotiq is 35=open/120=closed; OmniPicker is
0=open/120=closed. The versioned cache rebuilds scenes after conversion changes.
Public-dataset users do not run a separate calibration or migration command.
Downloads honor `FLASHDREAMS_MIN_CACHE_FREE_GB` and report cache-space failures
with the `HF_HOME` / `HF_ASSETS_CACHE` settings.
Prompts use the public action annotation covering the selected frame, falling
back to the public task name when the frame precedes the first annotation.

The dataset currently lists **CC BY-NC-SA 4.0**. Authentication and local
conversion do not grant additional usage or redistribution rights. Review
the upstream terms and obtain any necessary approval for your intended use.
No downloaded images, videos, state arrays, or generated calibration files
are distributed with the repository.

`--sample-dir /path/to/catalog` or `--scene-path /path/to/one-scene` still loads
your local files without fetching defaults. Paths refer to the model host,
not the remote controller client. `--sample <scene-name>` after `--` chooses
the initial scene.

## Preparation and warmup

With `--compile-denoise`, each scene runs hidden model/decoder chunks
during server initialization, before controller calibration. Automatic warmup
covers the chosen compilation/CUDA-graph paths. Hidden video is discarded,
rollout state resets to frame zero, and compiled transformer caches are retained.
Decoder history and controller input start fresh for the visible session.

`--compile-prewarm-steps N` overrides the count; `0` disables hidden warmup.
Without denoise compilation, hidden warmup remains disabled. Every selectable
scene is prepared during initialization, before the picker or calibration becomes
available. Resets and scene changes consume prepared state rather than performing
first-use setup. Both display modes use the same preparation path.
Retaining prepared scene caches increases startup time and GPU memory use; use
`--scene-path` to prepare only one scene. Existing `--prewarm-all-scenes` options
remain accepted but are no longer required.

Compilation is enabled by default. Optional postprocessing prepares separately
during session creation.

### Stationary live controls

Live VR automatically uses camera domain `2` for settled controller rows,
retaining normalized fixed-head-camera channels `0:9` and zeroing hand/gripper
channels `9:64`. Motion or trigger updates use manipulation domain `15` with the
normal calibrated actions. No extra server or laptop flag is required. Recorded
inputs are not automatically relabeled.

Both wrists must satisfy `max(translation_m, 0.3 * rotation_rad) <= 0.001` for
11 consecutive action rows. A gripper-openness change above `0.002` excludes the
current and next 15 rows; motion or excluded rows restart the 11-row count.
History advances only when diffusion accepts a chunk and resets with the rollout.
The camera stays fixed; actual head movement is not encoded.

This implements the training team's stationary-to-navigation recommendation
causally, without added input buffering. Unlike offline training labels, it
cannot relabel earlier rows or look ahead around future gripper changes.

The threshold is per row, not accumulated displacement: continuous motion below
1 mm per row can remain camera-only even as the wrist moves farther overall.
For example, 0.9 mm per row at 30 Hz is 2.7 cm/s. This responsiveness tradeoff
preserves the training rule used to reduce unintended gripper motion; a
cumulative-motion guard would change that rule.

## Scene format

Each scene is a directory containing exactly two data files:

```text
<scene-name>/
  first_frame.jpg
  scene.json
```

`scene.json` has these fields:

| Field | Meaning |
| --- | --- |
| `schema_version` | `1`. |
| `title` | Human-readable title; used for scene-selector cards. |
| `prompt` | Model conditioning text; tokenized at preparation time. |
| `fps` | Positive model-conditioning and default controller-sampling cadence, independent of UI refresh. |
| `image_size` | `[image_height, image_width, content_height, content_width]`; content is the top-left region before training-canvas padding. |
| `wrist_poses` | `left` and `right` initial 4×4 wrist-to-camera rigid transforms. |
| `gripper_open` | `left` and `right` initial openness in `[0, 1]`. |

The GA adapter resizes the visible content to 640×360 and reflection-pads the
bottom/right to the training canvas (832×480) before VAE encoding. It retains
the top-left 22×40 latent cells, matching training's floor division by 16. These
form 220 spatial tokens (11×20) and decode to **640×352** video. Scene previews
crop the bottom eight rows of the 640×360 content without stretching it; the
full content still reaches the encoder, and controller calibration is unchanged.
Prompts use the scene caption without an appended resolution sentence. Domain
and action normalization remain model constants, not duplicated scene data.
Initial poses and gripper values must describe the image; they are used for
controller calibration.

Live controller sampling defaults to `scene.json`'s `fps` (30 for the Beta
scenes). `--control-fps N` after `--` overrides both sampling and model temporal
conditioning before warmup. It does not change the operator UI refresh rate.

Use `--scene-path /path/to/<scene-name>` for one local scene, or
`--sample-dir /path/to/catalog` for a directory of scene folders. These options
are mutually exclusive and go after the V2 CLI's `--` separator.
Missing explicit local data and unknown scene names fail before model
preparation. Scene cards use the prepared title and first image. There is no
packaged-data fallback.

The Beta converter extracts the selected original video frame and reads its
synchronized end poses, gripper state, and per-frame head-camera calibration.
It creates the same JPEG/JSON format in the user's cache. The camera transform
comes from the public episode's robot calibration; it is not a copy of the
old bundled scenes' fixed calibration. Hardware/model quality should be
validated independently of download and conversion correctness.

The HDF5 reader adds `h5py` to this integration's dependencies. Its lockfile
and attribution are included; the release owner must complete the repository's
OSRB dependency approval before release.

There are no recorded actions, videos, or token arrays in this scene format, and
no `--sample-pickle` compatibility path. Use Isaac Teleop/IWER virtual controllers
for interactive headset-free testing, or the separate recorded-action replay
below for model-only testing. Hidden warmup uses synthetic actions and does not
require a recorded trajectory.

File loading is only part of preparation. Loading model/VAE weights, encoding
the first image, seeding inference caches, and compiling/warming model kernels
remain necessary. Removing trajectory data does not bypass those stages.

CPU tests cover launcher commands/security, lifecycle, actions, and reset.
Hardware testing is still required for GPU compilation latency, WebRTC/native
presentation, and Quest/CloudXR behavior.

## Recorded-action replay

`handdreams-replay` tests the same GA inference pipeline without controller
input, calibration, Isaac Teleop, or CloudXR. It works on a model GPU without
NVENC. It uses the standard V2 output modes, not the live demo's controller
window plugins. A GPU capable of running the model is still required.

### Public dataset scenarios

Enable replay for **Fold the T-shirt** (`fold-tshirt`):

```bash
uv run --no-project --python 3.12 python \
  integrations_v2/cosmos3_nano_sim_bimanual/apps/handdreams/launch.py server \
  --address "<SERVER_IP>" --replay
```

Open `http://<SERVER_IP>:8081/` and select **Fold the T-shirt**, the only replay
scene. It uses its matching recorded actions. Press Escape to return to the scene
picker. The normal live scene catalog is unchanged. Add
`--mode native-window` for a local window. No Isaac Teleop setup, EULA flag,
headset, or second client is needed.

On first use, the shared scene cache downloads the three catalog episodes from the
pinned AgiBotWorld-Beta dataset using **your Hugging Face credentials** and prepares
their first frames and scenes. Replay prepares normalized actions and model state
only for `fold-tshirt`. Accept the dataset's access terms and log in with
`hf auth login` (or set your own `HF_TOKEN`) first.
Source downloads are cached and can be reused offline; actions are converted in
memory at each startup so conversion changes do not reuse stale actions. Conversion
runs only when replay is requested; normal live startup does not prepare actions.

The recording includes camera/arm poses, grippers, and per-frame domain IDs.
Stationary frames use the training conversion's camera-domain masking, not live
controller smoothing or calibration. Poses use the episode's camera calibration,
not training's fixed robot calibration, so this is not a bit-for-bit reconstruction
of training's action files. A final incomplete action chunk is dropped
with its row count printed. No dataset files are bundled in the repository.

### External normalized recordings

For an already-prepared scene and matching action file:

```bash
uv run --package flashdreams-cosmos3-nano-sim-bimanual \
  flashdreams-run-v2 handdreams-replay \
  --mode webrtc --host 0.0.0.0 --port 8081 -- \
  --scene-path /absolute/path/to/scene \
  --recorded-actions /absolute/path/to/ga_actions.npy
```

For headless testing, use `--mode null`. For a local display, use
`--mode native-window`. To save output, use `--mode mp4 --output-path replay.mp4
--total-model-steps 100` before the separator; the step count is an upper bound.
The replay ends when the input is exhausted and uses the scene's FPS for
presentation (a positive whole number, matching V2's loop cadence contract).
Standard model options such as `--checkpoint`, `--device`, and
`--no-compile-denoise` remain available after `--`.

Supply your own **NPY floating-point `[N,59]` array of normalized GA actions**:

- `unified_v1` ordering, global-asinh normalization, 16-frame backward wrist
  anchors, and gripper widths represented in meters before normalization.
- Mask absent fingertip channels `18:33` and `43:58` to zero **after** normalization.
  The loader adds the five zero padding channels; it does not normalize again.
- Row zero is the transition from the supplied scene image to its next frame.
  The prompt, initial image, action sequence, and scene FPS must match the recording.
- Supply whole chunks: `N` must be a positive multiple of `4 * --chunk-size`
  (eight rows by default). The loader does not silently trim or repeat actions.

The integration's `action_format.normalize_and_pad(raw_ga_actions)[:, :59]`
produces this normalized layout from correctly constructed GA raw actions.
It is **not** a converter for pre-GA recordings. Old 29D/64D actions, scene
pickles, V1 replay archives, raw poses, and unnormalized arrays are not supported.
Shape checks cannot prove training semantics: export with the GA contract above.
External recordings are neither downloaded nor converted, and default to
AgiBot domain 15. These explicit files bypass the public scenario catalog.

The application prepares its model and scenes during initialization and reuses
the existing pipeline's cache, generate, finalize, and reset behavior. It adds
no inference implementation or controller setup.
