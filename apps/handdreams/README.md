<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# HandDreams

HandDreams is the demo application. The
[Cosmos adapter](../../integrations_v2/cosmos3_nano_sim_bimanual/apps/handdreams/README.md)
owns model configuration, scene encoding, and rollout wiring. FlashDreams V2 owns
the runtime, event delivery, presentation, and WebRTC/native operator displays.

Live input routes are supported for:

- Physical Quest on the model host: Quest Browser → USB/ADB → DGX CloudXR/Isaac Teleop.
- Physical Quest on a laptop: Quest Browser → USB/ADB → laptop CloudXR/Isaac Teleop →
  authenticated controller relay → DGX.
- Desktop virtual controllers: Isaac Teleop/IWER → CloudXR/Isaac Teleop on the
  model host.

One server accepts these clients, using the same calibration, action pipeline,
and operator UI. Choose WebRTC output (the default) or a native window on the
model host with the server's `--mode` option.
Native APK input/build tooling is not part of this V2 demo.

## Set up Isaac Teleop once per user

The HandDreams controller-input package provides the setup command. With the
`flashdreams-controller-input` command installed, run this from any directory on each machine
that runs CloudXR:

```bash
flashdreams-controller-input setup isaac-teleop
```

For a fresh checkout without the public command installed, run from its root:

```bash
python3 integrations_v2/cosmos3_nano_sim_bimanual/apps/handdreams/launch.py setup-isaac-teleop
```

Setup displays `isaacteleop[cloudxr,retargeters-lite]==1.4.142` and the
[NVIDIA CloudXR EULA](https://github.com/NVIDIA/IsaacCapture/blob/main/deps/cloudxr/CLOUDXR_LICENSE),
requires affirmative acceptance, installs the complete dependency closure from
platform-specific hashes into `~/.local/share/flashdreams/isaac-teleop`, and runs
`pip check` plus `doctor`. For unattended setup, `--accept-eula` records the
operator's explicit choice for that setup invocation. Existing matching
environments are checked and reused; mismatched environments are preserved with recovery instructions.
No Isaac Teleop or CloudXR binaries are copied into the FlashDreams release.

Setup, the checkout launcher, and direct V2/relay launches share this per-user
location, so changing the working directory does not change the selected SDK.
For a custom location, run setup with `--directory /absolute/path/to/isaac-teleop`
and launch with `--isaac-teleop-python /absolute/path/to/isaac-teleop/bin/python`.
Existing checkout-local `.venv-isaac-teleop` environments are preserved and are
not migrated automatically. To keep using one, pass
`--isaac-teleop-python /absolute/path/to/checkout/.venv-isaac-teleop/bin/python`.

See the [controller setup guide](controller_input/README.md#install-the-isolated-sdk)
for validation and recovery. Setup does not grant acceptance for runtime
startup: pass `--isaac-teleop-accept-eula` to the V2 runtime or `--accept-eula`
to the remote client launcher below.

## Run: one command per machine

Run from a checkout with `uv` and OpenSSL on PATH. Use the normal
FlashDreams GPU/driver and checkpoint-access prerequisites on the model host.
By default the server also starts CloudXR, so it needs supported Linux, Python 3.12,
glibc 2.35+, an NVIDIA GPU/driver with NVENC, and Vulkan. For a model host without
NVENC (such as a B300-only DGX), use the relay-only command below.
Use `flashdreams-run-v2` directly, like the other V2 apps. The application adds
controller input to the selected standard display; the V2 runtime owns the
window, session loop, and shutdown. No source-tree server launcher is required.

On DGX (replace the address and GPU as needed):

```bash
CUDA_VISIBLE_DEVICES=1 uv run --package flashdreams-cosmos3-nano-sim-bimanual \
  flashdreams-run-v2 handdreams \
  --mode webrtc --host 0.0.0.0 --port 8081 \
  --controller-host 10.111.69.99 --isaac-teleop-accept-eula -- \
  --sample-dir /absolute/path/to/scenarios
```

For a **relay-only model host**, disable local CloudXR before the `--` separator:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --package flashdreams-cosmos3-nano-sim-bimanual \
  flashdreams-run-v2 handdreams \
  --mode webrtc --host 0.0.0.0 --port 8081 \
  --controller-host 10.111.69.99 --no-local-cloudxr -- \
  --sample-dir /absolute/path/to/scenarios
```

The laptop's `remote-vr` command is unchanged and still runs CloudXR on an
NVENC-capable GPU. The model host retains authenticated controller input and
the calibration service, but starts no local CloudXR or headset automation and
shows no local IWER link. The calibration service still needs its configured
Python environment; it does not use NVENC. Without any controllers, use the
[recorded-action replay](../../integrations_v2/cosmos3_nano_sim_bimanual/apps/handdreams/README.md#recorded-action-replay)
instead; that route starts no calibration or Isaac services.

No scenario data is included in the package. Supply a directory containing
`<scene-name>/first_frame.jpg` and `<scene-name>/scene.json` for each scene, or use
`--scene-path /absolute/path/to/one-scene`. These paths are on the model host;
remote clients do not need the scene files. See [Scene assets](#scene-assets).

By default (`--mode webrtc`), the runtime prints the operator viewer URL.
Both output modes show the Isaac Teleop/IWER link in the UI, even when a local
Quest is connected. Keep this server running for
the input routes.
With local CloudXR enabled, startup prepares the Isaac Teleop web client in
`~/.cloudxr/static-client/`, or `TELEOP_WEB_CLIENT_STATIC_DIR` when set.
The first run downloads missing assets through the selected Isaac Teleop SDK;
a complete cache is reused without network access. This applies to both the
direct V2 command and `launch.py server`. If preparation fails, startup reports
the error instead of switching to an external client.
For browser-only IWER, skip the laptop command and follow
[Browser-only virtual controllers](#browser-only-virtual-controllers).

For native output, run the server from the model host's graphical desktop:

```bash
CUDA_VISIBLE_DEVICES=1 uv run --package flashdreams-cosmos3-nano-sim-bimanual \
  flashdreams-run-v2 handdreams \
  --mode native-window --isaac-teleop-accept-eula -- \
  --sample-dir /absolute/path/to/scenarios
```

The native window shows the same scene picker, calibration, video, and XR CONTROL
inset. It requires a working graphical display and the existing SlangPy/Vulkan
native-window support on the model host. A headless SSH session alone cannot
present that window. This mode starts no operator browser viewer or `/healthz`
endpoint; `--port` only controls the WebRTC viewer. Controller input still uses
CloudXR/Isaac Teleop and the authenticated relay.
The address defaults to `127.0.0.1` for local use; no host LAN interface is
required for the Python listeners. Set `--controller-host <DGX-LAN-IP>` to accept remote
controller clients with either display mode, using the same `remote-vr` laptop
command below. An explicitly supplied address must still belong to the host;
remove an old LAN address when running disconnected.
Display and controller options go before `--`; scene/model options go after it.
`--mode` belongs on the V2 runtime command, not `remote-vr`.

### Offline native-window run

After a successful online run has prepared this checkout and cached its assets,
connect the Quest by USB and join an offline Wi-Fi access point on the Quest.
That AP does not need internet or a connection to the DGX: the managed browser
page, signaling, and controller stream use ADB/USB.

Run from the DGX's graphical desktop, omitting `--controller-host`:

```bash
env -u QUEST_META_PIN -u TELEOP_STREAM_SERVER_IP \
  UV_OFFLINE=1 UV_NO_SYNC=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 \
  uv run --package flashdreams-cosmos3-nano-sim-bimanual \
  flashdreams-run-v2 handdreams \
  --mode native-window --isaac-teleop-accept-eula --quest-accept-server-certificate -- \
  --sample-dir /absolute/path/to/scenarios \
  --prewarm-all-scenes
```

These environment variables reuse the installed application environment without
syncing dependencies and make Hugging Face use its local cache. Keep the same
cache directories as the online run. Besides the main checkpoint, startup needs
the Qwen tokenizer, Wan2.2 VAE, cached TAEHV2 Super decoder weights,
installed Isaac Teleop/ADB/coturn, and all three cached
WebXR client files. Missing assets still require preparation while online; this
command cannot provision a new machine offline.

Local controller credentials live in `~/.config/flashdreams/quest-relay-127.0.0.1/`,
leaving the LAN credentials used by `remote-vr` unchanged. A remote laptop still
needs a reachable DGX LAN address, though that LAN need not have internet.
The UI links to the locally hosted Isaac Teleop client when its assets are
installed. This works without internet for both IWER and the USB headset path.
The fallback NVIDIA-hosted link requires internet.

### Server-attached and remote headsets

The server detects a single authorized USB Quest at startup. To automate its
browser with the local development certificate, run:

```bash
env -u QUEST_META_PIN CUDA_VISIBLE_DEVICES=1 \
  uv run --package flashdreams-cosmos3-nano-sim-bimanual \
  flashdreams-run-v2 handdreams \
  --mode webrtc --host 0.0.0.0 --port 8081 --controller-host 10.111.69.99 \
  --isaac-teleop-accept-eula --quest-accept-server-certificate -- \
  --sample-dir /absolute/path/to/scenarios
```

Add `--mode native-window` to this server command to use native output with
the directly attached Quest; omit it or use `--mode webrtc` for browser output.

Use `--quest-device SERIAL` to choose a headset explicitly, for example when
multiple authorized headsets are connected. Without a headset, or when discovery
is ambiguous, the server stays available for IWER and the remote relay. Discovery
runs once at startup; connect and authorize the headset before launching.

This uses the server's single CloudXR/Isaac runtime and manages the selected
Quest browser over ADB. It does not start a relay or connect over SSH. Install
ADB and coturn on the model host and authorize the headset there. Managed Quest
connections always use [USB-local transport](#quest-usb-transport), including
the client page, signaling, and streaming. No transport selector is needed.
The shared CloudXR runtime starts before the local headset wake gate. IWER stays
available while the headset sleeps or if optional local headset setup fails.
The server proxy listens on the selected IPv4 address and loopback: IWER connects
over the host network, while the managed Quest keeps its USB loopback endpoint.
`remote-vr` keeps its own CloudXR proxy loopback-only.
Controller input requires a concrete `--controller-host` interface
or hostname; `0.0.0.0` is rejected even with a separate advertised CloudXR address.
The managed browser waits for a physical headset wake by default. Use
`--no-quest-require-physical-wake` before `--` to skip the gate for an already-running
server-attached test (`--no-require-physical-wake` on `remote-vr`).
The operator UI reports local headset startup, the power-button prompt, the
Isaac Teleop web client URL, and the active controller source in both output modes.
Headset passcodes, privacy screens, immersive-VR Allow, and local-network
permissions may still require manual approval.

For a physical Quest attached to a separate VR client machine, such as your laptop:

```bash
env -u QUEST_META_PIN python3 integrations_v2/cosmos3_nano_sim_bimanual/apps/handdreams/launch.py remote-vr \
  --server nvidia@10.111.69.99 \
  --accept-eula --accept-server-certificate
```

The `remote-vr` command installs/reuses controller support in a separate Python
3.12 environment and invokes the SDK from
`~/.local/share/flashdreams/isaac-teleop`. It selects the single authorized ADB
device; use `--quest-device SERIAL` when needed.
The server command installs its workspace dependencies and reuses the existing
checkpoint cache. Initial dependency/model downloads still need network access
and any required credentials.

The server generates controller TLS credentials once under
`~/.config/flashdreams/quest-relay/`. The laptop obtains **only the certificate
and token** over verified SSH and reuses a private per-server copy. SSH access
and the server's host identity must already be trusted; the launcher does not
accept unknown SSH keys or replace existing credentials. No manual token export,
certificate copy, or additional credential setup is needed.

`--server nvidia@dgx.example.com` also works when the hostname resolves on the
laptop to exactly one IPv4 address matching the server's `--controller-host`. SSH uses
the original hostname; certificate verification and the controller connection
use the same resolved IP. Existing server certificates do not need replacement.
Names with multiple IPv4 addresses require an explicit IP instead. SSH-only
aliases and jump-host routing do not configure the controller connection:
the server's controller port must be directly reachable from the laptop.

CloudXR's headset-status/configuration API is unauthenticated; use an isolated
or trusted demo network. The optional SDK `CONTROL_TOKEN` is not supported and
inherited values are ignored. This does not disable the separate authenticated
laptop-to-server controller relay.

The server requires `--isaac-teleop-accept-eula`, and `remote-vr` requires
`--accept-eula`; pass these only after accepting the
[NVIDIA CloudXR EULA](https://github.com/NVIDIA/IsaacCapture/blob/main/deps/cloudxr/CLOUDXR_LICENSE)
for `isaacteleop[cloudxr,retargeters-lite]==1.4.142`.
`--quest-accept-server-certificate` on the server (`--accept-server-certificate`
on `remote-vr`) temporarily permits certificate errors for the
local CloudXR origin in the managed Quest tab. It defaults to disabled and does
not install permanent browser trust; keep it on each automated run unless the
browser already trusts that origin. This bypass is origin-scoped, not certificate
fingerprint-pinned. It does not change the DGX controller channel's TLS verification,
other browser origins, or the separate immersive-VR Allow permission.
Certificate and wake options also apply to an automatically detected local Quest;
they do not require `--quest-device` and do not disable IWER.

The laptop route needs supported Linux with an NVIDIA GPU/driver,
Vulkan, ADB, coturn, and a USB-authorized Quest. Quest does not need Wi-Fi access
to the laptop or DGX, or internet access. It still needs an active Wi-Fi IP for
WebRTC; an offline access point is sufficient.

The `remote-vr` command confirms sleep and waits for a power-button wake before
opening the Quest tab and connecting through the USB loopback endpoint.
Clear any headset passcode/privacy or browser permission prompt if one appears.
For an already-running test, `--no-require-physical-wake` skips that gate.

With `--mode webrtc`, open **http://10.111.69.99:8081/** for generated video;
the viewer needs working WebRTC media connectivity, and loading its HTTP page
alone does not establish video connectivity. With `--mode native-window`,
watch the window on the model host. The laptop needs access to DGX controller
TCP port `8445` for either display mode.
The same server also listens for direct CloudXR clients on TCP `48322` by
default; change that with `--quest-cloudxr-port` before `--`. Restrict these
listeners and their media traffic to the intended demo network.

## Quest USB transport

Both managed physical Quest routes carry the local WebXR client page, signaling,
and TURN-relayed media over ADB reverse connections. The old `--usb-local` and
`--cloudxr-host` selectors have been removed; USB-local is the only managed path.

Install `coturn` on the machine with the Quest (`sudo pacman -S --needed coturn`
on Arch Linux). `remote-vr` automatically reuses a complete local
web-client cache or copies the matching Isaac Teleop 1.4 assets from the DGX's
`~/.cloudxr/static-client/` through verified SSH. No separate copy command is
needed. The local cache is `~/.cloudxr/static-client/`, or the directory set by
`TELEOP_WEB_CLIENT_STATIC_DIR`. It must contain nonempty `index.html`, `bundle.js`,
and `bundle.emulator.js` files.

Transfers are staged before publication. Existing nonempty files are preserved;
an incomplete cache whose files differ from the server is rejected rather than
mixing client builds. Start the server once with host network access to prepare
its cache before the laptop copies it. Server startup also prepares the local
client for a directly attached DGX Quest; later runs reuse the complete cache.
Preparation runs on the host before launching VR, with no Quest WAN needed;
the running USB-local transport never downloads missing assets.
The launcher starts its own TURN process on an available local port; a system
coturn service is unnecessary and no `systemctl` changes are needed.
The examples omit `QUEST_META_PIN` for this run: PIN-based headset settings
can require internet and are rejected in this mode.
The client is served at `/client/` on the same local HTTPS origin as CloudXR.
Certificate acceptance stays scoped to that origin; browser flags need no changes.

Keep Quest Wi-Fi associated with an access point so the browser has an active
IP address for WebRTC. A dedicated offline AP is sufficient: it needs no WAN
uplink, and Quest does not need to share the laptop/DGX LAN. The laptop still
needs its normal DGX controller connectivity, plus viewer connectivity if
watching WebRTC video there. The headset uses the local USB endpoint automatically.
The upstream client may still attempt optional telemetry/profile requests;
this mode does not install a firewall or grant Quest internet access.
Initial Python/model provisioning can still require host network access.

The wake gate and browser permissions are unchanged. Cleanup removes
only the reverse mappings created for this session and stops its local TURN
process; occupied mappings cause startup to fail without replacing them.

## Browser-only virtual controllers

Use the same running V2 command above. No headset, laptop companion,
ADB, or `remote-vr` command is required for IWER.
An attached Quest does not select a different server route or require an IWER
restart; disconnect the active input before connecting another client.

1. Open `http://10.111.69.99:8081/` for generated video with `--mode webrtc`,
   or watch the model host's native window with `--mode native-window`.
2. Once CloudXR is listening, open its certificate URL
   (`https://10.111.69.99:48322/` by default) and trust the development
   certificate only after checking that it belongs to your model host.
3. Open the UI's Isaac Teleop Web Client link in another desktop tab. It uses
   `https://10.111.69.99:48322/client/` with the model host, CloudXR port, and OOB
   client registration prefilled. Allow local-network access if requested.
4. Use the WebXR `local` reference space, zero pose offsets/prediction, and no
   smoothing. Connect both virtual controllers and release their triggers.

The locally hosted client reports installed IWER explicitly. With both virtual
controllers tracked, hold either trigger for 0.5 seconds, then release both to
start from their current positions; no outline alignment is required. Physical
Quest controllers still require both triggers and the normal alignment stage.
Changing the active connection clears calibration and setup progress.

IWER identification is a client-declared setup hint, not hardware attestation or
an authorization check. A custom CloudXR client can claim it; connect only trusted
clients. The `remote-vr` relay always classifies its input as physical. This demo
generates video and does not command a physical robot.

Startup requires the locally hosted client so the launcher and UI links both
support explicit IWER identification. Missing assets are prepared before
controller startup; a preparation failure stops startup with recovery guidance.
Unknown or ambiguous clients retain full setup. `--auto-start` remains an
explicit override for **all** sources; it is not needed for automatic IWER setup.
The browser needs CloudXR/WebRTC media connectivity to the model host and HTTPS
access to the client.
`XR SESSION ACTIVE` alone does not prove the CloudXR connection is working.
Never use `0.0.0.0` as a client destination.

Only one tracked input source controls the server at a time. A source's lease
expires after one second without tracked input; switching sources resets the
session and calibration so tracking origins cannot be mixed. Disconnect or stop
the active client before switching routes.

## User flow

Physical Quest input uses this guided setup:

1. Choose a scene from the thumbnail strip. Startup and scene loading show a
   neutral panel, not the scene's first RGB frame.
2. Hold the controllers side by side at the same height. Close both claws with
   the **index triggers** (at least `0.9`) for **1 second**.
3. Release both triggers (at most `0.1`) and match the claws to the gray guides.
   Hold the aligned, open pose for **1 second** to start generation.

The trigger stage captures participant direction from left-to-right controller
placement, not pointing direction. Side grips remain clutch controls. The
trigger tutorial uses the controllers' shared horizontal heading as display
forward, preserving relative hand motion and trigger closure. Ambiguous pointing
directions fall back to hand placement; this display reference never affects
calibration. The alignment guide follows the hands' midpoint and compares their
separation and pointing directions; roll around a correctly pointed claw is ignored. Closed
claws turn red. The original tutorial MP4s ship locally and loop during setup.

The bottom-right **XR CONTROL** inset shows live calibrated poses and trigger
closing (left cyan, right orange), independent of generated-video updates.
It uses the accepted alignment reference, movement gain, and scene wrist
orientations with a fixed, zoomed-out view. Geometry outside the inset is cropped,
not pinned to its border. Scene changes and resets clear that reference.
Untracked hands are hidden without shifting the other hand.
This is an input preview, not the stateful clutch/command-shaping output.
The bottom-left stopwatch starts with the first live video frame, excluding setup
and initial generation. It continues through tracking loss and resets with the
scene or session.
The scene picker shows tracking and battery when the source reports it.

Press **Escape** to return to scene selection. The model and controller connection
remain alive. Click a scene to load it, or use left/right keyboard arrows to
highlight a scene and press Enter to prepare it.
Controller B/X/Y buttons are unbound. A reset or scene change
repeats setup. Tracking loss interrupts setup; during live generation it pauses
action delivery without discarding calibration.

Press keyboard **B** in the native window or WebRTC viewer to restart the
server-attached headset's WebXR setup. Its managed VR tab and USB helpers close
before a fresh sleep-to-wake gate: press the headset power button when prompted.
This explicit restart always requires physical wake, even if startup used
`--no-quest-require-physical-wake`. It also retries headset discovery after a failed
setup or disconnected stream, without reloading the model or restarting the
shared Isaac runtime. The controller's B button does nothing;
keyboard B does not restart a remote laptop's headset or disconnect an IWER client.

To inspect a connection failure in the headset, add
`--keep-browser-open-on-error` to `launch.py server` or `launch.py remote-vr`
before any `--` separator. With `flashdreams-run-v2`, use
`--quest-keep-browser-open-on-error` before `--` instead. This opt-in diagnostic
mode keeps the managed Quest tab, USB helpers, and headset wake state after a
browser startup or stream error once the tab has opened; earlier startup failures
still clean up normally. It does not keep the application alive if the shared
CloudXR runtime itself fails.
On the server-attached route, keyboard **B** cleans up and retries. On
`remote-vr`, press Ctrl-C to clean up, then rerun the command. Exiting always
performs normal cleanup; without the flag, failure cleanup is unchanged.

Closing the native window or stopping the demo/relay with Ctrl-C stops owned
CloudXR/browser services, restores managed headset settings, and sends the
managed headset to sleep. Closing only the WebRTC viewer tab leaves the server
running. Scene changes do not repeat the wake gate. After relay disconnection,
rerun `remote-vr`; the server can retain loaded model weights. If USB
loss prevents cleanup, exit immersive VR and put the headset to sleep manually.

## Advanced use

The model adapter supplies checkpoint/decoder defaults and the demo's
compiled/cache/CUDA-graph preset. Every selectable scene is prepared during server
initialization, before the picker or calibration. Scene changes and revisits reuse
prepared state; they never download assets or run first-use preparation.
Preparing the full catalog increases startup time and GPU memory use. Catalogs
are limited to **five scenes**, independently of the three default downloadable
scenes, because prepared scene caches stay resident for the application lifetime.
Larger catalogs fail before the app prepares previews,
controller calibration, or the model; use `--scene-path` for one scene or
`--sample-dir` pointing to a directory containing at most five scenes. This bounds
catalog growth, not the total VRAM required by the model and its configuration.
`--prewarm-all-scenes` remains
accepted for existing scripts but is no longer required. Both `--mode webrtc`
and `--mode native-window` use the same preparation path.
Hidden model warmup requires denoise compilation (enabled by the adapter);
`--compile-prewarm-steps 0` after `--` disables it.
See the [adapter guide](../../integrations_v2/cosmos3_nano_sim_bimanual/apps/handdreams/README.md)
for model/warmup details. Model-specific defaults belong to that integration.

Use `--checkpoint /path/to/export` to override checkpoint selection. The default
alias downloads the GA iteration-1380 model from
`nvidia/Cosmos3-Nano-Sim-Bimanual`, pinned to Hugging Face revision
`304bf3dc45901793f0782254a5270cc087b9a157` (GA, four denoising steps).
Remove pre-GA checkpoint paths and `--denoise-steps 2` overrides when upgrading;
the older weights use an incompatible action format. Private-repository access uses
`HF_TOKEN` or an existing Hugging Face login; never put tokens in scripts or Git.
Keep the same cache directories for offline runs. The core loader stores both
the six shards and a merged checkpoint (~60.7 GB total, plus other assets).

To avoid redownloading an existing converted **GA** export, pass its directory under
`${FLASHDREAMS_CACHE_DIR:-$HOME/.cache/flashdreams}/cosmos_interactive/` explicitly.
Local exports need their index and all referenced safetensors shards. Do not
reuse the older iteration-2000 export: its conditioning format is incompatible.
The TAEHV2 Super decoder downloads its public safetensors weights from the
author's GitHub repository at a pinned commit and reuses the local cache.
The default demo checkpoint and decoder do not require S3 credentials.

The standard `--mode webrtc` and `--mode native-window` choices automatically
attach the unified controller plugin for this app. Existing explicit
`quest-remote-*` modes remain supported; other applications are unchanged.
Omitting the mode defaults to WebRTC. The low-level `isaac-teleop-webrtc` mode
still provides direct Isaac input without the relay or managed headset.

Controller TLS credentials are generated or reused automatically. Explicit
`--controller-cert`, `--controller-key`, and `FLASHDREAMS_CONTROLLER_TOKEN` may
be supplied together to use an operator-managed identity. Viewer `--host` does
not change the controller bind; use a concrete `--controller-host` for LAN input.
Remove stale `TELEOP_STREAM_SERVER_IP` exports from older scripts unless an
explicit CloudXR advertisement override is intended.

The existing `launch.py server` command remains a compatibility wrapper around
the same standard V2 command. `launch.py remote-vr` remains the lightweight
remote-controller companion, not an application runner. Its `--` passes extra
options to the relay, for example `-- --startup-timeout 180`.

For example, warm all scenes with the standard runtime:

```bash
uv run --package flashdreams-cosmos3-nano-sim-bimanual \
  flashdreams-run-v2 handdreams \
  --mode webrtc --host 0.0.0.0 --port 8081 --controller-host 10.111.69.99 \
  --isaac-teleop-accept-eula -- \
  --sample-dir /absolute/path/to/scenarios \
  --prewarm-all-scenes
```

For headset-free testing, connect the Isaac Teleop web client with virtual
controllers as described in [browser-only input](#browser-only-virtual-controllers).
The release demo uses live controller actions, not recorded-trajectory playback.

Optional SwiftVR restoration remains available through
`--postprocess-preset swiftvr-2x-compiled --postprocess-device cuda:1` after
`--`. It is prepared during session setup and can be toggled in the video HUD.
It is not required for either controller route.

For WebRTC startup diagnostics, query `http://127.0.0.1:8081/healthz` on the server.
`"open": false` means application/session preparation is not complete;
model download, encoding, and compilation can take time. `"open": true`
means the output session is ready, not that a viewer has received video.
Native output has no HTTP health endpoint; use the server logs and local window.

## Scene assets

Scenario data is supplied externally, not packaged with the application.
Each scene directory contains `first_frame.jpg` and `scene.json`.
The JSON stores the display title,
model prompt, conditioning FPS, image geometry, and initial wrist poses/gripper
openness needed for calibration. Recorded videos, action trajectories, and
pre-tokenized prompt arrays are not shipped. Setup tutorial MP4s are separate UI
assets and are not model conditioning.

The model adapter reads these two files, tokenizes the prompt, and encodes the
first image. The scene format and local overrides are documented in the
[adapter guide](../../integrations_v2/cosmos3_nano_sim_bimanual/apps/handdreams/README.md#scene-format).
There is no scene pickle loader. The Cosmos adapter can prepare three default
scenes from the original gated AgiBotWorld-Beta dataset using the model host
user's own Hugging Face login. See the
[source list and download requirements](../../integrations_v2/cosmos3_nano_sim_bimanual/apps/handdreams/README.md#single-command-deployment).
Nothing from the dataset is packaged with the demo; complete prepared caches
work offline.

To use your own local catalog instead, pass
`--sample-dir /absolute/path/to/scenarios` after the V2 command's `--`.
The directory contains one subdirectory per scenario, each with `first_frame.jpg`
and `scene.json`. Both the scene picker thumbnails/titles and model conditioning
come from that external catalog. This works with WebRTC and native-window output;
remote clients do not need the scene files.
Use `--scene-path /absolute/path/to/one-scene` for a single
scene instead. Extra archival files, including trajectories, are ignored by the
live demo. The first discovered scene is selected by default; to choose another,
pass `--sample <existing-scene-name>` after `--`. With `flashdreams-run-v2`, put
the scene path options after its `--` separator.

Smaller scene files do not remove GPU preparation: model/VAE loading, first-image
encoding, inference-cache initialization, and compilation/warmup still run.
All scenarios are prepared during server startup; prepared scenes are reused on
subsequent visits.

## Code ownership

| Package | Responsibility |
| --- | --- |
| FlashDreams V2 | Lifecycle, model/UI threads, events, presentation/backpressure, standard WebRTC and plugin discovery. |
| `flashdreams_controller_input` | Source IPC, CloudXR, Quest browser/wake management, authenticated relay, immutable controller-pair events. |
| `handdreams` | Scenes, calibration, action shaping/sampling, UI/tutorials, reset. |
| `cosmos3_nano_sim_bimanual.apps.handdreams` | Model binding, deployment preset, model CLI, scene/action encoding. |
| `cosmos3_nano_sim_bimanual` | Checkpoint loading, model pipeline, inference, encoder/decoder. |

The app does not import a model integration or V1. The controller package is
independently installable on the laptop. Input uses the existing V2
`UserInputEvent` extension point, not a second event framework. Each frame is an
immutable pair; capture and session clocks remain separate. Grip-to-wrist
conversion is applied once in the Isaac source, for both Quest WebXR and desktop
IWER. The app accepts only canonical wrist poses. Its isolated calibration service
does not load a second Isaac controller graph; it owns participant calibration
and transactional action shaping, not device-pose conversion.

For local Isaac input, `--isaac-teleop-wrist-offset-m` is a window option (before
`--`). The remote relay uses the agreed 5 cm offset. There is no app-level
`--wrist-offset-m` option; calibration and the HUD use the source's wrist poses.

## CPU checks

```bash
uv run --no-sync pytest --import-mode=importlib -m ci_cpu \
  integrations_v2/cosmos3_nano_sim_bimanual/tests \
  apps/handdreams/tests \
  apps/handdreams/controller_input/tests \
  flashdreams/test_v2
```

These tests use synthetic checkpoints and fake inference/controller services.
They do not establish GPU, WebRTC, Quest, or CloudXR hardware acceptance.
