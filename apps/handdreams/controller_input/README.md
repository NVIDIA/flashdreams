<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Isaac Teleop controller input for FlashDreams V2

This package lives with HandDreams but remains independently installable on the
laptop. It owns controller transport and lifecycle, not participant calibration,
action shaping, scenes, or model inference.

| Mode | Input route |
| --- | --- |
| `quest-remote-webrtc` | Unified server: direct Quest/IWER through CloudXR or laptop Isaac Teleop over authenticated TLS |
| `quest-remote-native-window` | The same unified input with a native operator window |
| `isaac-teleop-webrtc` | Lower-level local-only CloudXR/OpenXR window |

These modes delegate output to standard V2 windows and publish one
immutable `ControllerFrameEvent` per complete observation. Missing hands use
inert values with `tracked=False`; consumers must not use those poses.
No native APK, Android build tools, or APK window mode are shipped.

## Install the isolated SDK

The independently installed `flashdreams-controller-input` package provides the
`flashdreams-controller-input` setup command without loading the inference runtime.
With the `flashdreams-controller-input` command installed, run from any directory:

```bash
flashdreams-controller-input setup isaac-teleop
```

If the public command is not installed yet, run the launcher from the checkout root:

```bash
python3 integrations_v2/cosmos3_nano_sim_bimanual/apps/handdreams/launch.py setup-isaac-teleop
```

Setup displays the public package `isaacteleop[cloudxr,retargeters-lite]==1.4.142`
and the [NVIDIA CloudXR EULA](https://github.com/NVIDIA/IsaacCapture/blob/main/deps/cloudxr/CLOUDXR_LICENSE),
then requires an affirmative answer. For unattended setup, pass `--accept-eula`
only after reviewing and accepting those terms. Setup selects the Linux
x86_64/aarch64 Python 3.12 hash lock, checks dependencies and runs the existing
`doctor`, then publishes `~/.local/share/flashdreams/isaac-teleop`. It preserves
an existing mismatched or manually created environment and explains how to move
it aside. It downloads public wheels into the sidecar; FlashDreams releases
contain no Isaac Teleop or CloudXR binaries. Setup does not authorize future
CloudXR startup: each launch still requires its acceptance flag.

Setup, the checkout launcher, and direct V2/relay commands all default to this
per-user location. SDK selection is independent of the working directory.
For a custom environment, use an absolute path for both commands:

```bash
flashdreams-controller-input setup isaac-teleop --directory /absolute/path/to/isaac-teleop
```

Then add `--isaac-teleop-python /absolute/path/to/isaac-teleop/bin/python` to
launch. Existing checkout-local `.venv-isaac-teleop` environments are preserved;
there is no automatic migration or fallback to them. To keep using one, select
`--isaac-teleop-python /absolute/path/to/checkout/.venv-isaac-teleop/bin/python`.
Setup with `--directory` can revalidate a matching managed environment, but
preserves and refuses to overwrite an environment created manually.

Rerunning setup validates an existing matching environment using its own Python;
`--python` selects the bootstrap interpreter only when creating an environment.
Ctrl-C removes unfinished staging, preserves a completed environment, and prints
retry instructions. Validation and venv creation have a two-minute deadline;
wheel installation has a thirty-minute deadline. A timeout reports the failed
command and preserves any existing environment.

For troubleshooting, run the local `doctor` command in [Launch](#launch).
The [sidecar lock guide](isaac_teleop/README.md) documents the pinned dependencies.

## Launch

Use the [HandDreams one-command guide](../README.md#run-one-command-per-machine).
After the separate SDK setup above, run `flashdreams-run-v2 handdreams`
with `--mode webrtc` or `--mode native-window` and
`--isaac-teleop-accept-eula`. The app selects the unified input plugin, which
creates or reuses private controller credentials automatically. The remote
client launcher never copies the server's private key or downloads model
packages onto the laptop. A physical Quest attached to a laptop
uses `remote-vr --accept-eula`; browser-only IWER connects
directly to CloudXR and the UI's hosted-client link without a relay,
headset, or ADB. The server's `--quest-cloudxr-port` defaults to `48322`.

For a Quest USB-authorized on the model host, use the app guide's
`--quest-device SERIAL` option before the V2 command's `--`. It manages the headset with ADB while
using the server's single CloudXR/Isaac runtime. The guide covers physical wake,
certificate acceptance, and required headset permissions for both routes.
Both managed Quest routes always use USB-local, with no transport selector;
see the [USB transport guide](../README.md#quest-usb-transport).

The lower-level relay remains available for deployments managing their own
credentials:

```bash
uv run --no-project --python 3.12 \
  --with-editable './apps/handdreams/controller_input[webxr]' \
  flashdreams-quest-relay \
  --server 10.111.69.99 --port 8445 --ca-cert /path/to/server.crt \
  --cloudxr-port 48322 \
  --isaac-teleop-accept-eula \
  --accept-server-certificate
```

This lower-level command expects `FLASHDREAMS_CONTROLLER_TOKEN` in its environment.
Never put it in command arguments or logs. The companion checks the server
certificate and hostname; there is no insecure controller TLS fallback.
Physical wake is required by default, as with the launcher. Pass
`--no-require-physical-wake` to skip the gate for an already-running test;
`--require-physical-wake` remains available to enable it explicitly.
For a server-attached headset, keyboard **B** in either output window restarts
only its managed browser/USB setup and requires a fresh physical wake. The
shared CloudXR source stays alive; a relayed laptop headset is still restarted
by rerunning `remote-vr` on that laptop.

Each managed Quest startup, including a **B** retry, force-stops Quest Browser
on the selected headset before opening the teleop tab. A fresh browser process
clears accumulated Android service connections. This closes all running Quest
Browser sessions, but preserves saved browser data and site permissions.
Desktop IWER browsers and the loaded model are not restarted.

CloudXR requires Linux x86_64/aarch64, Python 3.12, glibc 2.35+, an NVIDIA
GPU/driver, and Vulkan. `webxr` supplies browser automation; `runtime` supplies
FlashDreams window support on the model host. Neither installs Isaac Teleop into
the application environment. Live modes default to
`~/.local/share/flashdreams/isaac-teleop/bin/python`; use `--isaac-teleop-python`
to select a custom isolated interpreter.

If the sidecar fails, inspect its preceding output and run from the checkout root
(use your selected interpreter instead for a custom environment):

```bash
~/.local/share/flashdreams/isaac-teleop/bin/python -E -s \
  apps/handdreams/controller_input/flashdreams_controller_input/isaac_teleop_bridge.py doctor
```

The doctor checks the wheel and controller graph without starting CloudXR or an
XR session. Rerun setup for a missing environment; an existing environment is
validated without replacement. If doctor succeeds, check GPU/Vulkan support,
CloudXR certificates, and headset network connectivity before retrying launch.

## Network and lifecycle

```text
Quest Browser --USB/ADB--> laptop CloudXR + Isaac --TLS--> DGX controller input
Desktop IWER browser --> DGX CloudXR + Isaac --------------|
Quest Browser --USB/ADB--> DGX CloudXR + Isaac -------------|
Laptop operator browser <--V2 WebRTC video----------------'
```

For managed Quest connections the page, signaling, and media travel over USB.
The attached host needs `coturn` and all three cached client assets
(`index.html`, `bundle.js`, `bundle.emulator.js`) on the attached host; it
does not download them during a session. The `launch.py remote-vr`
launcher prepares a missing local cache from the DGX cache over verified SSH
before starting this transport. Quest still needs an active Wi-Fi IP for WebRTC,
but an offline AP on a separate LAN is sufficient.
The `remote-vr` CloudXR proxy and TURN listener bind only to IPv4 loopback. The
unified server's shared proxy binds to its selected IPv4 interface plus loopback
so desktop IWER and USB Quest clients can use the same runtime. Direct
`quest-remote-*` modes require a concrete `--controller-host` address or hostname;
wildcard binds are rejected even with `TELEOP_STREAM_SERVER_IP` set.
TURN accepts clients over TCP through ADB but permits
only UDP peer relaying; TCP peer allocations are disabled. The per-run TURN
credential still permits loopback UDP peers, so the attached headset and browser
must be trusted. This is not isolation from every host-local service.
The laptop needs DGX controller TCP `8445` and WebRTC media
connectivity, not merely access to its viewer's HTTP port `8081`.
For browser-only IWER, trust the model host's CloudXR certificate at its
HTTPS URL, open the UI's Isaac Teleop link, allow local-network access,
and connect both virtual controllers with triggers released. The browser needs
CloudXR/WebRTC media connectivity directly to the model host; HTTP `8081` alone
is not enough. CloudXR always runs on the unified server; the laptop route also
runs it on the laptop. Each CloudXR launch requires explicit EULA acceptance.

The unified server admits one tracked controller source at a time. Its tracked
lease expires after one second without tracked input; source handover resets the
session and calibration. Stop the active client before switching routes. Direct
CloudXR observations do not reliably distinguish physical Quest from IWER, so
guided setup is preserved. For IWER, explicitly add `--auto-start` after the
V2 command's `--` separator to skip setup for any connected source.

The wake gate confirms an Asleep → Awake transition before connecting the
managed browser tab. It cannot prove which physical source woke the device;
do not run other wake automation simultaneously. The relay responds to the
server while waiting. Headset passcodes, privacy screens, immersive-VR Allow,
or local-network permission may still require human approval.

`--quest-accept-server-certificate` on the V2 runtime or
`--accept-server-certificate` on `remote-vr` temporarily permits certificate
errors for the CloudXR origin on the host managing the owned tab.
It defaults to disabled, is not fingerprint-pinned,
and does not install persistent browser trust. Keep it for each automated run
unless the browser already trusts that origin. It does not affect controller
TLS, other browser origins, or immersive-VR permissions.
CloudXR's OOB status/configuration API and headset registration are
unauthenticated in this demo. The `remote-vr` proxy is loopback-only; the unified
server also exposes its hub on the selected interface for desktop IWER. Use an
isolated or trusted demo network when choosing a LAN interface. The direct
`isaac-teleop-webrtc` mode uses the SDK's default listener with the OOB hub disabled
by default and likewise requires a trusted network.
Inherited `CONTROL_TOKEN` values are ignored; any `CONTROL_TOKEN` entry in a
custom CloudXR env file is rejected. No control credential is sent in the
readiness probe or browser URL. The separate controller relay still requires
a shared token and verified TLS. The unified server reuses or creates its
private identity automatically; explicit identity overrides must supply both
certificate/key paths and `FLASHDREAMS_CONTROLLER_TOKEN` together.

The guard preserves boundary/dialog settings and uses proximity broadcasts.
Clear `QUEST_META_PIN` for managed USB sessions: PIN-based settings changes can
require headset internet access and are rejected on this path. Proximity
restoration is best effort on firmware that cannot report its previous state.

Scene changes and resets retain the controller connection and do not repeat
the wake gate. Source capture timestamps determine action cadence; server
receipt timestamps determine freshness. Different machines' monotonic clocks
are not subtracted. Queues are bounded: overflow, stale input, and disconnect
break continuity rather than replay old movement.

A new producer tracking session requests a V2 reset and recalibration because
its tracking origin may have changed. Brief tracking loss within the ownership
lease preserves calibration; acquiring input after the lease expires resets
the rollout, even when the same source returns. Ctrl-C, server disconnect, or
browser failure cleans up the owned tab, ADB forward, CloudXR services, and
managed headset settings, then explicitly sends the managed headset to sleep.
The sleep request is bounded and best effort if USB has disconnected; it never
toggles the power button or targets a headset owned by another launcher. Rerun the
`remote-vr` command after a disconnect; the DGX can retain its model.

## CPU checks

```bash
uv run --package flashdreams-controller-input --extra runtime --extra webxr --extra dev \
  pytest --import-mode=importlib -m ci_cpu \
  apps/handdreams/controller_input/tests
```

Tests use fake devices and loopback TLS. They cover clock/pose contracts,
bounded queues, reconnect/reset, physical wake, cleanup, and packaged modes;
they do not establish Quest/CloudXR hardware acceptance.

Installer tests also build a small first-party fixture wheel locally and exercise
real venv creation, pip hash rejection, doctor failure, publication, and reuse
without network access or SDK installation. These cases require Linux CPython
3.12 with `ensurepip`; set `FLASHDREAMS_SETUP_TEST_PYTHON` to a suitable interpreter
if the test runner uses another Python. They skip when that prerequisite is absent.
