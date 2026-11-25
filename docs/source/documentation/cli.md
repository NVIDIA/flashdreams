---
title: 'CLI Reference'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

## Command Shape To Run A Demo/Model

This section will only cover `<System Arguments>`. Refer to the [Running a model](../models/index.md#running-a-model) section for more information on `<Demo Preset>` and `<Demo Arguments>`:
```bash
uv run flashdreams-run-v2 <Demo Preset> <System Arguments> -- <Demo Arguments>
```

## System Arguments

### General

| Argument | Description |
| --- | --- |
| `--timeout {SECONDS,unbound}` | Bound the whole application run, including initialization and replacement sessions. |
| `--total-model-steps {N,unbound}` | Bound model steps across the whole application run, including replacement sessions. |
| `--mode {mp4,webm,webrtc,native-window}` | Select file, browser, or local-window presentation. Default: `mp4`. |
| `--stats-path PATH` | Write JSON model-step measurements and enable synchronized per-stage pipeline profiling. |

### MP4 and WebM

| Argument | Description |
| --- | --- |
| `--output-path PATH` | Video destination. Required in MP4 and WebM modes. |

### WebRTC Specific

| Argument | Description |
| --- | --- |
| `--host HOST` | Interface to serve on. Default: `127.0.0.1`. Nonlocal access requires the origin configuration below. |
| `--port PORT` | Port to serve on. Default: `0`, which selects an available port. |
| `--no-login` | Skip browser pairing on a trusted single-user machine (loopback only). Any local process can then take over the viewer. |

### Native Window Specific

| Argument | Description |
| --- | --- |
| `--window-title TITLE` | Native window title. Default: `FlashDreams`. |

### Default Demo Session Overrides

These options override the default demo session values requested by a particular demo implementation. Omit them to use the application's default values.

| Argument | Description |
| --- | --- |
| `--pixel-width N` | Override generated-frame width. |
| `--pixel-height N` | Override generated-frame height. |
| `--fps N` | Override the generated-frame playback rate. |
| `--layout {tchw,btchw,bcthw,bvtchw}` | Override generated tensor layout. |
| `--backpressure-mode {block,drop_oldest}` | Choose full-queue behavior: `block` waits for capacity to retain generated chunks; `drop_oldest` discards the oldest queued chunk to favor recent output. |
| `--presentation-mode {on_demand,continuous}` | Choose rendering behavior: `on_demand` presents each selected model frame when it arrives; `continuous` keeps the UI responsive between model frames by reusing the newest frame. |

### Examples

Stream to a browser:

```bash
uv run flashdreams-run-v2 DEMO_MODEL_SLUG --mode webrtc --port 8089
```

Open the link it prints. For a remote server, see
[browser access and pairing](#browser-access-and-pairing).

Write every generated frame in order to a WebM file:

```bash
uv run flashdreams-run-v2 DEMO_MODEL_SLUG --mode webm --output-path output.webm \
  --timeout unbound --backpressure-mode block --presentation-mode on_demand
```

MP4 and WebM modes require `--timeout` and/or `--total-model-steps`; pass `unbound` to
name an explicit unlimited policy. WebRTC and native-window modes may omit both
limits. When a session is replaced, it receives the time and model steps left
for the whole run.

Preload and validate a v2 application without opening a window:

```bash
uv run --no-sync flashdreams-run-v2 cam2v-lingbot \
  --preload-application -- --example-data
```

Preload runs `IApplication.init` and one model block. It defaults to warning;
set `FLASHDREAMS_PREPARATION_POLICY` to `none`, `warn`, or `error`. Findings
are written to `preparation_issues.txt` in the FlashDreams cache, or to
`FLASHDREAMS_PREPARATION_ISSUES_PATH` when set. Use
`--skip-preload-validation` to skip the first-block validation.

Open a local native window:

```bash
uv run flashdreams-run-v2 DEMO_MODEL_SLUG --mode native-window --window-title FlashDreams
```

## Browser access and pairing

V2 serves one browser owner at a time, and every launch pairs that browser by
default. The serving process creates a private pairing code once, in
`~/.config/flashdreams/webrtc-token` (`$XDG_CONFIG_HOME` is honored), and reuses
it on later launches. At startup it prints how to connect:

- In an interactive terminal, a one-click link such as
  `http://127.0.0.1:8089/#pair=CODE`. The code travels in the URL fragment,
  which browsers never send to a server; the page keeps it for this tab and
  removes it from the address bar.
- In logs and batch output (Slurm, `tee`, redirected output), only the URL and
  the file to paste the code from. The code is never written to a log.

The tab remembers pairing in its session storage, so refreshes and server
restarts need no new prompt; a new tab asks once. **Forget pairing** clears the
tab's copy and disconnects. Anyone with the code can replace the active
browser, so keep it out of command arguments, shared terminals, and logs. An
unpaired client is refused and cannot replace a paired browser, even from the
same machine. To rotate the code, delete the file and restart; paste the new
code when prompted.

Host and browser Origin must match the local service, so other websites,
cross-origin API requests, and hostnames that merely resolve to loopback are
refused.

On a trusted single-user machine, `--no-login` restores tokenless access at
`http://127.0.0.1:PORT`, `http://localhost:PORT`, or `http://[::1]:PORT`. It
authenticates by peer address only: every local account and process that can
reach the port gets full control, and so does anything that forwards the port
from loopback (SSH `-L` with `GatewayPorts`, `socat`, `kubectl port-forward`,
or a reverse proxy that adds no forwarding headers). Rejecting requests with
`Forwarded`/`X-Forwarded-*` headers is best-effort. The server logs a notice
in this mode, and refuses `--no-login` with a nonloopback `--host` or
configured origins.

To keep the code somewhere else, point `FLASHDREAMS_WEBRTC_TOKEN_FILE` at a
regular, nonsymlink file owned by the server user with mode `600`, containing
32–256 URL-safe characters. On Windows, keep it in your user profile, whose
permissions already restrict it to you. It replaces the default file. Only the
serving process reads it; other ranks of a distributed run never do.

For a remote server, an SSH tunnel needs no certificates or new service. Keep
`--host 127.0.0.1 --port 8089` and forward that port with the same local port,
for example `ssh -L 8089:127.0.0.1:8089 SERVER`; the server accepts only its
own Host, so a different local port is refused. Click the link printed in your
SSH terminal, or open `http://127.0.0.1:8089` and paste the code from the
server's file.

For an existing TLS reverse proxy, configure the exact external browser origin
and preserve its Host header when forwarding to the server:

```bash
export FLASHDREAMS_WEBRTC_ALLOWED_ORIGINS='https://viewer.example.com'
uv run flashdreams-run-v2 DEMO_MODEL_SLUG --mode webrtc --host 127.0.0.1 --port 8089
```

Origins are comma-separated, exact `https://host[:port]` values without paths,
wildcards, or trailing slashes. HTTP is accepted only for loopback origins,
including the SSH tunnel workflow. The browser refuses credential-bearing API
requests to another origin or an unencrypted nonloopback page.
Nonloopback listeners require explicit origins. Approved Host and Origin
checks also apply to static pages; an authenticated proxy must preserve the
owner bearer. `Forwarded`, `X-Forwarded-*`, and `X-Real-IP` never grant
identity or rewrite the selected policy. Do not publish a `--no-login`
listener through a proxy or port forwarder. Use TLS or an SSH tunnel for
remote signaling and uploads: the built-in HTTP listener does not encrypt
them. Restrict direct access to a proxy backend with host/network rules.

Offers, readiness, and uploads use the same owner authorization. An
unauthorized or invalid offer leaves the current peer connected. A valid
same-owner offer negotiates first, then retires the old peer, its frame queue,
pending picks, and buffered browser input. Retired peer callbacks cannot act
on the new peer. Disconnection retires the connection without revoking the
owner code, permitting ordinary refresh.

Editing or deleting the file does not change a running server; restart it.
Embedded applications can call `window.server.rotate_auth_token(new_token)`
from the application thread to revoke the old code and install a replacement,
or pass `None` to disable admission. This invalidates offers already
negotiating and disconnects the active peer; revocation never restores
no-login access. Viewer callbacks report actual active-peer connectivity and
invalidate consumers on retirement.

Migration: launches that relied on tokenless loopback now pair once per tab, or
pass `--no-login`. Older `--host 0.0.0.0` examples need allowed origins, or can
use the loopback SSH workflow. Applications constructing
`WebRTCClientWindow`/`WebRTCServer` directly keep tokenless loopback unless
they pass `auth_token=` (for example `load_pairing_token()` from
`flashdreams.runtime_v2.serving.security`) and, for remote origins,
`allowed_origins=`. Do not put the token in configuration objects that are
printed or serialized.

## See also

- [Quickstart](../quickstart/index.md)
- [Demo API](demo_api/index.md)
- [Offline Program Packager](tools/offline_program_packager.md)
