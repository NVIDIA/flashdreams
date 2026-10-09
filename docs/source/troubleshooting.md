---
title: 'Troubleshooting'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->

<!-- SPDX-License-Identifier: Apache-2.0 -->

Use this page for common first-run failures before opening an issue. Each
entry lists the visible symptom, the most likely cause, and the next concrete
step to try.

Current model applications implement `flashdreams.api_v2` and run with
`flashdreams-run-v2`. Runtime flags precede `--`; application flags follow it.
Do not move flags across that separator. See the
[CLI reference](documentation/cli.md) for the command shape.

## Launching a Demo Takes a Long Time

**Symptoms:**

- A demo takes a long time to launch.

**Likely cause:**

The first run downloads model assets and spends several minutes compiling (and potentially autotuning) kernels.

**Fix or next step:**

Let the first launch finish if it is still making progress. Subsequent launches
reuse compiled kernels and autotuning results, although CUDA graphs are
captured again for each process.

## Triton autotuning or warmup looks stuck

**Symptoms:**

- The first launch takes several minutes.
- Logs mention Triton autotuning or CUDA-graph warmup.
- Later runs are much faster than the first run.

**Likely cause:**

- Cold runs include one-time setup. The quickstart and model pages document that first launches can include downloads, Triton autotuning, CUDA-graph warmup, native-code compilation, etc...

**Fix or next step:**

Let the first launch finish if it is still making progress. Subsequent launches
reuse compiled kernels and autotuning results, although CUDA graphs are
captured again for each process.

Lock file for compilation (`pytorch`/`triton`/...) may be tracking a hanging-process. if this is the case, delete the offending lock file and rerun the command.

## CUDA or PyTorch build mismatch

**Symptoms:**

- A CUDA extension fails to build or load.
- ``flashdreams-run-v2 interactive-drive-omnidreams-perf --mode
  native-window`` exits instead of falling back to the default PyTorch path.
- Errors mention `nvcc`, a CUDA version, a GPU architecture, or missing CUDA
  libraries.

**Likely cause:**

The OmniDreams perf application config uses native DiT acceleration with
`native_dit_acceleration="required"`. That path requires a source checkout,
`git`, network access for pinned third-party sources, and a CUDA toolchain
with `nvcc` matching the installed PyTorch build. The fast perf variant also
uses native FP8 LightVAE. The native kernels are tuned for Blackwell GPUs;
use a model-page configuration documented for the installed GPU.

**Fix or next step:**

Start with the non-perf OmniDreams launch in [/models/omnidreams](models/omnidreams.md). The
perf application downloads its pinned third-party sources on first use. Verify
that the machine has network access, the required GPU, and a CUDA toolchain
that matches the PyTorch build before launching:

```bash

python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvcc --version

```

## Disk or cache exhaustion

**Symptoms:**

- A first run fails while downloading or loading checkpoints.
- The process reports no space left on device, stops mid-load, or leaves a
  partial Hugging Face cache.
- Output video or stats files are missing after an interrupted run.

**Likely cause:**

Model checkpoints and example assets are cached on first use. LingBot-World
downloads a checkpoint of about 70 GB through the Hugging Face cache and its
docs recommend keeping about 200 GB free for the model plus cache. Its example
data is stored in the FlashDreams user cache under
`example_data/lingbot_world/<NN>/`. Generated MP4s consume space wherever
`--output-path` points.

**Fix or next step:**

Check free space on both the repository filesystem and the Hugging Face cache
filesystem. If the default cache location is too small, point `HF_HOME` at a
larger volume before running the model:

```bash

export HF_HOME=/path/to/large/cache

```

Then rerun the same command so the downloader can reuse or repair the cache.

## Model download or authentication failure

**Symptoms:**

- A download returns 401, 403, not found, or gated-repository errors.
- OmniDreams scene or checkpoint downloads fail before the demo starts.
- LingBot-World fails while fetching the model checkpoint from Hugging Face.

**Likely cause:**

Most model runs need Hugging Face authentication. OmniDreams requires an
`HF_TOKEN` with read access to the `nvidia/omni-dreams-scenes` dataset and
the `nvidia/omni-dreams-models` model repository. Other model pages also
document first-run downloads from Hugging Face.

**Fix or next step:**

Export a valid token in the same shell that launches FlashDreams:

```bash

export HF_TOKEN=<your-hf-token>

```

If the token is already set, confirm that the account behind it can open the
model or dataset page referenced by the failing command, then rerun the
FlashDreams command.

## GPU out of memory

**Symptoms:**

- The run exits with `CUDA out of memory` or the Python process is killed
  during model load or generation.
- LingBot-World runs out of memory on a single GPU when using large
  `--total-blocks` values.
- Multi-GPU commands fail after one or more ranks report memory pressure.

**Likely cause:**

The selected model, resolution, rollout length, or GPU count does not fit the
available VRAM. The model pages list minimum VRAM expectations: OmniDreams is
about 48 GB, Self-Forcing is about 24 GB, and LingBot-World is about 120 GB.

**Fix or next step:**

Use a smaller documented run first: reduce `--total-blocks`, lower
`--pixel-height` and `--pixel-width` where the model supports those runtime
overrides, or use the model page's documented multi-GPU
`torchrun --nproc_per_node=<N>` launch. With `flashdreams-run-v2`, runtime
flags go before `--` and application flags such as `--total-blocks` go
after it. For LingBot-World, use the efficient streaming application slug
`cam2v-lingbot-world-fast-taehv-window15-sink3`.

<a id="webrtc-troubleshooting"></a>

## WebRTC connection or video does not appear

**Symptoms:**

- The URL printed by a WebRTC application is not reachable from the browser.
- The page loads but video never appears.
- The server seems idle before printing the connection URL.

**Likely cause:**

The WebRTC servers open their HTTP port only after model load and warmup. On a
remote or cloud GPU instance, the server port may not be reachable directly at
the host IP.

Reaching `http://<server>:8089/` proves that the HTTP page and signaling path
are reachable, but it does not prove that the WebRTC media path is reachable.
After signaling, WebRTC still needs a working media path
negotiated by ICE. Video uses SRTP, the control data channel uses SCTP over
DTLS, and both typically depend on UDP connectivity. Plain SSH `-L`
forwarding, or any other TCP-only forward of the HTTP port, does not forward
that WebRTC media path.

If the page loads but video does not appear, ICE may be unable to
find a working media path, or the browser may be hiding local IPs in ICE
candidates with mDNS hostnames.

**Fix or next step:**

Wait until the v2 server prints `Open http://<server-ip>:8089/ in a browser.`
For remote machines, forward the documented HTTP port and open the local URL:

```bash

ssh -L 8089:localhost:8089 <user>@<host>

```

Then open `http://localhost:8089/`.

If the page loads but the video still does not appear, use a setup that carries
both the HTTP path and the WebRTC media path:

- Direct browser access to the server network, with the firewall or security
  group allowing the WebRTC media traffic advertised during ICE negotiation in
  addition to the HTTP port.
- A VPN or UDP-capable forwarding solution between the browser and the server.
- A TURN relay that you provide and configure when direct peer-to-peer
  connectivity is blocked. STUN can help ICE discover reachable addresses, but
  STUN is not a relay. Do not assume a WebRTC demo deploys TURN unless its
  model page or launch command documents it.

If the media path is available and the connection still fails, check the
browser's ICE policy. Some browsers can hide local IP addresses behind mDNS
`.local` hostnames. Disable that behavior and reload:

- **Chrome / Edge:**
  `chrome://flags/#enable-webrtc-hide-local-ips-with-mdns` set to
  **Disabled**, then restart the browser.
- **Brave:** `brave://settings/privacy/security`, set *WebRTC IP handling
  policy* to **Default public and private interfaces**.
- **Firefox:** `about:config`, set
  `media.peerconnection.ice.obfuscate_host_addresses` to **false**.
