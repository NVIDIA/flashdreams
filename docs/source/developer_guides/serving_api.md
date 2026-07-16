<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# FlashDreams serving API

## Scope

`flashdreams-serve` is the central serving entry point. It discovers model variants
from the `flashdreams.serve_configs` entry-point group, selects one or more model
slugs, and starts a protocol adapter over one shared session service.

```bash
flashdreams-serve --list-models
flashdreams-serve lingbot-world-fast --protocol webrtc --host 0.0.0.0 --port 8080
flashdreams-serve lingbot-world-fast --protocol webrtc --eager-load
```

From a second shell on the same compute node, run the headless end-to-end client:

```bash
flashdreams-webrtc-client \
  --url http://127.0.0.1:8080 \
  --model lingbot-world-fast \
  --timeout 600
```

The command verifies model discovery, session creation, SDP negotiation, the ordered
control data channel, a keyboard action, the first `chunk_done` event, receipt of all
enqueued video frames, and session deletion. It prints the chunk timing metrics and
frame count as JSON. Run it on the serving node; forwarding only TCP port 8080 does
not forward WebRTC's negotiated UDP media path.

An integration registers a `ServeModelConfig`, which binds a public
`ModelDescriptor` to a lazy `ModelWorker` factory. During local development the
same config can be supplied without installing entry-point metadata:

```bash
export FLASHDREAMS_SERVE_CONFIGS='my-model=my_package.serving:SERVE_MY_MODEL'
```

## Architecture

The API separates long-lived model weights from per-client inference state.

```text
WebSocket / WebRTC / gRPC adapter
                |
          SessionService
        /        |        \
 discovery   lifecycle   ordering/leases
                |
         WorkerScheduler
                |
       WorkerLease + routing metadata
                |
      ModelWorker (weights, N session caches)
```

`SessionService` is the protocol-neutral control plane. It owns session IDs,
status, sequence numbers, idle leases, and worker leases. A lock per session
serializes its steps; unrelated sessions execute independently. Transport classes
inherit `ServingTransport`, so validation and lifecycle semantics do not drift
between WebSocket, WebRTC, and gRPC.

`LocalWorkerScheduler` lazily creates worker replicas and selects the least-loaded
compatible worker. `ModelCapabilities.sessions_per_worker` determines whether one
loaded model can host several session caches or requires an exclusive worker.
`ResourceRequest.gpu_count` and `placement` describe the indivisible resource gang
needed by a worker. A four-GPU context-parallel model therefore requests one
four-GPU worker, not four independent one-GPU sessions.

The local scheduler assumes the process launcher has already supplied the GPU
resource envelope. It enforces logical worker/session capacity but does not create
CUDA process groups. A Slurm, Kubernetes, or Dynamo scheduler implementation is
responsible for satisfying each `ResourceRequest` and returning the resulting
remote worker endpoint.

The current Lingbot adapter is intentionally conservative: its existing runtime
owns one mutable rollout cache, so it advertises one session per worker. Models
whose workers keep a `session_id -> cache` mapping can advertise a larger value
and immediately share weights across clients.

## Reference alignment

The control/data-plane split follows the
[SGLang Model Gateway](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/sgl_model_gateway.md):
one central endpoint discovers heterogeneous model workers, publishes capabilities,
and keeps transport routing separate from worker execution. FlashDreams adds an
explicit session-cache lifecycle because interactive diffusion advances mutable
state one step at a time.

The scheduler and lease boundary follows Dynamo's
[request and control planes](https://docs.nvidia.com/dynamo/design-docs/architecture-flow)
and leaves phase routing behind the worker endpoint. Dynamo's
[disaggregated serving design](https://docs.nvidia.com/dynamo/design-docs/disaggregated-serving)
motivates opaque routing and transfer metadata rather than embedding a local CUDA
address or cache representation in the public API.

## Endpoint contract

### `GET /v1/models`

Returns model slugs, accepted input/output modalities, supported transports,
sessions per worker, and placement requirements.

### `POST /v1/sessions`

```json
{
  "model": "lingbot-world-fast",
  "parameters": {},
  "lease_seconds": 300,
  "routing_hint": null
}
```

The server reserves worker capacity, initializes the session cache, and returns a
snapshot with `id`, `status`, `worker_id`, `sequence_number`, and
`lease_expires_at`.

By default workers are created lazily by the first session request. Pass
`--eager-load` to load and prewarm one worker for every selected model before the
network listener starts. Eager loading does not allocate a session cache, so the
first client uses the already-warmed worker without consuming extra capacity.

### `GET /v1/sessions/{session_id}`

Returns the current session snapshot. Clients should wait for `status=ready`
before opening a data-plane connection if a remote scheduler implements
asynchronous worker provisioning.

### `WS /v1/sessions/{session_id}/stream`

Each client message contains an exact next sequence number and a model-specific
multimodal input object:

```json
{"sequence_number": 0, "input": {"prompt": "Turn left"}}
```

The server emits one or more typed output events. A worker sets `final=true` on
the event that completes a logical step. Sequence numbers prevent retries or two
connections from accidentally advancing the same cache twice.

### `POST /v1/sessions/{session_id}/webrtc/offer`

Accepts `{ "sdp": "...", "type": "offer" }` and returns the worker's answer.
WebRTC media and data channels use the same allocated session cache as the other
transports.

### `DELETE /v1/sessions/{session_id}`

Closes media/data connections, releases the session cache, and returns worker
capacity to the scheduler.

## Dynamo extension point

A future Dynamo integration implements `WorkerScheduler` instead of changing any
transport or endpoint. `WorkerLease.routing_token` can carry a Dynamo request-plane
address and `transfer_metadata` can carry opaque KV/cache-transfer information.
The model descriptor and resource request become discovery metadata for workers.

This boundary also accommodates disaggregated execution: a scheduler may return a
logical lease backed by separate encode/prefill, diffusion/denoise, and decode
workers. The session service still sees one ordered stream, while the scheduler
and remote worker endpoint own phase routing, discovery, scaling, and state
transfer. gRPC protobuf bindings remain a deployment adapter over the same shared
lifecycle; the core class currently defines that mapping but intentionally does
not ship generated protobuf code.

## Slurm validation

Do not load models or run the test suite on a login node. Check for a reusable job
first and run commands inside it:

```bash
squeue -u gtong
/home/gtong/work/srun.sh 1 JOB_ID
cd /lustre/fsw/portfolios/healthcareeng/users/gtong/flashdreams-serving-api
uv run --project flashdreams --with pytest --with pytest-asyncio \
  pytest -m ci_cpu flashdreams/tests/test_serving_api.py
```
