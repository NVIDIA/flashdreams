<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlashDreams — Software Architecture and Design Document (SADD)

Following `SWE-PLC-L1-002-BasicPLC-SADD-TMPL` (Oct 5, 2023 baseline). FlashDreams-scoped subset of the broader omni-dreams + flashdreams SADD ([`../../../omni-dreams/docs/arch/sadd.md`](../../../omni-dreams/docs/arch/sadd.md)), narrowed to the `flashdreams` repo for the MVSB-32946 acknowledgment path.

## Documentation Control

| Item | Description |
|---|---|
| Title | FlashDreams SADD |
| Author(s) | Jonathan McCaffrey (jmccaffrey@nvidia.com) — PLC Security PIC for FlashDreams; AI assist (Claude Opus 4.7, 1M context) |
| Revision | 2026-05-20 (draft) |
| State | Development |
| Reviewed in | GitHub PR on `github.com/NVIDIA/flashdreams` (worktree branch `dev/jmccaffrey/arch-tava`) |
| PLC-L1 SADD Template Revision | SWE-PLC-L1-002-BasicPLC-SADD-TMPL, Oct. 5, 2023 |

### Approvers

| Date | Name | Notes |
|---|---|---|
| TBD | Security PIC (flashdreams) | TAVA acknowledgment (MVSB-32946) |
| TBD | Program Lead (Aditya Mahajan / Sanja Fidler) | Architecture sign-off |

### Reviewers

| Date | Name | Notes |
|---|---|---|
| TBD | Sean Kunde (GTC Export) | Export classification (MVSB-32940) |
| TBD | OSRB liaison | OSS exposure of architecture surfaces |

### Revision History

| Date | Author | Summary |
|---|---|---|
| 2026-05-20 | jmccaffrey | Extract FlashDreams-scoped SADD from upstream omni-dreams + flashdreams SADD; pull-up of TAVA prerequisites for MVSB-32946. |

---

## 1. Introduction

### 1.1 Purpose and Scope

This document records the architectural details for **FlashDreams** — a generative video world-model inference stack (offline recipes via `flashdreams-run` + interactive server adapters under `integrations/`) — sufficient to serve as input to the manual TAVA 2.0 process tracked in [MVSB-32946](https://jirasw.nvidia.com/browse/MVSB-32946) and to satisfy the SADD requirement for PLC-L1.

**In scope (FlashDreams TOE):**

- `flashdreams/flashdreams/` — recipes (`cosmos`, `taehv`, `template`, `wan`), core (attention / distributed / checkpoint / io), infra (pipeline / diffusion / encoder / decoder / runner), plugins (registry), scripts (`flashdreams-run` CLI).
- `integrations/alpadreams/` — gRPC server + Ludus HD-map renderer + profiling server.
- `integrations/lingbot/` — WebRTC server (single active session).
- `integrations/{cosmos_predict2, wan21, self_forcing, causal_forcing, fastvideo_causal_wan22}/` — additional recipe / framework adapters.
- The supply chain that loads these into memory: HuggingFace, S3 (`pdx.s8k.io`), and the upstream `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` base image. FlashDreams ships **no canonical pre-published container**: operators build locally from `docker/Dockerfile` (commit `ab74b58`).

**Out of scope (Operational Environment, OE):**

- `omni-dreams/samples/interactive-drive` — operator-facing consumer of FlashDreams; covered by the upstream omni-dreams + flashdreams SADD.
- `omni-dreams/post-training` — fine-tune training stack; covered upstream.
- Upstream Cosmos2 / Wan / Cosmos-Reason1 / LightVAE / TAEHV source trees (vendored under `post-training/omnidreams/_src/` in omni-dreams).

### 1.2 Assumptions

1. Operator runs the inference servers (lingbot, alpadreams gRPC) on **trusted private hardware**; bind-host defaults of `0.0.0.0` are operationally rebound to loopback for sensitive sessions (OE-1).
2. Operator HuggingFace and S3 credentials are correctly scoped (least-privilege; revocable) and not committed to git (OE-3). When the operator uses a private registry for their FlashDreams container, the registry's PAT / token is also operator-managed.
3. Multi-node deployments rely on a cluster admin to enforce tenant isolation at the IMEX / shared-FS layer (OE-2; relevant only to the omni-dreams post-training consumer).
4. No NVIDIA-side ingress, egress, or storage of operator-generated prompts, frames, or session data (architecture invariant, restated from the MVSB-32946 ticket comment).
5. FlashDreams ships **no canonical pre-published container image** (OE-7); operators build locally from `docker/Dockerfile` against `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04`. The supply-chain trust delegates upward to the upstream `nvidia/cuda` base image.

### 1.3 Constraints

1. **Platforms:** Linux (DGX Station with GB300 reference; RTX 6000 Pro Blackwell Max-Q verified for offline recipes).
2. **Distribution:** Open-source under Apache 2.0 on `github.com/NVIDIA/flashdreams` (open-model weights via `huggingface.co/nvidia/omni-dreams-*` once MVSB-33270 / 33271 close).
3. **Steady-state latency:** ~900 ms / chunk in interactive serving on a single GPU. Any added post-decode filter must fit within ≤50 ms / chunk to avoid degrading user-visible interactivity below 1 Hz.
4. **Python 3.12, torch / CUDA / NCCL** versions pinned in `uv.lock` for reproducibility.
5. **Container runtime**: operators build locally from `docker/Dockerfile` against `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` (no canonical pre-published image); native installs (`uv sync` on a host with CUDA + ffmpeg + libnccl-dev pre-installed) are also supported.

### 1.4 Dependencies

| Team / Company | Deliverable | Ref |
|---|---|---|
| Cosmos2 (NVIDIA) | Cosmos2 SV-HDMap world-model weights | § 3.2.2 HF interface |
| Cosmos-Reason1 (NVIDIA) | Cosmos-Reason1-7B text encoder | § 3.2.2 HF interface |
| LightX2V | Autoencoders (LightVAE / LightTAE) | § 3.2.2 HF interface |
| HuggingFace | Hosting for `nvidia/omni-dreams-*` and `nvidia-omni-dreams-lha/*` mirror | § 3.2.2 HF interface |
| NVIDIA SIL S3 (`pdx.s8k.io`) | Pre-release / internal checkpoint hosting | § 3.2.2 S3 interface |
| OSRB | Apache 2.0 contrib approval for flashdreams | MVSB-33271 (PR plan) |
| Product Legal | Open Model License v2.0 vs Apache 2.0 weights decision | MVSB-33270 |
| Privacy Review | Sign-off on no-PII posture; review of input-gate / output-anonymizer FSRs | MVSB-33273 |

### 1.5 Definitions, Acronyms, Abbreviations

| Term | Description |
|---|---|
| TOE | Target of Evaluation (TAVA term) — FlashDreams for this analysis |
| DFD | Data Flow Diagram |
| CP | Context Parallelism (torch.distributed group) |
| FSDP | Fully Sharded Data Parallel |
| DiT | Diffusion Transformer |
| VAE / TAE | Variational / Temporal Auto-Encoder |
| HD-map | High-definition vector map raster, used as per-frame conditioning |
| Anonymizer | Output post-processing filter that blurs faces and license plates |
| Content gate | Input safety filter that rejects CSAM / gore at session init |
| STRIDE | Spoofing / Tampering / Repudiation / Information disclosure / DoS / Elevation of privilege |
| FSR | Functional Security Requirement (TAVA v3 step 3) |
| POA&M | Plan of Action and Milestones (TAVA v3 step 4) |
| OE | Operational Environment assumption |

### 1.6 References

| SWE # | Input Work Product | Revision | Location |
|---|---|---|---|
| — | MVSB-32946 Jira ticket | 2026-05-20 | https://jirasw.nvidia.com/browse/MVSB-32946 |
| — | TAVA v3 (Oct 31, 2024) guidance | Approved by Darrell Hunt | Confluence PRODSEC |
| — | TAVA 2.0 manual page | — | https://nvidia.atlassian.net/wiki/spaces/PRODSEC/pages/2569277880 |
| — | Excel-Based Simple TAVA Template | — | NVIDIA SharePoint (Software Product Security) |
| SWE-PLC-L1-002 | BasicPLC SADD template | Oct. 5, 2023 | Glean → Google Docs (id `1EOU6o4nMUA_lGKa9_DsnByamLv3T_VvcTKMACfz153Q`) |
| — | NIST SP 800-30 Rev 1 | 2012 | NIST |
| — | Upstream omni-dreams + flashdreams TAVA | 2026-05-15 | [`../../../omni-dreams/internal/docs/planning/tava_omni_dreams_flashdreams.md`](../../../omni-dreams/internal/docs/planning/tava_omni_dreams_flashdreams.md) |
| — | This MR's architecture views | 2026-05-20 | [Static view](architecture.md#static-view), [Dynamic view](architecture.md#dynamic-view), [Data view](architecture.md#data-view), [Deployment view](architecture.md#deployment-view) |

---

## 2. Architectural Details

The FlashDreams runtime is a single Python process tree on operator-controlled hardware that loads a recipe via the unified `flashdreams-run` console script, builds a `Pipeline = DiT + encoder + VAE/TAE` from a checkpoint resolved via `FD-CKPT`, and either runs to completion (offline generation) or enters a serving loop. Two adapter packages expose FlashDreams over the network locally:

- **lingbot** — minimal WebRTC server (`GET /request_session`, `POST /api/webrtc/offer`, DataChannel actions `{keydown, keyup, step}`). Single active session per server process.
- **alpadreams** — gRPC server (InitializeSession / Step / CloseSession), with optional session-recording to disk and a separate profiling server. Protobuf stubs compiled from `integrations/alpadreams/alpadreams/grpc/protos/*.proto` via `compile_protos.sh`.

Full block diagrams in [Static view](architecture.md#static-view); runtime interactions in [Dynamic view](architecture.md#dynamic-view); dataflow with trust boundaries in [Data view](architecture.md#data-view); physical placement in [Deployment view](architecture.md#deployment-view).

### Key architectural assumptions and limitations

1. **Single-host trust collapse** — when consumed from `omni-dreams/samples/interactive-drive`'s documented `docker run`, the container uses `--network=host --ipc=host`, mounts the Wayland socket, and forwards `$SSH_AUTH_SOCK`. The container is effectively trusted-equal-to-host. Acceptable for local dev; documented as OE-1.
2. **One active session per server process** (lingbot) — simplifies STRIDE-D analysis but is a DoS pinhole if exposed beyond loopback (T-DOS-1).
3. **No in-process integrity check on downloaded checkpoints** — relies on HTTPS + HF/S3 ACLs; a poisoned cache survives container restarts (FSR_FD_04).
4. **Default network bind is `0.0.0.0`** in both lingbot (`integrations/lingbot/lingbot/webrtc/server.py:66`) and alpadreams gRPC (`integrations/alpadreams/alpadreams/grpc/server.py:1336`, `add_insecure_port`) — an audit hot-spot to be tightened (FSR_FD_10).
5. **Few-step output anonymizer is the only architecturally viable design**: a full general-purpose face/plate detector would push chunk wall-clock past the interactive budget. The architecture treats per-chunk output anonymization as **opt-in** (FSR_FD_07); the **input content gate** is **on-by-default** (FSR_FD_06).
6. **Reference implementation for both filters**: [`nvidia/Cosmos-1.0-Guardrail`](https://huggingface.co/nvidia/Cosmos-1.0-Guardrail). Pre-Guard image side serves the CSAM/gore gate; post-Guard (RetinaFace + plate + blur) is the few-step network sized for the ≤50 ms / chunk budget on a small auxiliary GPU slot.

---

## 3. Design Details

### 3.1 Design Alternatives

| Alternative | Considered | Chosen because |
|---|---|---|
| **Single-binary monolith** vs **recipe registry + plugin discovery** | both | Recipe registry lets external teams ship new model integrations without forking flashdreams. Plugin discovery is a known TAVA hot-spot (FSR_FD_09). |
| **Inference over REST** vs **WebRTC + gRPC** | both | WebRTC for interactive (low-latency video track + DataChannel for actions); gRPC for streaming inference. REST was rejected for steady-state video. |
| **Pickle checkpoints (`.pt`)** vs **safetensors** | both | Mixed: pickle for internal HF/S3, safetensors where upstream provides them. Pickle is an arbitrary-code-execution surface (STRIDE-T); captured as FSR_FD_04. |
| **Output anonymizer always-on** vs **opt-in** | both | Opt-in chosen because a full detector pass at every chunk doubles steady-state latency on a single GPU; **input content gate is always-on** because it runs once per session. |

### 3.2 Static Design

See [Static view](architecture.md#static-view) for full component / class diagrams.

#### 3.2.1 Configuration Data

| Source | Purpose | Attack-surface notes |
|---|---|---|
| `FLASHDREAMS_INTERNAL_STORAGE` env | flip checkpoint URLs to `s3://flashdreams` | env-var injection redirects fetch; defaults to off |
| `OMNI_DREAMS_HF_ORG` env var | choose `nvidia` (default) vs `nvidia-omni-dreams-lha` mirror — see `integrations/alpadreams/alpadreams/hf.py` | swap to attacker-controlled HF org is the supply-chain pivot — pin to known orgs in code (FSR_FD_08). No CLI flag today; env-only. |
| `HF_TOKEN` env | HuggingFace authentication | secret in env; leaked logs are a disclosure vector (FSR_FD_02) |
| Operator's registry token (if container is pulled rather than locally built) | container pull | secret in env or in `docker login`; operator-managed |
| `credentials/s3_checkpoint.secret` file | S3 AWS keys | secret on disk; chmod 600; never committed |
| YAML configs (per-recipe) | model + pipeline overrides | untrusted-yaml safe-load only |
| `--config_name`, `--prompt`, `--image-path`, `--synthetic-initial-rgb` | CLI flags | length / Unicode-validated (FSR_FD_14); MIME / magic-validated (FSR_FD_15) |
| Slurm `--container-image` (omni-dreams consumer) | runtime image | pinned by digest in CI; tag-only at dev |
| `FLASHDREAMS_TRUSTED_PLUGINS` env | plugin allowlist | feeds FSR_FD_09 default-deny |

#### 3.2.2 External Interface and Specification

| Name of software | Owner | Type | Operational implications | Data |
|---|---|---|---|---|
| HuggingFace Hub | HF (3rd party) | HTTPS, token-auth | Outbound only. Token can grant private-repo read; revocable. | Models (.pt / .safetensors), datasets, tokenizers. Wire format: HF resolver URLs, range-GET on LFS objects. |
| Upstream CUDA registry (`nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04`) | NVIDIA (DockerHub / NGC) | HTTPS | Outbound only; pulled at `docker build` time. FlashDreams ships no canonical pre-published image (OE-7). | OCI base image. |
| S3 (`pdx.s8k.io`) | NVIDIA SIL | HTTPS, AWS sig v4 | Outbound only; pre-signed for downloads. | Checkpoints. Wire format: bucket / key. |
| Browser viewer (lingbot WebRTC) | n/a (operator) | HTTP + WebRTC | Inbound; default bind `0.0.0.0:8089` per README today (FSR_FD_10 target: `127.0.0.1`). Single active session. | HTML/CSS/JS viewer assets; SDP offer/answer; DataChannel JSON `{action: keydown|keyup|step}`; SRTP video track. |
| gRPC clients (`replay_client`, external integrators) | NVIDIA / external | HTTP/2 + protobuf | Inbound or loopback. | Stubs from `alpadreams/grpc/protos/*.proto`: InitializeSession, Step, CloseSession. |
| `flashdreams-run` CLI | n/a (operator) | stdin / argv | Local. | text prompt; image path; recipe slug. |

#### 3.2.3 Dependencies

See § 1.4. Integration validation plan below.

##### 3.2.3.1 Integration Validation Plan

| Functionality to Validate | Teams Participating | Interfaces Covered | Date Complete |
|---|---|---|---|
| HuggingFace checkpoint fetch + load (wan, cosmos, taehv recipes) | SIL eng | HF interface, FD-CKPT | TBD |
| S3 internal-storage fallback toggled by `FLASHDREAMS_INTERNAL_STORAGE` | SIL eng | S3 interface, FD-CKPT | TBD |
| LING-RTC end-to-end browser session (loopback) | SIL eng | LING-RTC HTTP + WebRTC | TBD |
| ALPA-GRPC InitializeSession → Step → Close | SIL eng | ALPA-GRPC | TBD |
| Input content gate rejects CSAM / gore probe image | Privacy + SIL | Gate at session init | TBD |
| Output anonymizer blurs faces/plates within ≤50 ms/chunk budget | Privacy + SIL | Anon stage in FD-INFRA | TBD |
| `flashdreams-run` plugin registry default-deny | SIL eng | FD-REG | TBD |

### 3.3 Dynamic Design

See [Dynamic view](architecture.md#dynamic-view) for sequence diagrams.

#### 3.3.1 Functionality and Behavior

- **`flashdreams-run` offline**: cold-start loads checkpoint, builds a Pipeline, runs to completion, writes mp4 + JSON via rank-0 I/O gate.
- **lingbot WebRTC**: warm at server start (preload runtime + model + config), then per session: `GET /request_session` → `POST /api/webrtc/offer` → optional input gate on initial RGB → SDP negotiation → DataChannel actions → AR inference per chunk → video track + `chunk_done` event.
- **alpadreams gRPC**: stateless across InitializeSession boundaries (session ID returned); per-step streamed responses; optional disk recording behind `--record`; profiling server bound loopback by default.

#### 3.3.2 Control Flow

See [Dynamic view](architecture.md#dynamic-view) §§ 1–5.

#### 3.3.3 Data Flow

See [Data view](architecture.md#data-view) §§ 1–4.

#### 3.3.4 Error Handling

| Failure | Behavior |
|---|---|
| HF / S3 download error (4xx / 5xx) | Fail-fast at cold-start; CLI returns non-zero. README documents `401/403` path for missing token. |
| Checkpoint integrity violation (FSR_FD_04 once wired) | Fail-fast before pinning to CUDA; log the URI but not the bytes. |
| Input content gate hit | Reject the session at init; return `403` (HTTP path) or pre-RUN abort (CLI path). Do not retry. |
| WebRTC negotiation timeout | Tear down the single active session; logs to local file only. |
| gRPC client disconnect mid-Step | Server-side cleanup of `session_id`; recording flushes if enabled; per-session GPU buffers zeroed (FSR_FD_18). |
| Distributed init failure (NCCL) | All ranks abort; torchrun returns non-zero; rank 0 captures the stack to local logs. |
| GPU OOM | Fail-fast with documented `--len_t` knob to shrink chunk size. |
| Unknown plugin entry-point | Refused unless in `FLASHDREAMS_TRUSTED_PLUGINS` (FSR_FD_09). |

All logs are local; no centralized log shipping is part of the TOE. Log scrubbing per FSR_FD_02 / FSR_FD_03 is the responsibility of the core logger filter once wired.

#### 3.3.5 Logging and Debugging

- Rank-0-only logs in distributed runs (architectural invariant).
- Profiling server (`alpadreams/grpc/profiling_server.py`) exposes CPU/GPU profile traces; binds to a configurable port and **should bind loopback in production** (FSR_FD_11).
- Session recordings (`session_recorder` → `recording_io`) are opt-in (`--record`) and write `.pt` + `.json` to disk with mode `0600` (FSR_FD_21).

#### 3.3.6 State Machine

See [Dynamic view](architecture.md#dynamic-view) § 4 (checkpoint resolution) and § 5 (distributed init).

### 3.4 Security Design

An FSR (Functional Security Requirement) is a SHALL-statement bound to a specific control that traces back to one or more threats (what it mitigates), forward to one or more security objectives (what it supports), and out to a Test/Measurement (how SQA verifies it). FlashDreams uses the **Excel-Based Simple TAVA Template** column structure: `FSR ID | Functional Security Requirement | Risk Level | Risk Response | Responsibility | Priority Level | Test / Measurement`. Every FSR is prefixed `FSR_FD_NN`.

#### Canonical FSR sheet — pointer

The single source of truth is [`fsr_table.md`](fsr_table.md) — 24 canonical `FSR_FD_NN` rows plus 12 NIM-template mapping rows, in the Excel-Based Simple TAVA Template column structure. See [`tava.md`](tava.md) § 2.4.2 for the per-threat risk levels each FSR mitigates, and § 3.1 for the security objectives each FSR supports.

As of upstream station check 2026-05-15, **20 of 24 FSRs have as-built contract+stub tests passing** in `omni-dreams/internal/tests/security/` (**230 / 230** tests in 1.86 s). FSR_FD_05 is reframed as operator-side guidance because FlashDreams ships no canonical container image (commit `ab74b58`). The FlashDreams-side mirror at `flashdreams/tests/security/` is queued as a follow-up — see [`tava.md`](tava.md) § 6 Q-06.

### 3.5 Test Automation

| Category | Approach |
|---|---|
| Open box (white box) | Unit tests in `flashdreams/tests/` (CPU-safe via `pytest -m "not manual"`), `integrations/*/tests/`. Pre-commit runs ruff + ty. |
| Closed box (black box) | gRPC integration tests in `integrations/alpadreams/tests/`; lingbot WebRTC route tests in `integrations/lingbot/tests/`. |
| Security in CI/CD | Pre-commit (ruff + ty + REUSE/SPDX lint), SBOM scan (nSpect) on release candidates, container-image digest pinning. |
| Compiler / code sanitization | Python type-check via `ty`; no native compile chain beyond Triton kernels. |
| Fuzzing | **Recommended** (FSR_FD_24): target ALPA-GRPC protobuf decode + any USDZ ingest path. POA&M item. |
| Security-focused testing | Mirror the upstream `internal/tests/security/` Protocol + stub + positive/negative test pairs into `flashdreams/tests/security/`. |
| Offensive security | Out of scope for the MVSB-32946 deliverable; recommended ad-hoc review once the input gate / anonymizer land. |

### 3.6 Other Design Considerations

#### 3.6.1 Resource Limits

| Software Module | Resource | Requirement |
|---|---|---|
| FD-INFRA (Pipeline) | GPU DRAM | Fits within one GPU at `chunk_size=2`; ~40–80 GB depending on recipe + CP size |
| FD-INFRA | Wall-clock per chunk | ~900 ms steady-state on single H100 reference |
| LING-RTC | Active sessions | **1** per process (architectural invariant) |
| Output anonymizer | Wall-clock per chunk | ≤50 ms / chunk hard budget |
| Input content gate | Wall-clock per session | ≤200 ms one-shot at init |

#### 3.6.2 High Availability

FlashDreams is **not** a high-availability service. Each server process is single-tenant. Recovery is by operator restart. This is consistent with the "no NVIDIA-side ingress/egress/storage" invariant — there is no production hosting plane.

#### 3.6.3 Scalability

- **Inference**: scales sideways by spinning up additional independent server processes on distinct GPU sets (no shared session state across processes).
- **Multi-GPU**: scales up via `torchrun --nproc_per_node=N` with FSDP × CP context-parallelism; only rank 0 terminates the serving listener.

#### 3.6.4 Future Work

1. **Land FSR_FD_04** (checkpoint integrity) before the 1.1 GA Apache-2.0 flip.
2. **Default the network binds to loopback** (FSR_FD_10) in OSS-facing docs and CLI.
3. **Land the input content gate** (FSR_FD_06) — single Cosmos-1.0-Guardrail call at session init.
4. **Specify a few-step output anonymizer reference** for downstream adopters (FSR_FD_07).
5. **Document operator-side Sigstore guidance** in `docs/security/SECURITY.md` so operators who publish a FlashDreams container know to sign and verify (FSR_FD_05).
6. **Mirror upstream `internal/tests/security/` into `flashdreams/tests/security/`** so FlashDreams stands alone as the FSR-coverage source-of-truth for external operators.
