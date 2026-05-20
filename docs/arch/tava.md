<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# TAVA — FlashDreams (MVSB-32946 subset)

**Document Type:** Threat and Vulnerability Analysis (Manual TAVA 2.0)
**Process:** TAVA v3 (Oct 31, 2024) Confluence guidance — Assets → Threats → FSRs → POA&M
**Based On:** [`architecture.md`](architecture.md) @ HEAD; flashdreams v0.3
**Author:** Jonathan McCaffrey (TAVA PIC) with AI assist (Claude Opus 4.7, 1M context)
**Status:** Draft 2026-05-20; pending Security PIC review under MVSB-32946

> **Relationship to the upstream TAVA**: this is the FlashDreams-scoped subset of [`../../../omni-dreams/internal/docs/planning/tava_omni_dreams_flashdreams.md`](../../../omni-dreams/internal/docs/planning/tava_omni_dreams_flashdreams.md), narrowed to the **flashdreams TOE** (recipes + core + infra + integrations; excluding omni-dreams interactive-drive and post-training). The upstream document remains the source of truth where the two TOEs overlap. Coverage numbers cited as "as-built on station 2026-05-15" reference the upstream `omni-dreams/internal/tests/security/` package; the FlashDreams test mirror is planned for a follow-up MR.

---

## Executive Summary

FlashDreams is a generative video world-model inference stack (offline recipes + interactive WebRTC / gRPC adapters). Distribution will be **public open-source via GitHub + HuggingFace** under Apache 2.0 (code) plus an open-model license (weights) once MVSB-33270 / MVSB-33271 close. That distribution flip — Eval-license private to Apache 2.0 public — is the single largest reason this TAVA exists.

The core threat surfaces are:

- **Generative-AI safety surface** — operator-supplied initial frames can drive the model to amplify harmful content (CSAM / gore / non-consensual real-person likenesses) downstream into video output (T-CONT-1 / T-PRIV-1).
- **Server endpoints exposed by default on `0.0.0.0`** — lingbot WebRTC (`:8089`) and (configurably) the alpadreams gRPC + profiling server — without authentication or encryption in the documented developer flow (T-NET-1 / T-AUTH-1).
- **Supply-chain integrity** of model weights (HF `nvidia/omni-dreams-*` and `nvidia/Cosmos-1.0-Guardrail`, S3 `s3://flashdreams`) — `torch.load` on a poisoned `.pt` is arbitrary-code execution (T-CKPT-1). FlashDreams ships **no canonical pre-published container image** (OE-7, commit `ab74b58`); operators build locally from `docker/Dockerfile` against `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04`, so the container-image threat shape collapses to "upstream `nvidia/cuda` base image" + "operator-side build pipeline".
- **Plugin discovery** via Python entry-points (`flashdreams-run`) — third-party packages installed in the operator's venv get imported and executed at module-import time (T-PLUG-1).

**Overall risk rating: HIGH** — driven by (1) AI-safety surface on operator-supplied media, (2) public OSS distribution amplifying supply-chain blast radius, and (3) default-on `0.0.0.0` exposure without auth or TLS. The risk is **tractable** because every H-or-higher threat has a clear architectural seam already prototyped upstream and ready to mirror into `flashdreams/tests/security/`.

The single highest-leverage residual ask before flashdreams 1.0 GA is to land **Cosmos-1.0-Guardrail pre-Guard** (FSR_FD_06) as a default-on hard gate at session init — one model invocation per session that drops T-CONT-1 from Critical to residual Low.

---

## Step 0 — Introduction

| Field | Value |
|---|---|
| NSPECT ID | NSPECT-6O5R-39LY (flashdreams) |
| Program | FlashDreams (GA + OSS) — v0.3 |
| Jira | [MVSB-32946](https://jirasw.nvidia.com/browse/MVSB-32946); PLC L1; manual TAVA 2.0 path (gated by MVSB-32940 export classification) |
| TAVA PIC | Jonathan McCaffrey (Security PIC TBD for ack) |
| Inputs | [Static](architecture.md#static-view) · [Dynamic](architecture.md#dynamic-view) · [Data](architecture.md#data-view) · [Deployment](architecture.md#deployment-view) · [`sadd.md`](sadd.md) |
| Methodology | NIST SP 800-30r1 qualitative risk; STRIDE for threat ID |

---

## Step 1 — Security Assets (TOE, Architecture, DFD, Asset Inventory)

### 1.1 Target of Evaluation (TOE)

The TOE is the **runtime process tree** that executes FlashDreams inference, including:

**In scope:**

- `flashdreams/flashdreams/` — recipe + core + infra packages and the `flashdreams-run` CLI
- `flashdreams/integrations/alpadreams/` — gRPC server + Ludus HD-map renderer + profiling server
- `flashdreams/integrations/lingbot/` — WebRTC server (single active session)
- `flashdreams/integrations/{cosmos_predict2,wan21,self_forcing,causal_forcing,fastvideo_causal_wan22}/`
- The supply chain that loads these into memory: HuggingFace, S3 (`pdx.s8k.io`), and the upstream `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` base image. FlashDreams ships **no canonical pre-published container image** (commit `ab74b58`); operators build locally from `docker/Dockerfile`.

**Out of scope (Operational Environment, OE):**

- The host Linux kernel + NVIDIA driver
- Operator's browser
- `omni-dreams/samples/interactive-drive` — consumer of FlashDreams; covered upstream
- `omni-dreams/post-training` — fine-tune training stack; covered upstream
- Slurm cluster admin posture (relevant only to the omni-dreams post-training consumer)
- Upstream Cosmos2 / Wan / Cosmos-Reason1 / LightVAE source trees (vendored under `post-training/omnidreams/_src/` in omni-dreams)

### 1.2 TOE Description

See [Static view](architecture.md#static-view) for the full block diagram. Summary:

1. The operator launches a flashdreams-run-derived process (CLI, gRPC server, or WebRTC server). That process loads a checkpoint from HF or S3, builds a DiT + VAE/TAE pipeline, and either runs to completion (offline generation) or enters a server loop.
2. Inputs arriving at runtime: keyboard / WebRTC DataChannel actions, optional initial RGB image, HD-map raster from Ludus, text prompt, scene file.
3. Outputs: RGB frames (served via WebRTC video track or streamed as gRPC frames), optional disk recordings, logs.
4. No NVIDIA-side ingress, egress, or storage of operator data is in scope (architecture invariant from MVSB-32946 comment).

### 1.3 Security Assumptions (OE)

| ID | Assumption |
|---|---|
| OE-1 | Operator runs the inference servers on **trusted private hardware** with the documented `--network=host --ipc=host` Docker invocation (when consumed via `omni-dreams/samples/interactive-drive`). Servers bound to non-loopback addresses are only exposed on trusted LANs. |
| OE-2 | Out-of-scope for FlashDreams TOE; relevant for the omni-dreams post-training consumer (Slurm cluster admin enforces tenant isolation). |
| OE-3 | Operator HuggingFace and S3 credentials are correctly scoped (least-privilege; revocable) and not committed to git. When the operator uses a private registry for their FlashDreams container, the registry's PAT / token is also operator-managed. |
| OE-4 | The host Linux kernel + NVIDIA driver are patched per NVIDIA's PLC standard. |
| OE-5 | Cosmos-Guardrail weights are sourced from `huggingface.co/nvidia/Cosmos-1.0-Guardrail` or an NVIDIA-published mirror; the Guardrail itself is treated as a trusted, NVIDIA-evaluated artifact. |
| OE-6 | Operator network policy may block outbound `huggingface.co` egress. In that environment, Cosmos-Guardrail weights MUST be pre-staged on local disk by the operator. Stations without egress and without pre-staged weights SHALL refuse to start a session rather than silently bypass the gate. |
| OE-7 | FlashDreams ships **no canonical pre-published container image** (per commit `ab74b58`). Operators build locally from `docker/Dockerfile` against the upstream `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` base image, or `uv sync` natively against a host CUDA stack. Any operator-built image lives under the operator's own registry / access controls. The container-image supply-chain trust delegates upward to the upstream `nvidia/cuda` base image; T-IMG-1 narrows from "external supply-chain compromise of an NVIDIA-published FlashDreams image" to "tampered `nvidia/cuda` base or tampered operator build pipeline". FSR_FD_05 becomes **operator guidance** (Sigstore-sign if you publish; `cosign verify` if you pull) rather than a FlashDreams-side enforcement. |

### 1.4 Architecture Diagram

Submitted alongside this TAVA: [Static view](architecture.md#static-view), [Dynamic view](architecture.md#dynamic-view), [Deployment view](architecture.md#deployment-view).

### 1.5 Dataflow Diagram

[Data view](architecture.md#data-view) §§ 1–3 with explicit trust boundaries.

### 1.6 Security Asset Inventory

| Asset ID | Asset | Where it lives | C/I/A | Sensitivity |
|---|---|---|---|---|
| A-CKPT | Model weights (cosmos, wan, taehv, plus recipe checkpoints fetched by adapters: alpadreams, lingbot_world, predict2 / predict2_multiview) | HF, S3, GPU DRAM at runtime | C+I | NVIDIA-proprietary pre-GA; Apache 2.0 / OML at GA |
| A-CFG | YAML recipe configs and `--config_name` registry | Repo + GPU DRAM | I | Public on GA |
| A-PROMPT | Operator text prompts | RAM, possibly disk logs | C | Operator-confidential |
| A-INPUT-RGB | Operator-supplied initial RGB / scene first frame | RAM, scene cache | C (if PII) | Possibly PII |
| A-FRAME | Generated video frames | RAM, WebRTC track, gRPC stream, optional disk | C+I | Operator-confidential; possibly contains identifiable faces/plates |
| A-HDMAP | HD-map raster (Ludus output / cached PNG) | RAM, disk | I | Public synthetic |
| A-SESSION | Active session state (WebRTC PeerConnection, gRPC session_id) | Server RAM | C+I+A | Operator-confidential |
| A-RECORDING | Optional session recording (`.pt` + `.json`) | Disk | C+I | Operator-confidential |
| A-SECRET | HF_TOKEN, GITHUB_PAT, `credentials/s3_checkpoint.secret` | Env, disk | C+I | Secret |
| A-IMG | Operator-built FlashDreams container image (from `docker/Dockerfile` on top of upstream `nvidia/cuda:13.2.1`) | Disk, operator's registry | I | Operator-owned (FlashDreams ships no canonical image — see OE-7) |
| A-LOGS | Logs, profiling traces | Disk | C | Operator-confidential |
| A-PLUGIN | Plugin registry / entry-point loaded code | Process | I | Code execution surface |
| A-RUNTIME | Python process / GPU context / NCCL group | RAM, GPU | I+A | Process integrity |

---

## Step 2 — Security Threats

### 2.1 Attacker Modeling

| Actor | Intent | Capability | Access | Notes |
|---|---|---|---|---|
| **Curious LAN-mate** | View someone else's interactive session, exfiltrate generated frames | Network-only; default browser tooling | LAN broadcast / WiFi co-tenant | Triggered by `0.0.0.0` default binds (LING-RTC `:8089`; ALPA-GRPC configurable) |
| **Supply-chain attacker** | Place poisoned weights or container bytes on the path between NVIDIA / upstream and operator | Cloud-side compromise of HF mirror; or compromise of upstream `nvidia/cuda` base image; or operator-side build-pipeline compromise (OE-7) | Indirect; requires HF mirror / upstream-CUDA compromise OR operator-side CI access | Highest blast radius is the HF weights path; container-image path narrowed to "operator-owned build" after OE-7 |
| **Malicious recipe / plugin author** | Register a plugin under the flashdreams entry-point that runs code on `flashdreams-run` | Local install via `uv pip install` | Operator's venv | T-PLUG-1 |
| **Operator under coercion / mistake** | Process attacker-supplied input image to elicit harmful generations (CSAM, gore, deepfakes of real people) | Submit a crafted initial RGB | Local CLI / WebRTC | T-CONT-1, T-PRIV-1 |
| **Insider with repo write** | Backdoor in pre-GA code | Git push | git remote | Mitigated by code review + signed commits; out of TAVA scope but acknowledged |
| **State-level actor** | Steal pre-release weights or training data | Sophisticated supply chain + endpoint compromise | Any | Treated as ceiling; mitigations are NVIDIA-wide controls, not TOE-specific |

### 2.2 Trust Boundaries

| ID | Boundary | Crossing point(s) |
|---|---|---|
| TB-1 | Operator workstation ↔ external networks | HF / S3 / upstream `nvidia/cuda` outbound; LING-RTC + ALPA-GRPC inbound |
| TB-2 | Operator browser / gRPC client ↔ inference server (loopback or LAN) | HTTP signaling, WebRTC media, gRPC streaming |
| TB-3 | Container ↔ host (`--network=host` + `--ipc=host`) | Effective trust collapse documented as OE-1 |
| TB-4 | Pipeline ↔ disk (recordings, logs) | session_recorder I/O |
| TB-5 | Pre-Guard / post-Guard ↔ inference pipeline | Cosmos-Guardrail integration seam |

### 2.3 Threat Inventory (STRIDE)

| Threat ID | STRIDE | Asset(s) | Attack path |
|---|---|---|---|
| T-NET-1 | S, T, I, D | A-SESSION, A-FRAME | LAN-mate connects to lingbot `:8089` while LING-RTC bound `0.0.0.0` and hijacks the active WebRTC session |
| T-NET-2 | I | A-LOGS | LAN-mate reaches the profiling server (alpadreams) |
| T-AUTH-1 | S | A-SESSION | gRPC client impersonates legitimate operator (no auth interceptor) |
| T-CKPT-1 | T, E | A-CKPT, A-RUNTIME | Poisoned `.pt` pickle on HF/S3 path executes arbitrary code at `torch.load` |
| T-CKPT-2 | T | A-CKPT | Substituted weights generate biased / dangerous output without RCE |
| T-CFG-1 | T | A-CFG | `OMNI_DREAMS_HF_ORG` redirected to attacker-controlled HF org pulls poisoned weights |
| T-PLUG-1 | E | A-PLUGIN, A-RUNTIME | Malicious entry-point-registered plugin gets `import`-ed by `flashdreams-run` |
| T-CONT-1 | I | A-FRAME, reputation | Operator (or attacker via WebRTC) supplies a CSAM / gore initial frame; the world model amplifies into a generated video |
| T-PRIV-1 | I | A-FRAME | Model emits a recognizable face or license plate (training-data echo) |
| T-INPUT-1 | T, D | A-SESSION | Malformed JSON / oversized payload on WebRTC DataChannel crashes or wedges the server |
| T-RECORD-1 | I | A-RECORDING, A-PROMPT | Disk-resident `.pt` recordings readable by other local users |
| T-SECRET-1 | I | A-SECRET | HF_TOKEN / S3 key leak into logs or stderr |
| T-IMG-1 | T | A-IMG | Tampered upstream `nvidia/cuda` base image, or operator-side build-pipeline compromise (narrowed by OE-7 — FlashDreams ships no canonical pre-published image) |
| T-MOUNT-1 | E | A-RUNTIME | Container compromise pivots to host via Wayland socket, SSH agent forward, shared IPC (architecturally accepted under OE-1 when consumed via interactive-drive) |
| T-DOS-1 | D | A-SESSION | LAN-mate floods lingbot signaling; single active session forces eviction of legitimate user |
| T-FUZZ-1 | T, E | A-RUNTIME | Malformed protobuf at ALPA-GRPC; malformed media at SceneLoader (consumer-side) |

### 2.4 Risk Analysis (NIST SP 800-30 Rev 1, qualitative)

Likelihood and Impact each scored on 5-level scale (VL, L, M, H, VH). Overall Likelihood combines *threat-causes-adverse-impact* with *threat-occurs* (default: take the lower of the two unless one is VL). Risk = Overall Likelihood × Impact. Acceptance criteria: **H or VH risk must have a P1 FSR** with a near-term POA&M; **M risk** must have a P2 FSR or documented residual; **L/VL** may be accepted with rationale.

#### 2.4.1 Risk matrix

| Likelihood ↓ \ Impact → | VL | L | M | H | VH |
|---|---|---|---|---|---|
| **VH** | L | M | H | VH | VH |
| **H** | L | M | M | H | VH |
| **M** | VL | L | M | H | H |
| **L** | VL | L | L | M | M |
| **VL** | VL | VL | L | L | M |

#### 2.4.2 Per-threat risk table

| Threat | Likelihood (occur × cause-adverse) | Impact | Risk | Rationale |
|---|---|---|---|---|
| T-NET-1 (LAN-mate session hijack) | M × M = M | H (operator-confidential content disclosed) | **H** | `0.0.0.0` is the documented default; impact gated by whether real prompts contain sensitive content |
| T-NET-2 (profiling server reach) | L × M = L | M | **L** | Off by default; FSR_FD_11 sets loopback-default |
| T-AUTH-1 (gRPC spoofing) | L × M = L | M | **L** | Dev configs unauth'd; rises to M if exposed |
| **T-CKPT-1 (pickle RCE)** | L × H = L | **VH** (RCE in operator process) | **M→H** | Likelihood ceiling raised by mirror/HF-org spoofing chain (T-CFG-1) |
| T-CKPT-2 (silent weight swap) | L × H = L | H | **M** | Output integrity / brand |
| T-CFG-1 (HF org redirect) | L × H = L | H | **M** | Defensible with a code-pinned allowlist (FSR_FD_08) |
| **T-PLUG-1 (plugin RCE)** | L × H = L | **VH** | **M→H** | Entry-point discovery is uplift-style attack on dev workstation |
| **T-CONT-1 (CSAM/gore in init RGB)** | M × H = M | **VH** (legal, reputational) | **H** | Mitigated by Cosmos-Guardrail pre-Guard (FSR_FD_06) |
| **T-PRIV-1 (face / plate emission)** | M × M = M | H (privacy regulator + brand) | **H** | Mitigated by Cosmos-Guardrail post-Guard (FSR_FD_07); opt-in for now |
| T-INPUT-1 (DataChannel DoS) | M × M = M | M | **M** | Single active session amplifies |
| T-RECORD-1 (recording perm) | M × M = M | M | **M** | Disk perms control (FSR_FD_21) |
| T-SECRET-1 (token in logs) | M × H = M | H | **H** | Standard hygiene (FSR_FD_02) |
| T-IMG-1 (image tamper) | VL × H = VL | H | **L** | OE-7 collapses the threat to upstream `nvidia/cuda` + operator-side build; FSR_FD_05 is operator guidance |
| T-MOUNT-1 (container pivot) | L × H = L | H | **M** | OE-1 accepted; docs FSR (FSR_FD_22) |
| T-DOS-1 (signaling flood) | M × M = M | M | **M** | Single-session invariant amplifies; FSR_FD_20 rate limits |
| T-FUZZ-1 (proto / scene parse) | L × M = L | M | **L** | Parse surfaces narrow today; FSR_FD_15 / FSR_FD_17 / FSR_FD_24 |

---

## Step 3 — Security Objectives and Functional Security Requirements

### 3.1 Security Objectives

| ID | Objective | C/I/A/Authenticity/Accountability |
|---|---|---|
| SO-1 | Preserve confidentiality of operator-supplied prompts, frames, and session content. | C |
| SO-2 | Guarantee that only NVIDIA-published model weights and container images execute in the TOE. | I, Authenticity |
| SO-3 | Prevent the TOE from generating or amplifying CSAM, gore, or disclosing real-person likenesses without consent. | I, accountability (under privacy regulation) |
| SO-4 | Limit blast radius of any TOE compromise to the operator's session — no privilege escalation to host. | I, A |
| SO-5 | Protect secrets (HF_TOKEN, S3 keys, GitHub PAT) in transit and at rest. | C |
| SO-6 | Maintain availability of the single active session under expected operator load. | A |

### 3.2 Functional Security Requirements (FSRs)

The canonical FSR sheet is [`fsr_table.md`](fsr_table.md), in the **Excel-Based Simple TAVA Template** column structure (`FSR ID | Functional Security Requirement | Risk Level | Risk Response | Responsibility | Priority Level | Test / Measurement`).

`fsr_table.md` populates the NIM TAVA template form (`FSR_<TOE>_NN`) for the FlashDreams TOE by mapping each row to the canonical `FSR_FD_NN` upstream FSR and citing the upstream as-built test (`omni-dreams/internal/tests/security/test_*.py`) where one exists.

Risk-level enum: Very Low / Low / Moderate / High / Very High.
Risk-response enum: Remediate / Transfer / Share / Avoid / Accept.
Priority enum: P0 (Must Implement) / P1 (Will not block GA if not implemented) / R2 (research / recommended post-GA).

### 3.2.1 FSR coverage summary

| Status | Count | FSRs |
|---|---|---|
| ✅ As-built contract+stub tests passing upstream (mirror to FlashDreams planned) | 20 | FSR_FD_01, _02, _03, _04, _06, _08, _09, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _23 |
| 📄 Documented control landed (no test seam needed) | 1 | FSR_FD_22 (README trust-collapse callout) |
| ⏳ Externally blocked or operator-side | 3 | FSR_FD_05 (operator guidance — FlashDreams ships no canonical image), FSR_FD_07 (Cosmos-Guardrail real backend — HF egress per OE-6), FSR_FD_24 (OSS-Fuzz post-GA) |

---

## Step 4 — Plan of Actions and Milestones (POA&M)

Risks are assigned a treatment (Avoid / Accept / Remediate / Transfer / Share) and a target. Items already in flight in the omni-dreams 1.1 plan are cross-linked.

| FSR ID | Item | Risk Response | Target | Owner | Cross-link |
|---|---|---|---|---|---|
| FSR_FD_06 | Input content gate via Cosmos-Guardrail pre-Guard | Remediate | Land in repo + tests before 1.0 GA | SIL eng + Security PIC | Upstream MR (positive/negative test pair landed; real backend pending) |
| FSR_FD_07 | Cosmos-Guardrail post-Guard `--anonymize` opt-in | Remediate | Land `--anonymize` flag + benchmark before 1.0 GA | SIL eng | `omni-dreams/internal/docs/planning/cosmos_guardrail_benchmark.md` |
| FSR_FD_10 | Default-loopback bind | Remediate | Flip defaults; warning banner; ≤2 wk after MR review | SIL eng | Upstream `bind_address.py` |
| FSR_FD_12 | gRPC / HTTP token interceptor | Remediate | Land gRPC interceptor + lingbot HTTP token check before non-loopback bind ships | SIL eng | |
| FSR_FD_04 | Checkpoint integrity manifest | Remediate | Manifest + safetensors-preferred; before 1.1 OSRB close | SIL eng | MVSB-33271; upstream `checkpoint_verifier.py` |
| FSR_FD_08 | HF-org allowlist | Remediate | Land at CLI parse; before 1.0 GA | SIL eng | Upstream `hf_org_allowlist.py` |
| FSR_FD_09 | Plugin allowlist | Remediate | Default-deny in `RunnerRegistry.discover_plugin`; before 1.0 GA | SIL eng | Upstream `plugin_registry.py` |
| FSR_FD_02 | Log redaction of secrets | Remediate | Patch into core logger; before 1.1 GA | SIL eng | Upstream `log_redactor.py` |
| FSR_FD_03 | Log redaction of PII | Share | Land scrubber for face/plate strings; before 1.1 GA | SIL eng + Operator | (paired with FSR_FD_07 output) |
| FSR_FD_13 | DataChannel schema validation | Remediate | Patch lingbot DataChannel handler; before 1.0 GA | SIL eng | Upstream `datachannel_validator.py` |
| FSR_FD_05 | Operator-side image-signing guidance (no canonical NVIDIA image post-`ab74b58`) | Share | Add a `docs/security/SECURITY.md` section recommending `cosign sign` / `cosign verify` for operators who publish FlashDreams containers | Documentation + SIL eng | |
| FSR_FD_01 | TLS 1.2+ on non-loopback endpoints | Remediate | Add reverse-proxy invocation snippet + native TLS support; before non-loopback bind ships | SIL eng | (paired with FSR_FD_10) |
| FSR_FD_11 | Profiling-server separate exposure flag | Remediate | After FSR_FD_10 lands | SIL eng | |
| FSR_FD_14 | Text input validation | Remediate | Add length + Unicode + parameter validation at CLI / gRPC parse | SIL eng | |
| FSR_FD_15 | Media-file input validation | Remediate | Add MIME / magic / size validation on `--synthetic-initial-rgb` + adapter inputs | SIL eng | (input to FSR_FD_06) |
| FSR_FD_16 | Media-parsing sandbox (seccomp) | Remediate | Subprocess-isolate media parsing; post-GA | SIL eng | |
| FSR_FD_17 | Media-parsing format restriction | Remediate | Dispatch-layer format check before parser entry | SIL eng | |
| FSR_FD_18 | Inference state cleared between sessions | Remediate | Audit + zero per-session GPU buffers on tear-down | SIL eng | |
| FSR_FD_19 | Response restricted to requesting session | Remediate | Enforce session_id correlation in ALPA-GRPC | SIL eng | (lingbot already enforces via single-session invariant) |
| FSR_FD_20 | Per-client rate limits | Remediate | Token-bucket on signaling endpoints | SIL eng | |
| FSR_FD_21 | Session recording mode 0600 + off-by-default | Remediate | Tiny patch to recording_io | SIL eng | |
| FSR_FD_22 | Trust-collapse docs for `--network=host` | Share | Reference upstream README docker-run section + add a FlashDreams security note | Documentation + SIL eng | |
| FSR_FD_23 | Slurm tenant isolation banner + IMEX check (omni-dreams consumer) | Share | Banner-only in 1.0; full check post-GA | SIL eng + Cluster Ops | |
| FSR_FD_24 | OSS-Fuzz coverage of parsers | Remediate | Post-GA / dot release | SIL eng | |
| T-MOUNT-1 residual | Accept the trust-collapse for interactive Docker invocation (with OE-1 docs) | Accept | n/a | Security PIC | |

---

## Step 5 — TAVA Closeout Checklist

- [ ] All four arch views attached.
- [ ] SADD attached.
- [ ] All threats traced to ≥1 FSR or a documented accept/transfer rationale.
- [ ] All H-or-higher risks have a P1 FSR.
- [ ] POA&M items have an owner and a milestone.
- [ ] Security PIC review (TBD).
- [ ] Export classification MVSB-32940 decided → pick TAVA 3.0 nSpect ack path or manual TAVA 2.0 link in nSpect.
- [ ] Acknowledge in NSPECT-6O5R-39LY (TAVA 3.0 path) **or** link this doc in `Lifecycle & Documentation → Threat and Vulnerability Analysis (TAVA)` (manual TAVA 2.0 path).
- [ ] MVSB-32946 → Done.

---

## Step 6 — Open Questions for Engineering / Stakeholders

| # | Question | Blocks FSR(s) | Owner |
|---|---|---|---|
| Q-01 | What auth mechanism will alpadreams gRPC + lingbot HTTP support? mTLS, JWT, both? Does it differ between operator-LAN and cross-internet deployments? | FSR_FD_01, FSR_FD_12 | SIL eng + Security PIC |
| Q-02 | When operators publish a FlashDreams container, should Cosmos-1.0-Guardrail weights be pre-staged in the image, or fetched at first run? Pre-staging closes OE-6 in restricted-egress environments. | FSR_FD_06, FSR_FD_07 | Documentation + SIL eng |
| Q-03 | What is the canonical FSR_FD_08 allowlist on GA? Today the env-var defaults to `nvidia` (per `integrations/alpadreams/alpadreams/hf.py`); the documented internal mirror is `nvidia-omni-dreams-lha`. Confirm whether GA retains both or rotates. | FSR_FD_08 | Product Lead |
| Q-04 | Are entry-point-discovered plugins expected at all on GA? If not, FSR_FD_09 can be tightened from allowlist to **always-deny**. | FSR_FD_09 | SIL eng |
| Q-05 | What's the retention policy for opt-in session recordings (`.pt` / `.json`)? Strictly local? | FSR_FD_21, FSR_FD_03 | Privacy Review (MVSB-33273) |
| Q-06 | When does the FlashDreams-side `flashdreams/tests/security/` mirror of the upstream `omni-dreams/internal/tests/security/` package land? | All FSRs with as-built tests | SIL eng |
| Q-07 | What is the export-classification outcome of [MVSB-32940](https://jirasw.nvidia.com/browse/MVSB-32940)? | TAVA closeout path | Export Compliance |
| Q-08 | Is there a named **Security PIC** for the FlashDreams 1.0 GA release? Closeout requires PIC review and acknowledgment. | All P0 FSRs | Program Management |

---

## Step 7 — Recommended Next Steps

| Priority | Action | Owner | MR / Ticket |
|---|---|---|---|
| **P0** | Land Cosmos-1.0-Guardrail pre-Guard real backend at session init (FSR_FD_06) | SIL eng + Security PIC | Upstream + benchmark follow-up |
| **P0** | Flip default network bind to loopback in lingbot + alpadreams (FSR_FD_10); warning banner when `--allow-public-bind` set | SIL eng | Upstream `bind_address.py` |
| **P0** | Land checkpoint integrity verifier in `FD-CKPT` path before `torch.load` (FSR_FD_04) | SIL eng | Upstream + MVSB-33271 |
| **P0** | Wire log-redaction filter into core logger (FSR_FD_02) + PII scrub for face/plate strings (FSR_FD_03) | SIL eng | Upstream `log_redactor.py` |
| **P0** | Wire HF-org allowlist into CLI parse (FSR_FD_08) | SIL eng | Upstream `hf_org_allowlist.py` |
| **P0** | Wire plugin allowlist into `RunnerRegistry.discover_plugin` (FSR_FD_09); decide whether GA permits plugins at all (Q-04) | SIL eng | Upstream `plugin_registry.py` |
| **P0** | Wire DataChannel schema validator into lingbot.webrtc.session (FSR_FD_13) | SIL eng | Upstream `datachannel_validator.py` |
| **P0** | Define and implement gRPC + HTTP token interceptor (FSR_FD_12); mandatory when non-loopback | SIL eng | Follow-up MR |
| **P0** | Mirror upstream `internal/tests/security/` into `flashdreams/tests/security/` | SIL eng | This MR + follow-up |
| **P1** | Land `--anonymize` post-Guard opt-in on output (FSR_FD_07) | SIL eng | Cosmos-Guardrail benchmark plan |
| **P1** | Document operator-side Sigstore guidance in `docs/security/SECURITY.md` (FSR_FD_05) — FlashDreams ships no canonical image, so this is "if you publish, sign and verify" | Documentation + SIL eng | Follow-up MR |
| **P1** | Add text + media input validation (FSR_FD_14, FSR_FD_15) | SIL eng | Follow-up MR |
| **P1** | Audit per-session GPU buffer cleanup (FSR_FD_18); test cross-session leak (FSR_FD_19) | SIL eng | Follow-up MR |
| **P1** | Rate-limit signaling endpoints (FSR_FD_20) | SIL eng | Follow-up MR |
| **R2** | Media-parsing seccomp sandbox (FSR_FD_16) and format-dispatch restriction (FSR_FD_17) | SIL eng | Post-GA dot release |
| **R2** | OSS-Fuzz integration for proto / parse seams (FSR_FD_24) | SIL eng | Post-GA dot release |

---

## Appendix A — Recommendations summary

> The single highest-leverage change is to land **Cosmos-1.0-Guardrail pre-Guard at session init** as a default-on hard gate (FSR_FD_06) — it closes the highest-impact threat (T-CONT-1) with one model call per session, and the reference implementation already exists. Pair it with **default loopback binds for all listeners** (FSR_FD_10) and a **checkpoint-integrity manifest** (FSR_FD_04), and the highest-risk threats reduce to residual M-level operational controls before FlashDreams 1.0 GA.

---

## Appendix B — Mapping to Excel-Based Simple TAVA Template

| Excel tab | Source in this doc |
|---|---|
| Introduction | Step 0 (table) + Executive Summary |
| Target of Evaluation | Step 1.1, 1.2, 1.3 (OE-1..OE-7) |
| Architecture | Step 1.4 (links to `architecture.md`) |
| Dataflow | Step 1.5 (links to `architecture.md` Data view) |
| Security Assets | Step 1.6 (C/I/A sensitivity grid) |
| Attacker Model | Step 2.1 |
| Threat Model | Step 2.2 (trust boundaries) + 2.3 (STRIDE inventory) |
| Risk Analysis | Step 2.4 (NIST SP 800-30r1 qualitative; matrix + per-threat table) |
| Security Objectives | Step 3.1 (SO-1..SO-6) |
| FSRs | [`fsr_table.md`](fsr_table.md) (canonical `FSR_FD_NN` sheet with Excel column layout) |
| POA&M | Step 4 (FSR_FD_NN-keyed) |
| Open Questions | Step 6 |
| Recommended Next Steps | Step 7 |
