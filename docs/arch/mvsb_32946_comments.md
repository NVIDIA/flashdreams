<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Comments to post on MVSB-32946

Two prepared Jira comments for [MVSB-32946 — *Security: TAVA - Submit and Acknowledge - [flashdreams] [GA and OSS]*](https://jirasw.nvidia.com/browse/MVSB-32946). The Jira CLI on this worktree is not authenticated, and the MaaS-Jira MCP requires interactive browser OAuth, so these were generated for the user to paste manually (or to drop in via `jira-cli issue comment MVSB-32946 -b "$(cat …)"` once `jira-cli auth set-token` has been run).

---

## Comment 1 — FlashDreams-scoped TAVA / SADD / FSR doc set has landed in-tree

**Subject:** TAVA doc set landed in `flashdreams/docs/arch/`

Posting the FlashDreams-scoped subset of the TAVA materials in-tree under `flashdreams/docs/arch/` on branch `worktree-dev/jmccaffrey/arch-tava`:

| Doc | Purpose |
| --- | --- |
| `docs/arch/README.md` | Index + cross-refs to the upstream omni-dreams doc set |
| `docs/arch/SKILL.md` | Agent-runnable security-architect workflow (compact subset of the upstream skill) |
| `docs/arch/architecture.md` | Four-view architecture (Static / Dynamic / Data / Deployment) narrowed to the FlashDreams TOE |
| `docs/arch/sadd.md` | SADD per `SWE-PLC-L1-002-BasicPLC-SADD-TMPL` |
| `docs/arch/tava.md` | TAVA narrative (assets / threats / risk / objectives / POA&M) for the FlashDreams TOE |
| `docs/arch/fsr_table.md` | Canonical FSR sheet in the Excel-Based Simple TAVA Template column structure (FSR_FD_01..36) |

**TOE shape (FlashDreams subset):**

In scope: `flashdreams/flashdreams/` (recipes `cosmos`/`taehv`/`template`/`wan`, core, infra, plugins, scripts) plus `integrations/{alpadreams, lingbot, cosmos_predict2, wan21, self_forcing, causal_forcing, fastvideo_causal_wan22}`. Supply chain: HuggingFace + S3 (`pdx.s8k.io`) + upstream `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` base image.

Out of scope: `omni-dreams/samples/interactive-drive` and `omni-dreams/post-training` (covered upstream in [`omni-dreams/internal/docs/planning/tava_omni_dreams_flashdreams.md`](https://gitlab-master.nvidia.com/sil/omni-dreams)).

**Key drift correction vs. the upstream 2026-05-15 TAVA snapshot:** per commit `ab74b58` ("Remove ghcr.io/nvidia/flashdreams base image from CI and scripts"), FlashDreams ships **no canonical pre-published container image**. Operators build locally from `docker/Dockerfile`. As a result:

- OE-7 reframed from "GHCR ACL'd image" to "no canonical NVIDIA-published image; supply-chain trust delegates upward to upstream `nvidia/cuda`".
- T-IMG-1 risk **downgraded from M to L** (likelihood drops because there is no central NVIDIA distribution point).
- FSR_FD_05 (Sigstore) **downgraded from P1 to P2** and reframed as **operator-side guidance** ("if you publish a FlashDreams container, `cosign sign`; if you pull one, `cosign verify`"). The FlashDreams-side enforcement disappears.

**FSR sheet column structure** matches the NIM TAVA template you referenced in the parent ticket: `FSR ID | Functional Security Requirement | Risk Level | Risk Response | Responsibility | Priority Level | Test / Measurement`. NIM-template slots 25..36 are mapped onto existing `FSR_FD_*` rows (or marked **Avoid** where the FlashDreams TOE invariants — no NIM API surface, no Triton in the TOE, no NGC publishing of FlashDreams models — make the slot moot). The mapping table is at the bottom of `docs/arch/fsr_table.md`.

Ready for Security PIC review pending the MVSB-32940 export classification outcome (which decides whether we go through nSpect TAVA 3.0 ack or stay on the manual TAVA 2.0 link). I'll watch this ticket for sign-off; let me know if the FSR risk levels / responsibilities need adjustment before that.

---

## Comment 2 — FSR coverage status + open questions before acknowledgment

**Subject:** FSR coverage at acknowledgment-time + open questions

Coverage as of `worktree-dev/jmccaffrey/arch-tava` HEAD:

| Status | Count | FSRs |
| --- | --- | --- |
| ✅ As-built contract+stub tests passing upstream (mirror to `flashdreams/tests/security/` planned) | 20 | FSR_FD_01, _02, _03, _04, _06, _08, _09, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _23 |
| 📄 Documented control (no test seam needed) | 1 | FSR_FD_22 (trust-collapse callout — split between FlashDreams `docs/security/SECURITY.md` and the upstream interactive-drive README) |
| ⏳ Externally blocked or operator-side | 3 | FSR_FD_05 (operator-side image-signing guidance), FSR_FD_07 (Cosmos-Guardrail real backend pending HF-egress benchmark — see `omni-dreams/internal/docs/planning/cosmos_guardrail_benchmark.md`), FSR_FD_24 (OSS-Fuzz post-GA) |
| 🔁 NIM-template slots mapped onto an existing `FSR_FD_NN` | 9 | FSR_FD_25, _27, _28, _29, _30, _31, _32, _33, _36 |
| 🚫 NIM-template slots avoided by FlashDreams TOE architecture invariant | 3 | FSR_FD_26 (no managed SLA), FSR_FD_34, _35 (no Triton in TOE) |
| **Total slots covered** | **36** | 24 canonical + 12 NIM-template mapping rows |

**Highest-leverage residual ask before FlashDreams 1.0 GA**: land Cosmos-1.0-Guardrail pre-Guard (FSR_FD_06) as a default-on hard gate at session init. One model call per session; closes the highest-impact threat (T-CONT-1 — operator-supplied CSAM/gore initial frame amplified by the world model). The reference implementation exists; the seam is already prototyped upstream.

**Week-1 mitigation candidates** (all P0; ~3.75 engineering-days total): see `docs/arch/tava.md` Step 7 Recommended Next Steps. Six items have working contract+stub tests upstream that need product-side wiring in the FlashDreams repo (loopback bind default + warning banner; log-redaction filter; HF-org allowlist at config parse; plugin-allowlist default-deny; session-init input gate plumbing; DataChannel JSON schema validation).

**Open questions blocking specific FSR sign-off** (full list in `docs/arch/tava.md` Step 6):

1. **Q-01** — auth mechanism for ALPA-GRPC + lingbot HTTP signaling: mTLS, JWT, both? Blocks FSR_FD_01 + FSR_FD_12.
2. **Q-02** — should operators pre-stage Cosmos-1.0-Guardrail weights in their FlashDreams container, or fetch at first run? Pre-staging closes OE-6 in restricted-egress environments. Blocks FSR_FD_06 + FSR_FD_07.
3. **Q-03** — canonical FSR_FD_08 allowlist on GA: today the env-var defaults to `nvidia` (per `integrations/alpadreams/alpadreams/hf.py`); the documented internal mirror is `nvidia-omni-dreams-lha`. Confirm GA retains both or rotates. Blocks FSR_FD_08.
4. **Q-04** — are entry-point-discovered plugins expected at all on GA? If not, FSR_FD_09 tightens from allowlist to **always-deny**. Blocks FSR_FD_09 hardening.
5. **Q-06** — when does the FlashDreams-side `flashdreams/tests/security/` mirror of the upstream `omni-dreams/internal/tests/security/` package land? (Without it, the FSR coverage table cites upstream test counts, which is fine for the TAVA acknowledgment but suboptimal for an external operator inspecting the FlashDreams repo alone.)
6. **Q-07** — MVSB-32940 export classification outcome (decides TAVA 3.0 vs. manual TAVA 2.0 ack path).
7. **Q-08** — named Security PIC for the FlashDreams 1.0 GA release? (Closeout requires PIC review + acknowledgment.)

Requesting Security PIC review when convenient. Happy to walk through any of the above on a quick sync.
