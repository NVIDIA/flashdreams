<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlashDreams FSR table — Excel-Based Simple TAVA Template

**Canonical Functional Security Requirement (FSR) sheet for the FlashDreams TOE**, in the column structure used by the NIM TAVA template that the user supplied for MVSB-32946:

`FSR ID | Functional Security Requirement | Risk Level | Risk Response | Responsibility | Priority Level | Test / Measurement`

Companion to:

- [`architecture.md`](architecture.md) — four-view FlashDreams architecture
- [`sadd.md`](sadd.md) — Software Architecture and Design Document
- [`tava.md`](tava.md) — TAVA narrative (assets, threats, risk, objectives, POA&M)

Conventions:

- FSR IDs prefixed `FSR_FD_NN` (FD = FlashDreams). The leading 24 rows mirror the canonical upstream omni-dreams + flashdreams sheet at [`../../../omni-dreams/internal/docs/planning/tava_omni_dreams_flashdreams.md`](../../../omni-dreams/internal/docs/planning/tava_omni_dreams_flashdreams.md) §3.2.1.
- The user-supplied NIM template enumerated 36 FSR slots (`FSR_<TOE>_01..36`). FSR slots `FSR_FD_25..36` cover NIM-template areas (e.g. mandatory-AuthN/AuthZ, scaling, non-prod parity, request-scope memory) that map to existing `FSR_FD_*` SHALL-statements or are out-of-scope for the FlashDreams TOE (no NVIDIA-side ingress / egress / storage; no NGC publishing pipeline; no NIM API surface). For each, the **Risk Response** column captures the mapping (Map / Transfer / Avoid / Accept).
- Risk-level enum: Very Low / Low / Moderate / High / Very High.
- Risk-response enum: Remediate / Transfer / Share / Avoid / Accept (extended with **Map** to indicate this row maps onto an existing FSR_FD_NN).
- Priority enum: P0 (Must Implement) / P1 (Will not block GA if not implemented) / R2 (research / recommended post-GA).

---

## FSR ID column structure (per row)

Every row uses the column structure the user supplied:

> **Functional Security Requirement** — Detailed security requirement decomposed from the high-level security objectives; provides the development team rationale and actionable direction on what priority countermeasures need to be considered in the architecture and design.
> **Risk Level** — Calculated risk level associated with the risk that the FSR mitigates (Very Low / Low / Moderate / High / Very High), drawn from [`tava.md`](tava.md) § 2.4.2.
> **Risk Response** — Risk response associated with the risk that the FSR mitigates (Remediate / Transfer / Share / Avoid / Accept / Map).
> **Responsibility** — Team / org / element responsible for implementing the FSR.
> **Priority Level** — P0 (Must Implement) / P1 (Will not block if not implemented) / R2.
> **Test / Measurement** — Whether the FSR is testable by SQA, and the test steps that verify it.

---

## Canonical FSR sheet

### FSR_FD_01 — TLS on non-loopback endpoints

- **Functional Security Requirement.** All non-loopback access to FlashDreams server endpoints (LING-RTC HTTP signaling, ALPA-GRPC, ALPA-PROF) SHALL be encrypted with TLS 1.2 or better when bound non-loopback, and SHALL meet the specifications in the prodsec wiki. Where available, prefer profiles such as `ELBSecurityPolicy-TLS-1-2-2017-01` for AWS-fronted deployments. WebRTC media SHALL continue to use DTLS-SRTP (aiortc default — equivalent to TLS for media). Loopback-only deployments MAY skip TLS per operator policy.
- **Risk Level.** High.
- **Risk Response.** Remediate.
- **Responsibility.** SIL Engineering — Serving (flashdreams `integrations/alpadreams/alpadreams/grpc/server.py`, `integrations/lingbot/lingbot/webrtc/server.py`).
- **Priority Level.** P0 (non-loopback) / P1 (loopback).
- **Test / Measurement.**
  - SQA Test Steps:
    1. Pre-stage TLS profile config (`modern`, `compat`, or `mtls`).
    2. Start LING-RTC + ALPA-GRPC with `--allow-public-bind` and the configured TLS profile.
    3. Run `pytest internal/tests/security/test_tls_profile.py -v` (upstream — to be mirrored into `flashdreams/tests/security/test_tls_profile.py`).
  - PASS: `modern` returns TLS 1.3; `compat` returns TLS 1.2 with AEAD-only ciphers (no RC4 / 3DES / CBC-SHA1); `mtls` requires client cert; loopback-skip allowed. `tls10` / `tls11` / `sslv3` / `weak` / `any` and unknown profiles explicitly refused. Sweep test asserts every defined profile meets the TLS-1.2 floor.
  - FAIL: any non-loopback endpoint accepts an unencrypted or TLS < 1.2 connection.
  - Pentest: `nmap --script ssl-enum-ciphers -p <port>` against a public-bound LING-RTC reports no TLS 1.0 / 1.1 / SSLv3 acceptance.

### FSR_FD_02 — Logs SHALL strip secrets

- **Functional Security Requirement.** Service log data stored locally or transmitted remotely SHALL NOT contain NVIDIA confidential data such as personal and enterprise secrets, API keys, cryptographic keys, or authentication tokens (`HF_TOKEN`, `GITHUB_PAT`, AWS access / secret keys, bearer tokens, full `Authorization` headers).
- **Risk Level.** High.
- **Risk Response.** Remediate.
- **Responsibility.** SIL Engineering — Logging (`flashdreams/flashdreams/core/io/`).
- **Priority Level.** P0.
- **Test / Measurement.**
  - SQA Test Steps:
    1. Configure FlashDreams with a fake `HF_TOKEN` (e.g. `hf_…`), a synthetic GitHub PAT (`ghp_…`), and an AWS access-key pair.
    2. Run `flashdreams-run <slug>` and exercise representative code paths (CLI offline run; lingbot session; gRPC `InitializeSession → Step → Close`).
    3. Run `pytest internal/tests/security/test_log_redactor.py -v` (upstream — 11 secrets-redaction tests + the `logging.Filter` end-to-end test) against the production log writer.
  - PASS: 11 / 11 redaction cases; the `logging.Filter` end-to-end test confirms secret bytes never reach the sink. As-built upstream on station 2026-05-15: 11 / 11 ✅.
  - FAIL: any pattern reaches the sink unredacted.
  - Pentest: grep the on-disk log files for `hf_`, `ghp_`, `AKIA`, `Bearer ` — no matches.

### FSR_FD_03 — Logs SHALL strip PII

- **Functional Security Requirement.** Service log data (excluding log fields that contain customer-supplied data clearly labelled as such) SHALL NOT contain personally identifiable information such as real-person face crops extracted from generated frames, license-plate strings extracted from frames, NVIDIA-internal emails, Slack handles, or operator network identifiers beyond what is needed for a single debug session.
- **Risk Level.** Moderate.
- **Risk Response.** Share (FlashDreams logging filter + operator's own scrubbing on export).
- **Responsibility.** SIL Engineering — Logging + Operator.
- **Priority Level.** P0.
- **Test / Measurement.**
  - SQA Test Steps:
    1. Configure FlashDreams with sample PII payloads in prompts and frames (synthetic faces / plates).
    2. Run representative paths.
    3. Run `pytest internal/tests/security/test_log_redactor.py -v` — 11 PII tests on top of the 11 FSR_FD_02 secrets tests.
  - PASS: positive coverage for NVIDIA email (incl. regional `nvidia-<region>.com`), ALPR plate shapes (`ABC123`, `123ABC`, `ABC-1234`), Slack user / channel / DM IDs; anti-bypass for "external-domain emails MUST NOT be silently scrubbed" and "all-caps words like CHECKPOINT MUST NOT match the Slack-ID regex". Upstream on station 2026-05-15: 22 / 22 in the combined redactor file ✅.
  - FAIL: any PII pattern reaches the sink unredacted.

### FSR_FD_04 — Checkpoint integrity before `torch.load`

- **Functional Security Requirement.** Model checkpoints loaded by `FD-CKPT` SHALL be verified against a trusted manifest (SHA-256 today, Sigstore signature once FSR_FD_05 lands) **before** invoking `torch.load`. Prefer `safetensors` for new internal artifacts. Parties downloading published checkpoints SHALL have the ability to verify integrity and authenticity via signature verification (in the OSS flow, the manifest + signature ship alongside the weight files).
- **Risk Level.** Very High.
- **Risk Response.** Remediate.
- **Responsibility.** SIL Engineering — Pipeline (`flashdreams/flashdreams/core/checkpoint/`).
- **Priority Level.** P0.
- **Test / Measurement.**
  - SQA Test Steps:
    1. Pre-stage a checkpoint + manifest pair (`weights.safetensors` + `manifest.json` with sha256).
    2. Run `flashdreams-run <slug>` and verify load succeeds.
    3. Tamper with a single byte in `weights.safetensors` (`dd if=/dev/urandom of=weights.safetensors bs=1 count=1 conv=notrunc seek=1024`).
    4. Re-run; verify the loader rejects the checkpoint BEFORE `torch.load`.
    5. Run `pytest internal/tests/security/test_checkpoint_verifier.py -v` (upstream).
  - PASS: 9 / 9, including the single-byte-tamper assertion that `loader_fn.invocations == []` on every failure mode (mismatch, missing-manifest, missing-file, wrong-path-type, `/proc` TOCTOU). Upstream on station 2026-05-15: 9 / 9 ✅.
  - FAIL: tampered checkpoint is loaded into memory.

### FSR_FD_05 — Operator-side container-image signing guidance

- **Functional Security Requirement.** FlashDreams ships **no canonical pre-published container image** (per commit `ab74b58`); operators build locally from `docker/Dockerfile`. When an operator publishes a FlashDreams container to their own registry, the build/publish pipeline SHOULD Sigstore-sign the image and the documented `docker pull` flow SHOULD `cosign verify` the signature before the image is allowed to start. The FlashDreams `docs/security/SECURITY.md` SHALL carry this guidance.
- **Risk Level.** Low (downgraded from High once OE-7 / commit `ab74b58` removed the canonical image — see [`tava.md`](tava.md) § 2.4.2 T-IMG-1).
- **Risk Response.** Share (operator-side build, FlashDreams documentation).
- **Responsibility.** Documentation + SIL Engineering (with the operator).
- **Priority Level.** P2 (operator guidance).
- **Test / Measurement.**
  - SQA Test Steps:
    1. Confirm `docs/security/SECURITY.md` carries a "container-image signing" section recommending `cosign sign` at publish time and `cosign verify` at pull time.
    2. (Operator-side, illustrative only — not part of FlashDreams CI:) `cosign verify <operator-registry>/flashdreams:<tag>` → expect success on a signed image; non-zero on a tampered or unsigned image.
  - PASS: doc section present; operator-side verification command documented.
  - FAIL: no doc section, or the doc fails to mention the upstream `nvidia/cuda` base image as the trust root.

### FSR_FD_06 — Input content gate at session init

- **Functional Security Requirement.** The session SHALL run **Cosmos-1.0-Guardrail pre-Guard** (image side, sourced from `imaginaire/auxiliary/guardrail/video_content_safety_filter/`) on the initial RGB frame (`--synthetic-initial-rgb`, scene first frame, WebRTC `request_session` first frame) once per session. On hit, the session SHALL be rejected with a documented error code; **no inference runs**.
- **Risk Level.** Very High.
- **Risk Response.** Remediate.
- **Responsibility.** SIL Engineering — Safety + Security PIC.
- **Priority Level.** P0.
- **Test / Measurement.**
  - SQA Test Steps:
    1. Pre-stage `nvidia/Cosmos-1.0-Guardrail` weights (OE-5; pre-staged per OE-6).
    2. Start LING-RTC; submit a synthetic CSAM / gore probe image as the initial RGB.
    3. Verify the session is rejected with the documented error code; verify no inference call is made (`pipeline.invocations == 0`).
    4. Run `pytest internal/tests/security/test_input_content_gate.py -v` (upstream).
  - PASS: 5 / 5; the negative case asserts `pipeline.invocations == 0` on reject. Upstream on station 2026-05-15: 5 / 5 stub ✅; real backend pending [`cosmos_guardrail_benchmark.md`](../../../omni-dreams/internal/docs/planning/cosmos_guardrail_benchmark.md).
  - FAIL: a probe image classified by the gate as unsafe still results in pipeline invocation.

### FSR_FD_07 — Output anonymizer (opt-in `--anonymize`)

- **Functional Security Requirement.** When `--anonymize` is set, the session SHALL run **Cosmos-1.0-Guardrail post-Guard** (RetinaFace + plate detect + Gaussian blur, sourced from `imaginaire/auxiliary/guardrail/face_blur_filter/`) on every generated chunk, blurring detected face and license-plate regions. Steady-state overhead SHALL be ≤50 ms / chunk.
- **Risk Level.** High.
- **Risk Response.** Remediate.
- **Responsibility.** SIL Engineering — Pipeline + Privacy.
- **Priority Level.** P1 (1.0) / P0 (post-GA if regulator scope shifts).
- **Test / Measurement.**
  - SQA Test Steps:
    1. Pre-stage Cosmos-Guardrail weights.
    2. Run with `--anonymize` against a frame containing a synthetic face → output frame has face region blurred above SSIM threshold.
    3. Repeat with a plate region.
    4. Measure per-chunk overhead → assert ≤50 ms median on reference HW.
  - PASS: blur SSIM threshold met on face + plate; overhead ≤50 ms median; opt-out path leaves frames unmodified.
  - FAIL: face / plate visible above threshold, or overhead exceeds 50 ms median. Latency-budget validation pending [`cosmos_guardrail_benchmark.md`](../../../omni-dreams/internal/docs/planning/cosmos_guardrail_benchmark.md).

### FSR_FD_08 — HF-org allowlist

- **Functional Security Requirement.** `OMNI_DREAMS_HF_ORG` SHALL be validated against a code-resident allowlist `{nvidia, nvidia-omni-dreams-lha}` (default: `nvidia`, per `integrations/alpadreams/alpadreams/hf.py:DEFAULT_OMNI_DREAMS_HF_ORG`). Unknown values SHALL be rejected at config parse, BEFORE any HF resolver runs. (There is no `--hf-org` CLI flag today — the env-var is the single configuration surface; if a flag is added later, the same validation SHALL apply.)
- **Risk Level.** High.
- **Risk Response.** Remediate.
- **Responsibility.** SIL Engineering — Adapter HF resolver (`integrations/alpadreams/alpadreams/hf.py`) and any other recipe / adapter consuming the env-var.
- **Priority Level.** P0.
- **Test / Measurement.**
  - SQA Test Steps:
    1. Set `OMNI_DREAMS_HF_ORG=attacker-controlled-org` and run `flashdreams-run <slug>` → expect immediate reject at config parse, no HF resolver invocation.
    2. Set `OMNI_DREAMS_HF_ORG=NVIDIA-OMNI-DREAMS-LHA` (case smuggling) → expect reject unless the allowlist is explicitly case-insensitive (and the doc says so).
    3. Run the existing `integrations/alpadreams/tests/test_internal_storage.py` and the upstream `pytest internal/tests/security/test_hf_org_allowlist.py -v`.
  - PASS: 9 / 9 upstream; typo-squat rejection. Default `nvidia` accepted; `nvidia-omni-dreams-lha` accepted; everything else rejected.
  - FAIL: an unknown org reaches the HF resolver.

### FSR_FD_09 — Plugin registry default-deny

- **Functional Security Requirement.** The FlashDreams runner registry SHALL refuse to register a third-party plugin whose origin module is not in the `FLASHDREAMS_TRUSTED_PLUGINS` allowlist, unless `--allow-untrusted-plugins` is passed. A plugin attempting to register a slug that an in-tree runner already owns SHALL be rejected unconditionally.
- **Risk Level.** Very High.
- **Risk Response.** Remediate.
- **Responsibility.** SIL Engineering — Recipe Registry (`flashdreams/flashdreams/plugins/registry.py`).
- **Priority Level.** P0.
- **Test / Measurement.**
  - SQA Test Steps:
    1. Install a synthetic out-of-tree plugin that exposes a `flashdreams.runners` entry-point claiming an unknown slug.
    2. Run `flashdreams-run --help` → expect plugin NOT registered; factory NOT called.
    3. Install a plugin that claims an in-tree slug (`wan21-t2v-1.3b-480p`) → expect unconditional reject.
    4. Run `pytest internal/tests/security/test_plugin_registry.py -v` (upstream).
  - PASS: 11 / 11, including the invariant `untrusted_factory.calls == 0` during refusal and the in-tree-slug-shadow-rejection assertion. Upstream on station 2026-05-15: 11 / 11 ✅.
  - FAIL: an untrusted plugin is imported by `flashdreams-run`.

### FSR_FD_10 — Default-loopback bind

- **Functional Security Requirement.** All FlashDreams network listeners (LING-RTC, ALPA-GRPC, ALPA-PROF) SHALL default to `127.0.0.1`. Any non-loopback bind (wildcard OR specific routable IP) SHALL require an explicit `allow_public_bind=True` / `--allow-public-bind` opt-in and SHALL emit a warning banner naming the co-requisite FSR_FD_01 (TLS) and FSR_FD_12 (auth).
- **Risk Level.** High.
- **Risk Response.** Remediate.
- **Responsibility.** SIL Engineering — Serving.
- **Priority Level.** P0.
- **Test / Measurement.**
  - SQA Test Steps:
    1. Start LING-RTC and ALPA-GRPC with default flags; verify `ss -ltn` shows binding to `127.0.0.1` only.
    2. Pass `--host 0.0.0.0` without `--allow-public-bind` → expect refusal at startup.
    3. Pass `--allow-public-bind` → expect bind succeeds and stderr emits the FSR_FD_10 warning banner.
    4. Pass `--host 10.0.0.5` (specific routable) without opt-in → expect refusal.
    5. Run `pytest internal/tests/security/test_bind_address.py -v` (upstream).
  - PASS: 12 / 12, including the anti-bypass "specific routable IP without opt-in is refused". Upstream on station 2026-05-15: 12 / 12 ✅.
  - FAIL: any listener binds non-loopback without opt-in.

### FSR_FD_11 — Profiling server bind loopback by default

- **Functional Security Requirement.** The alpadreams profiling server SHALL bind loopback by default; non-loopback exposure SHALL require a separate explicit flag (`--allow-public-profiling`) distinct from the main server's `--host`.
- **Risk Level.** Moderate.
- **Risk Response.** Remediate.
- **Responsibility.** SIL Engineering — Serving (`integrations/alpadreams/alpadreams/grpc/profiling_server.py`).
- **Priority Level.** P1.
- **Test / Measurement.**
  - SQA Test Steps:
    1. Start ALPA-GRPC with `--allow-public-bind` (FSR_FD_10) but no profiling-specific opt-in; verify profiling server still binds loopback.
    2. Pass `--allow-public-profiling` → expect non-loopback bind + the profiling-specific banner naming FSR_FD_11.
    3. Run `pytest internal/tests/security/test_profiling_bind.py -v` (upstream).
  - PASS: 8 / 8, including the architectural anti-bypass test asserting the profiling endpoint MUST NOT follow the main listener without its own opt-in. Upstream on station 2026-05-15: 8 / 8 ✅.
  - FAIL: profiling server binds non-loopback without its own opt-in.

### FSR_FD_12 — Token interceptor (mandatory non-loopback)

- **Functional Security Requirement.** The alpadreams gRPC server and the lingbot signaling HTTP endpoint SHALL support a token interceptor that authenticates every call. Interceptor SHALL be mandatory when the listener is bound non-loopback (FSR_FD_10 opt-in).
- **Risk Level.** Very High.
- **Risk Response.** Remediate.
- **Responsibility.** SIL Engineering — Serving.
- **Priority Level.** P0 (non-loopback) / P1 (loopback).
- **Test / Measurement.**
  - SQA Test Steps:
    1. Configure ALPA-GRPC with a bearer-token allowlist.
    2. Call `InitializeSession` without an `Authorization` header → expect refusal; verify `handler.calls == 0`.
    3. Repeat with wrong token, lowercase header, multiple Authorization headers (smuggling), constant-time partial prefix.
    4. Run `pytest internal/tests/security/test_auth_interceptor.py -v` (upstream).
  - PASS: 12 / 12. Golden bearer path succeeds; every failure mode asserts `handler.calls == 0`. Upstream on station 2026-05-15: 12 / 12 ✅.
  - FAIL: any unauthorized request reaches the handler.

### FSR_FD_13 — DataChannel schema validation

- **Functional Security Requirement.** The lingbot DataChannel SHALL validate each message as `{action ∈ {keydown, keyup, step}, key? ∈ {w, a, s, d, space, shift, ctrl}}` with payload ≤256 bytes. Messages outside this schema SHALL be dropped silently without disturbing the active session.
- **Risk Level.** Moderate.
- **Risk Response.** Remediate.
- **Responsibility.** SIL Engineering — Serving (`integrations/lingbot/lingbot/webrtc/session.py`).
- **Priority Level.** P0.
- **Test / Measurement.**
  - SQA Test Steps:
    1. Send well-formed actions; verify enqueue.
    2. Send malformed JSON, oversized payloads, unknown action / key fields; verify silent drop and the session remains active.
    3. Run `pytest internal/tests/security/test_datachannel_validator.py -v` (upstream).
  - PASS: 15 / 15, including the 11-case adversarial sweep that asserts the validator never raises. Upstream on station 2026-05-15: 15 / 15 ✅.
  - FAIL: a malformed payload crashes the session or reaches the runner.

### FSR_FD_14 — Text input validation

- **Functional Security Requirement.** All text inputs to FlashDreams APIs (`--prompt`, recipe CLI text fields, gRPC `Step` prompt) SHALL be length-bounded (≤4096 chars default), Unicode-validated (reject the `U+E0020..U+E007F` "tag" range and other non-printables), and parameter-typed before being passed to `text_encoder`.
- **Risk Level.** Low.
- **Risk Response.** Remediate.
- **Responsibility.** SIL Engineering — Pipeline.
- **Priority Level.** P0.
- **Test / Measurement.**
  - SQA Test Steps:
    1. Send a prompt with a tag-char (e.g. `U+E0041` inside or appended), a Trojan-Source bidi-override, a lone surrogate, ZWJ flood (>16), null / bell / DEL controls, oversize (>4096) → each rejected.
    2. Send legitimate multilingual text, emoji-ZWJ, tab / newline / CR → accepted.
    3. Run `pytest internal/tests/security/test_text_input_validator.py -v` (upstream).
  - PASS: 17 / 17. Every reject asserts `encoder.calls == 0`; adversarial-sweep test asserts the validator never raises. Upstream on station 2026-05-15: 17 / 17 ✅.
  - FAIL: any invalid input reaches the encoder.

### FSR_FD_15 — Media-file input validation

- **Functional Security Requirement.** All media-file inputs (JPEG / PNG initial RGB, mp4 reference clips, USDZ scenes when consumed via downstream samples) SHALL be size-bounded (≤512 MiB default), MIME-validated, magic-byte-validated, and rejected on parse failure BEFORE any downstream rendering or inference. Reject array-batched media inputs above 16 elements (`MAX_BATCH_SIZE`).
- **Risk Level.** Moderate.
- **Risk Response.** Remediate.
- **Responsibility.** SIL Engineering — CLI + adapter input layers.
- **Priority Level.** P0.
- **Test / Measurement.**
  - SQA Test Steps:
    1. Submit valid JPEG / PNG / mp4 (with matching `ftyp`) / USDZ → accepted.
    2. Submit oversize, empty, unknown-extension, smuggled magic (`.jpg` with PNG bytes), non-`ftyp` mp4, directory, nonexistent path → each rejected.
    3. Submit a batch of 17 → expect whole-batch reject.
    4. Run `pytest internal/tests/security/test_media_file_validator.py -v` (upstream).
  - PASS: 18 / 18. Every reject asserts `parser.calls == 0`. Upstream on station 2026-05-15: 18 / 18 ✅.
  - FAIL: an invalid media file reaches the parser.

### FSR_FD_16 — Media-parsing sandbox

- **Functional Security Requirement.** Media-parsing libraries used by FlashDreams adapters (and any image / video decoders called by `--synthetic-initial-rgb`) SHALL be sandboxed (subprocess with seccomp filter or equivalent) to limit syscalls and memory access from a malicious media file.
- **Risk Level.** Moderate.
- **Risk Response.** Remediate.
- **Responsibility.** SIL Engineering — Adapter input layer.
- **Priority Level.** P1 (1.0) / P0 (post-GA).
- **Test / Measurement.**
  - SQA Test Steps:
    1. Confirm parsing runs in a `multiprocessing.Process` (spawn context) with `PR_SET_NO_NEW_PRIVS=1` via ctypes/prctl.
    2. Submit an input that causes the parser to raise; verify `SandboxFailure(reason=child-raised:...)`; verify parent globals unaffected.
    3. Submit an input that hangs; verify killed at `timeout=0.5s`.
    4. Run `pytest internal/tests/security/test_parser_sandbox.py -v` (upstream).
  - PASS: 9 / 9. Linux-only assertion: child's `/proc/self/status` reports `NoNewPrivs: 1`. Upstream on station 2026-05-15: 9 / 9 ✅.
  - FAIL: a parser crash propagates to the parent, or a hang exceeds the timeout.

### FSR_FD_17 — Media-parser format restriction

- **Functional Security Requirement.** The media-parsing library SHALL be restricted to the relevant format (JPEG / PNG for `--synthetic-initial-rgb`; mp4 for reference clips); other formats SHALL be rejected at the format-dispatch layer, not deeper in a generic parser.
- **Risk Level.** Moderate.
- **Risk Response.** Remediate.
- **Responsibility.** SIL Engineering — Adapter input layer.
- **Priority Level.** P1.
- **Test / Measurement.**
  - SQA Test Steps:
    1. Submit a JPEG to the synthetic-initial-rgb dispatch → accepted.
    2. Submit an mp4 to the synthetic-initial-rgb dispatch → expect refusal at dispatch (parser MUST NOT be called even though it's registered — anti-bypass).
    3. Run `pytest internal/tests/security/test_format_dispatch.py -v` (upstream).
  - PASS: 9 / 9; full audit-context in exception. Upstream on station 2026-05-15: 9 / 9 ✅.
  - FAIL: a non-permitted format reaches the parser.

### FSR_FD_18 — Session state deallocation + GPU zero

- **Functional Security Requirement.** FlashDreams SHALL NOT persist inference request inputs (prompts, init frames, intermediate latents) in memory or on disk beyond the active session scope. On session close (WebRTC tear-down, gRPC `CloseSession`, CLI exit), all per-session tensors and buffers SHALL be deallocated and the GPU memory zeroed before another session is accepted.
- **Risk Level.** Moderate.
- **Risk Response.** Remediate.
- **Responsibility.** SIL Engineering — Pipeline.
- **Priority Level.** P0.
- **Test / Measurement.**
  - SQA Test Steps:
    1. Open session A; submit prompt + frames; close.
    2. Open session B; assert no residual bytes from A reach B (aliased-reference test).
    3. Attempt to reuse a non-empty session state object → expect `SessionLifecycleError`.
    4. Run `pytest internal/tests/security/test_session_state.py -v` (upstream).
  - PASS: 11 / 11. Bytearrays AND latent buffers zeroed in place BEFORE truncation; two-session round-trip carries 0 bytes. Upstream on station 2026-05-15: 11 / 11 ✅.
  - FAIL: any bytes from A observable in B.

### FSR_FD_19 — Per-session response routing

- **Functional Security Requirement.** FlashDreams SHALL restrict disclosure of inference responses (RTP video track, gRPC Step response, generated mp4) to the session that submitted the originating request. The single-active-session invariant in lingbot is the architectural enforcement; alpadreams gRPC SHALL enforce via session_id correlation.
- **Risk Level.** Moderate.
- **Risk Response.** Remediate.
- **Responsibility.** SIL Engineering — Serving.
- **Priority Level.** P0.
- **Test / Measurement.**
  - SQA Test Steps:
    1. Open multiple gRPC sessions; verify each session's sink receives only its own frames.
    2. Attempt to address a sink with another session's `session_id` → expect `UnknownSession`; verify `target_sink.received == []`.
    3. Close a session; attempt to route to it → expect refusal.
    4. Run `pytest internal/tests/security/test_session_router.py -v` (upstream).
  - PASS: 10 / 10. `new_session_id()` is high-entropy (1000 unique). Upstream on station 2026-05-15: 10 / 10 ✅.
  - FAIL: a frame for session A reaches session B's sink.

### FSR_FD_20 — Per-client rate limits

- **Functional Security Requirement.** FlashDreams server endpoints (LING-RTC signaling, ALPA-GRPC) SHALL apply per-client rate limits: ≤10 SDP offers / minute / source IP for lingbot; ≤60 Step calls / minute / session_id for ALPA-GRPC. Rate-limit violations SHALL log and drop, not crash. (Maps to NIM API4:2023 Unrestricted Resource Consumption.)
- **Risk Level.** Low.
- **Risk Response.** Remediate.
- **Responsibility.** SIL Engineering — Serving.
- **Priority Level.** P1.
- **Test / Measurement.**
  - SQA Test Steps:
    1. Burst 11 SDP offers / minute / source → 11th and beyond denied.
    2. Idle bucket for 5 minutes; verify capacity does not extend by idle (no accrual attack).
    3. Independent buckets per client key; clock-rollback does not credit tokens.
    4. Run `pytest internal/tests/security/test_rate_limiter.py -v` (upstream).
  - PASS: 13 / 13 on a token-bucket with injectable monotonic clock (hermetic, no `sleep`). Upstream on station 2026-05-15: 13 / 13 ✅.
  - FAIL: a burst over capacity is accepted.

### FSR_FD_21 — Recording mode 0600 + opt-in

- **Functional Security Requirement.** The alpadreams `session_recorder` SHALL write recording files (`.pt`, `.json`) with mode `0600` and to a path under the operator's home directory only. Recording SHALL be **off** by default; enabling requires an explicit `--record` flag whose use is logged.
- **Risk Level.** Low.
- **Risk Response.** Remediate.
- **Responsibility.** SIL Engineering — gRPC.
- **Priority Level.** P1.
- **Test / Measurement.**
  - SQA Test Steps:
    1. Start ALPA-GRPC without `--record`; submit `InitializeSession → Step → Close`; verify no `.pt` or `.json` on disk.
    2. Start with `--record /tmp/outside-home` → expect refusal.
    3. Start with `--record $HOME/sessions/`; submit a session → verify resulting files at mode `0o600`.
    4. Place an existing `.pt` with mode `0o666` at the target path; start with `--record`; verify recorder refuses to overwrite (TOCTOU defense).
    5. Run `pytest internal/tests/security/test_recording_perms.py -v` (upstream).
  - PASS: 8 / 8. Upstream on station 2026-05-15: 8 / 8 ✅.
  - FAIL: a recording file lands with wider mode than `0o600`.

### FSR_FD_22 — Trust-collapse docs (OE-1)

- **Functional Security Requirement.** Documentation of the FlashDreams TOE and its omni-dreams consumer SHALL flag the trust-collapse caused by `--network=host` + `--ipc=host` + Wayland-socket mount + SSH-agent forward (OE-1). The FlashDreams README + `docs/security/SECURITY.md` SHALL note that the FlashDreams `docker run` flow does NOT require those flags; the upstream omni-dreams interactive-drive README SHALL carry the trust-collapse callout where those flags are still documented.
- **Risk Level.** Moderate.
- **Risk Response.** Share.
- **Responsibility.** Documentation + SIL Engineering.
- **Priority Level.** P1.
- **Test / Measurement.**
  - SQA Test Steps:
    1. Confirm `docs/security/SECURITY.md` carries a "container-flag hygiene" section that names `--network=host`, `--ipc=host`, Wayland socket, SSH-agent forward and links to OE-1.
    2. Confirm `omni-dreams/samples/interactive-drive/README.md` carries the `[!IMPORTANT]` callout citing FSR_FD_22 / OE-1 (upstream invariant).
  - PASS: both doc sections present.
  - FAIL: callout missing in either location.

### FSR_FD_23 — Slurm IMEX banner + collision check

- **Functional Security Requirement.** The Slurm launch helper for `post-training` (omni-dreams consumer of FlashDreams) SHALL emit a banner reminding the operator of OE-2 (tenant-isolation assumption) and SHALL refuse to start if it detects an obviously shared IMEX channel name pattern.
- **Risk Level.** Moderate.
- **Risk Response.** Share.
- **Responsibility.** Cluster Operations + SIL Engineering (`omni-dreams` repo).
- **Priority Level.** P1.
- **Test / Measurement.**
  - SQA Test Steps:
    1. Launch with a unique channel name → accepted; banner references OE-2 + FSR_FD_23 + "cluster administrator".
    2. Launch with forbidden names (`default`, `imex`, `test`, `demo`, `shared`, `channel`, `ipp`, `pub`, `public`) → refused.
    3. Active-peer collision (case-insensitive) → refused.
    4. Run `pytest internal/tests/security/test_slurm_imex.py -v` (upstream).
  - PASS: 12 / 12. Upstream on station 2026-05-15: 12 / 12 ✅.
  - FAIL: a colliding channel name reaches the launcher.

### FSR_FD_24 — Continuous fuzz coverage

- **Functional Security Requirement.** Continuous fuzz coverage SHALL include the alpadreams protobuf decoders and any media-parse surface exposed by FlashDreams adapters. Findings shall be reported via PSIRT and fed back into FSR_FD_15 / FSR_FD_16 / FSR_FD_17 bounds.
- **Risk Level.** Low.
- **Risk Response.** Remediate.
- **Responsibility.** SIL Engineering — Pipeline.
- **Priority Level.** R2 (post-GA).
- **Test / Measurement.**
  - SQA Test Steps:
    1. Confirm OSS-Fuzz integration (or local AFL harness) targets the ALPA-GRPC proto decoders and any USDZ parser used by adapters.
    2. Track fuzz findings in PSIRT.
  - PASS: OSS-Fuzz target lands and runs continuously; ad-hoc AFL run on the proto / USDZ corpus before each release.
  - FAIL: a release ships without a fuzz run on these surfaces.

---

## NIM-template FSR slots 25–36 (mapping to FlashDreams TOE)

The user-supplied NIM TAVA template enumerates `FSR_<TOE>_01..36`. Slots 25–36 in that template cover NIM-API-specific concerns (mandatory AuthN/AuthZ, scaling, non-prod parity, request-scope memory, etc.). For the FlashDreams TOE, each maps onto an existing `FSR_FD_NN` row above or is **Avoided** by the architecture invariant ("no NVIDIA-side ingress / egress / storage of operator data"). Each row uses the same column structure; **Risk Response** is `Map` (mapped onto an existing FlashDreams FSR) or `Avoid` (the asset / surface does not exist in the FlashDreams TOE).

| Slot | NIM-template intent | FlashDreams mapping | Risk Level | Risk Response | Responsibility | Priority | Test / Measurement |
|---|---|---|---|---|---|---|---|
| **FSR_FD_25** | Operator (NIM customer in NIM template) SHALL provide AuthN/AuthZ to limit access to API endpoints to authorized and authenticated users / processes (NIM template FSR 24). | **Map → FSR_FD_12** (token interceptor). When the operator chooses public-bind (FSR_FD_10 opt-in), the token interceptor is mandatory. | Very High | Map | SIL Engineering — Serving + Operator | P0 (non-loopback) / P1 (loopback) | See FSR_FD_12 Test / Measurement above. |
| **FSR_FD_26** | Operator SHALL configure compute / storage / memory / network scaling to meet service SLA (NIM template FSR 25). | **Avoid** — FlashDreams has no managed availability SLA. Single-process / single-active-session by architecture. Operator owns scaling end-to-end. | Very Low | Avoid | Operator | P1 | Documented in SADD § 3.6.2 (High Availability) and § 3.6.3 (Scalability). |
| **FSR_FD_27** | Operator SHALL configure API calls to be transmitted using authenticated encryption (TLS 1.2+) to prevent disclosure / modification (NIM template FSR 26). | **Map → FSR_FD_01** (TLS) + **FSR_FD_12** (token). Mandatory on non-loopback. | Moderate | Map | Operator + SIL Engineering — Serving | P0 (non-loopback) / P1 (loopback) | See FSR_FD_01 and FSR_FD_12 Test / Measurement above. |
| **FSR_FD_28** | Non-production API deployments SHALL be secured similarly to production (NIM template FSR 27). | **Map → FSR_FD_01 + FSR_FD_10 + FSR_FD_12**. Documented in `SECURITY.md`: any FlashDreams listener that processes operator data is treated as production-grade regardless of label. | Very Low | Map | SIL Engineering | P1 | Doc review: confirm `SECURITY.md` carries the "non-prod parity" callout. |
| **FSR_FD_29** | Per-client rate-limits to prevent resource abuse and overconsumption (NIM template FSR 28; OWASP API4:2023). | **Map → FSR_FD_20.** | Very Low | Map | SIL Engineering — Serving | P1 | See FSR_FD_20 Test / Measurement above. |
| **FSR_FD_30** | SHALL NOT persist inference request data beyond request scope; SHALL validate memory used to process one user's request is freed before another user's request (NIM template FSR 29). | **Map → FSR_FD_18** (session state deallocation + GPU zero). | Moderate | Map | SIL Engineering — Pipeline | P0 | See FSR_FD_18 Test / Measurement above. |
| **FSR_FD_31** | SHALL logically isolate processing of inference requests from one user to another (NIM template FSR 30). | **Map → FSR_FD_19** (per-session response routing) + the single-active-session invariant in LING-RTC. | Low | Map | SIL Engineering — Serving | P0 | See FSR_FD_19 Test / Measurement above. |
| **FSR_FD_32** | SHALL restrict disclosure of responses to the user that submitted the request (NIM template FSR 31). | **Map → FSR_FD_19.** | Moderate | Map | SIL Engineering — Serving | P0 | See FSR_FD_19 Test / Measurement above. |
| **FSR_FD_33** | Front-end API server SHALL be configured with secure protocols and to open only needed ports; restrict access to those ports to authorized users / processes (NIM template FSR 32). | **Map → FSR_FD_01 + FSR_FD_10 + FSR_FD_12.** LING-RTC HTTP signaling on `:8089` (loopback default); ALPA-GRPC on a configurable port; no other listeners. | Moderate | Map | SIL Engineering — Serving | P0 | See FSR_FD_10, FSR_FD_01, FSR_FD_12 Test / Measurement above. |
| **FSR_FD_34** | Triton Inference Server (NIM) SHALL be configured with secure protocols and to open only needed ports; restrict access to those ports to authorized users / processes (NIM template FSR 33). | **Avoid** — FlashDreams does not run a Triton Inference Server in the TOE. The recipe pipelines call torch directly (FD-INFRA). When deployed downstream behind a Triton, that Triton is governed by the operator's serving stack, not by FlashDreams. | Moderate | Avoid | Operator | P0 (operator-managed if Triton is added downstream) | Doc note in `SECURITY.md`: "If you place FlashDreams behind Triton or another inference server, that server's hardening is outside the FlashDreams TOE." |
| **FSR_FD_35** | Access to Triton Inference Server resources SHALL be accessible only via the FlashDreams API (NIM template FSR 34). | **Avoid** — see FSR_FD_34. The flashdreams Pipeline is in-process; there is no separate Triton resource layer in the TOE. | Moderate | Avoid | Operator | P0 (operator-managed if Triton is added downstream) | Doc note in `SECURITY.md`. |
| **FSR_FD_36** | Media parsing library sandboxed (NIM template FSR 35) + restricted to relevant format (NIM template FSR 36). | **Map → FSR_FD_16 + FSR_FD_17.** | Moderate | Map | SIL Engineering — Adapter input layer | P1 (1.0) / P0 (post-GA) | See FSR_FD_16 and FSR_FD_17 Test / Measurement above. |

---

## Coverage summary

| Status | Count | FSRs |
|---|---|---|
| ✅ As-built contract+stub tests passing upstream (mirror to FlashDreams planned) | 20 | FSR_FD_01, _02, _03, _04, _06, _08, _09, _10, _11, _12, _13, _14, _15, _16, _17, _18, _19, _20, _21, _23 |
| 📄 Documented control (no test seam needed) | 1 | FSR_FD_22 (trust-collapse callout) |
| ⏳ Externally blocked or operator-side | 3 | FSR_FD_05 (operator guidance — FlashDreams ships no canonical image), FSR_FD_07 (Cosmos-Guardrail real backend — HF egress), FSR_FD_24 (OSS-Fuzz post-GA) |
| 🔁 Mapped onto an existing FSR_FD_NN | 9 | FSR_FD_25, _27, _28, _29, _30, _31, _32, _33, _36 |
| 🚫 Avoided by FlashDreams TOE architecture invariant | 3 | FSR_FD_26 (no managed SLA), FSR_FD_34, FSR_FD_35 (no Triton in TOE) |
| **Total slots covered** | **36** | (24 canonical + 12 NIM-template mapping rows) |

---

## Notes on the NIM-template mapping

The user-supplied template covered the NIM (NeMo Inference Microservices) TOE. FlashDreams differs from NIM in three architecturally load-bearing ways, which is why several NIM FSRs map rather than apply:

1. **No managed service SLA.** FlashDreams is a developer / researcher inference stack; the operator owns availability. NIM-template FSR 25 (compute scaling) becomes an operator-side concern (Avoid).
2. **No Triton in the TOE.** FlashDreams pipelines call torch directly via `FD-INFRA`. There is no separate Triton Inference Server inside the FlashDreams TOE, so NIM-template FSRs 33 / 34 are noted as "operator-managed if a Triton is added downstream" (Avoid).
3. **No NGC / NVAIE / API Catalog publishing of FlashDreams models.** FlashDreams publishes weights through HuggingFace (`nvidia/omni-dreams-*`) and S3 (`pdx.s8k.io`). Several NIM-template FSRs (e.g. NGC mandatory access controls, NGC encryption-at-rest, nSpect Helm-chart NSPECT-ID tagging, NIM-model byte-code malware scanning, NIM-model source-code OSS scanning) are addressed by NVIDIA-wide controls outside the FlashDreams TOE. These have been folded into the surviving `FSR_FD_04` (checkpoint integrity at the FlashDreams loader) and `FSR_FD_05` (operator-side container-image signing guidance) rather than enumerated separately.

4. **No canonical FlashDreams container image.** Per commit `ab74b58`, FlashDreams ships no pre-published image; operators build locally from `docker/Dockerfile` against `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04`. NIM-template FSRs that assumed a canonical NVIDIA-published image (e.g. "container hardened and scanned per the NVIDIA OCI Container Security Hardening SRD", NIM template FSR 20) become **operator-side responsibilities** for FlashDreams. The FlashDreams `docker/Dockerfile` follows reasonable hygiene defaults (minimal layers; explicit version pinning; no secrets in build args), and `docs/security/SECURITY.md` will surface this as operator guidance.

The intent of the mapping rows (`FSR_FD_25..36`) is to provide a clear "this NIM-template slot is covered by FlashDreams FSR X" or "this NIM-template slot does not apply to the FlashDreams TOE because Y", so that the Security PIC can sign off the TAVA against MVSB-32946 without having to re-derive the mapping at acknowledgment time.
