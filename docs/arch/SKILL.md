<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

---
name: flashdreams-security-architect
description: Author / review / verify the FlashDreams TAVA + SADD + FSR test surface. Compact recipe scoped to the flashdreams repo (recipes, core, infra, integrations/{alpadreams,lingbot,…}) — companion to the broader omni-dreams + flashdreams skill at omni-dreams/docs/arch/SKILL.md.
---

# FlashDreams — security-architect skill (compact)

Agent-runnable recipe for security-architect work on the **flashdreams** repo. Use when you need to:

- Update one of the four architecture views (static / dynamic / data / deployment) after a code change.
- Cross-reference a TAVA threat (`T-<NAME>-N`) or FSR (`FSR_FD_NN`) to the relevant view section.
- Add a new FSR contract seam under `flashdreams/tests/security/` (planned) and wire it into the canonical FSR sheet here.
- Form a defensible opinion on the design before signing off the TAVA against MVSB-32946.

This skill is the FlashDreams-scoped subset of the broader omni-dreams + flashdreams skill at `omni-dreams/docs/arch/SKILL.md` and its reviewer companion `omni-dreams/docs/arch/review/SKILL.md`. The broader skill remains the source of truth when a change crosses the omni-dreams / flashdreams boundary (e.g. the interactive-drive sample, the post-training stack).

## TOE shape (one-paragraph mental model)

The FlashDreams TOE is the runtime process tree that loads a recipe (`cosmos`, `taehv`, `template`, `wan`, plus integration-registered slugs such as `wan21-*`, `lingbot_world`, `alpadreams`) into memory via `flashdreams-run`, fetches weights from HuggingFace or S3 through `FD-CKPT`, builds a `Pipeline = DiT + encoder + VAE/TAE`, and either runs to completion (offline generation) or enters a serving loop (lingbot WebRTC `:8089`; alpadreams gRPC; profiling server). FlashDreams ships **no canonical pre-published container image**: operators build locally from `docker/Dockerfile` against `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` (OE-7 narrows the supply-chain threat correspondingly). There is **no NVIDIA-side ingress, egress, or storage of operator data** — this is the central architectural invariant the TAVA preserves.

## Inputs the skill assumes are already in place

- [`architecture.md`](architecture.md) — four-view architecture for the FlashDreams TOE.
- [`sadd.md`](sadd.md) — Software Architecture and Design Document per `SWE-PLC-L1-002-BasicPLC-SADD-TMPL`.
- [`tava.md`](tava.md) — FlashDreams-scoped TAVA 2.0 (MVSB-32946 subset).
- [`fsr_table.md`](fsr_table.md) — canonical FSR sheet in the **Excel-Based Simple TAVA Template** column structure.
- Security test seams (planned): `flashdreams/tests/security/` (mirrors the omni-dreams `internal/tests/security/` package one-to-one).

## Quick workflows

### 1. Update an architecture view to reflect a code change

```bash
# Find which view section to edit
grep -nE '^##|^###' docs/arch/architecture.md | head -20

# After editing, propagate any renamed identifiers to:
#   docs/arch/architecture.md § "Naming index"
#   docs/arch/sadd.md § 3.2.2 (Interface table)
#   docs/arch/tava.md § 1.6 (Asset table) + § 3.2 (FSR responsibility column)
#   docs/arch/fsr_table.md (canonical sheet)
```

### 2. Add a new FSR contract seam (planned home: `flashdreams/tests/security/`)

```bash
# 1. Define the Protocol / stub in flashdreams/tests/security/<topic>.py
#    (mirror the structure used in omni-dreams/internal/tests/security/<topic>.py)
# 2. Write positive + negative tests in flashdreams/tests/security/test_<topic>.py
# 3. Run the suite (CPU-only, no GPU weights):
uv run pytest flashdreams/tests/security/ -v
# 4. Add a row to fsr_table.md keyed by FSR_FD_NN, then update tava.md § 3.2.
# 5. Commit with subject: "test(security): FSR_FD_NN <topic>"
```

### 3. Sanity-check mermaid syntax before committing

Mermaid 10.7.0 (the user's preview) is strict. Quote labels containing any of `< > / : { } " — · …`. The most common pitfalls:

| Hazard | Fix |
|---|---|
| `<digit` in a label | Replace with `under`, `≤`, or spell-out |
| `[/text]` without closing `/` | Quote as `Node["/text"]` |
| `[/path/with/slashes/]` | Use `Node["path with slashes"]` |
| `{a,b,c}` in a class member or edge label | Plain text without braces |
| `--` em-dash in subgraph label | Quote: `subgraph X["label — with dash"]` |

Quick scan for hazards inside the mermaid code blocks of `architecture.md`:

```bash
awk '/^```mermaid/{in_mm=1} /^```$/{in_mm=0} in_mm' docs/arch/architecture.md \
  | grep -nE '<[0-9]|\[/[^]"]*[^/"]\]|^\s*[A-Za-z_]+\.[A-Za-z_]+\s*-->'
```

## Cross-reference invariants (the **4-C** self-review)

When you change anything in `architecture.md`, check the following five places stay consistent:

1. **Consistent** — component names, interface names, trust-boundary labels match across all four views, the SADD interface table, and the TAVA asset table.
2. **Correct** — diagrams reflect code as of HEAD; cite file paths inline so reviewers can verify.
3. **Complete** — all TAVA v3 required sections present (Intro, TOE, Architecture Diagram, DFD, Security Assets, Attacker Model, Threats, Risk Analysis, Security Objectives, FSRs, POA&M).
4. **Concise** — no prose where a diagram suffices; no diagram where a table suffices.

(Fifth implicit C: **Codified** — every H-or-higher threat has a P0 FSR with a planned product-side wiring OR an as-built contract+stub test.)

## Operating-environment assumptions (do not silently weaken)

Documented in [`tava.md`](tava.md) § 1.3:

- **OE-1** — operator-trusted hardware; the Docker `--network=host --ipc=host` invocation collapses container/host trust.
- **OE-2** — Slurm cluster admin enforces tenant isolation (out of scope for flashdreams TOE; relevant for the omni-dreams post-training consumer).
- **OE-3** — operator credentials least-privilege + revocable + not in git.
- **OE-4** — host kernel + driver patched.
- **OE-5** — Cosmos-Guardrail weights from `nvidia/Cosmos-1.0-Guardrail` or NVIDIA mirror.
- **OE-6** — operator network may block `huggingface.co`; weights pre-stage required.
- **OE-7** — FlashDreams ships **no canonical pre-published container image**: operators build locally from `docker/Dockerfile` against `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` (per commit `ab74b58`). Any operator-built image lives under the operator's own registry / access controls; supply-chain trust delegates upward to the upstream `nvidia/cuda` base image.

## Where things live (FlashDreams scope)

| Concern | Where |
|---|---|
| Architecture diagrams (four views) | [`architecture.md`](architecture.md) |
| SADD (PLC-L1 template) | [`sadd.md`](sadd.md) |
| TAVA report (FlashDreams subset of MVSB-32946) | [`tava.md`](tava.md) |
| Canonical FSR sheet (Excel-style columns) | [`fsr_table.md`](fsr_table.md) |
| Upstream skill (omni-dreams + flashdreams) | `../../../omni-dreams/docs/arch/SKILL.md` |
| Upstream review skill | `../../../omni-dreams/docs/arch/review/SKILL.md` |
| Upstream canonical TAVA + asset inventory | `../../../omni-dreams/internal/docs/planning/tava_omni_dreams_flashdreams.md` |
| Upstream FSR contract+stub tests | `../../../omni-dreams/internal/tests/security/` |
| Planned FlashDreams security tests | `flashdreams/tests/security/` (mirror, to be landed) |

## Do not

- Do not edit `architecture.md` and forget to update the cross-reference sites above.
- Do not weaken the architectural invariants asserted in `internal/tests/security/*.py` upstream (e.g. "no inference SHALL run on a rejected verdict"; "untrusted factory MUST NOT be called during refusal"; "loader_fn SHALL NOT be invoked on checksum mismatch") when you mirror them here.
- Do not approve a control without first articulating the threat it closes (`tava.md` § 2.3).
- Do not treat OE-1..OE-7 as code-enforced — they are delegations to operator / cluster admin / upstream registry.
- Do not fabricate test counts. Test/Measurement cells in [`fsr_table.md`](fsr_table.md) cite the as-built upstream test file when applicable; if no test exists for an FSR yet, say so explicitly.
