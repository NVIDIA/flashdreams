<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# FlashDreams architecture, TAVA, and FSR documentation

This directory holds the FlashDreams-scoped architecture documents that
support the manual TAVA 2.0 acknowledgment path for
[MVSB-32946](https://jirasw.nvidia.com/browse/MVSB-32946).

| Document | Purpose |
| --- | --- |
| [`architecture.md`](architecture.md) | Four-view architecture (Static / Dynamic / Data / Deployment) for the FlashDreams TOE. |
| [`sadd.md`](sadd.md) | Software Architecture and Design Document per `SWE-PLC-L1-002-BasicPLC-SADD-TMPL`. |
| [`tava.md`](tava.md) | FlashDreams-scoped TAVA narrative (assets, threats, risk, security objectives, POA&M). |
| [`fsr_table.md`](fsr_table.md) | Canonical FSR sheet in the Excel-Based Simple TAVA Template column structure (FSR_FD_01..36). |

The agent-runnable workflow for this surface lives as a project skill at [`../../agentic/skills/flashdreams-security-architect/SKILL.md`](../../agentic/skills/flashdreams-security-architect/SKILL.md) — follow the `agentic/skills/README.md` opt-in instructions to symlink it into `.claude/skills/` or `.cursor/skills/`.

## Relationship to the upstream omni-dreams documents

The single source of truth across both repos is the upstream pair at
`omni-dreams/docs/arch/architecture.md` and
`omni-dreams/internal/docs/planning/tava_omni_dreams_flashdreams.md`.
This directory is the **FlashDreams-scoped subset** of that source, narrowed
to:

- `flashdreams/flashdreams/` (recipes, core, infra, plugins, scripts)
- `flashdreams/integrations/` (alpadreams gRPC, lingbot WebRTC, plus other adapters)
- The supply chain that loads these (HuggingFace, S3, GHCR)

The `omni-dreams/samples/interactive-drive` consumer and the
`omni-dreams/post-training` trainer are out of scope here — they are
covered in the upstream omni-dreams + flashdreams documents.

## Where the as-built tests live

FSR contract+stub tests landed upstream at
`omni-dreams/internal/tests/security/`. As of the upstream station check
on 2026-05-15, 20 of 24 FSRs have passing contract+stub tests
(230 / 230 passing in 1.86 s). The FlashDreams-side mirror at
`flashdreams/tests/security/` is queued for a follow-up MR — see
[`tava.md`](tava.md) § Step 6, Q-06.
