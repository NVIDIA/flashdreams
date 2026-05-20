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

# Code review and ownership

Welcome — and thank you for sending a pull request to FlashDreams.
This document explains *who* approves changes in each part of the
repository, *what* the project's "+2 Code Review" gate actually is,
and *how* the policy fits alongside the project's broader contributor
guidance.

It is a companion to [`CONTRIBUTING.md`](../CONTRIBUTING.md), not a
replacement. If you have not read CONTRIBUTING yet, start there for
the fork-and-PR mechanics, DCO sign-off, testing markers, and the
SPDX header convention. This document picks up where CONTRIBUTING's
[*Code review and merge*](../CONTRIBUTING.md#code-review-and-merge)
section leaves off.

## At a glance

- Every PR to `main` requires approval from a **qualified "+2"
  reviewer** for the touched paths. "+2" describes the reviewer's
  qualification level, not a head-count — one qualified +2 approval
  is the bar, not two separate +1 approvals.
- The +2 reviewers for each subsystem are listed in
  [`/.github/CODEOWNERS`](../.github/CODEOWNERS). The sections below
  are a human-readable map of the same file.
- Reviewers apply the [+2 Security Review Checklist](#2-security-review-checklist)
  at the bottom of this document on every review.
- For security-sensitive paths, the project's Security PIC (or
  delegate from `@NVIDIA/flashdreams-security`) signs off in addition
  to the subsystem CODEOWNER.
- Outside contributors are welcome everywhere. CODEOWNERS controls
  who must *approve* a change, not who may *open* one.
- The +2 gate cannot be waived. Any operational exception requires a
  documented audit trail approved by the Security PIC.

## What "+2 Code Review" means here

In some review systems "+2" means two separate +1 approvals. In the
FlashDreams policy it means a **single approval from a reviewer
qualified at the +2 level**:

1. **Qualified reviewer.** CODEOWNERS membership is the project's
   ledger of qualified +2 reviewers. Reviewers are trained as required
   by NVIDIA PLC before being added.
2. **Checklist on every review.** Qualified reviewers run the
   [+2 Security Review Checklist](#2-security-review-checklist) below
   on every PR — it's a short, stable list, not a fresh checklist per
   PR.
3. **Authenticated identity.** Approvals are tied to authenticated
   GitHub accounts; shared accounts are not permitted.

Mechanically on GitHub, the gate is implemented as branch protection
on `main`:

- **Require pull request reviews before merging** — enabled.
- **Require review from Code Owners** — enabled. (This is the +2
  enforcement: only a qualified reviewer can satisfy it.)
- **Dismiss stale approvals when new commits are pushed** — enabled.
- **Restrict who can push to matching branches** — enabled; direct
  pushes are not allowed.
- **Do not allow bypassing the above settings** — enabled, including
  for administrators.

The CI checks documented in
[`CONTRIBUTING.md` → Testing](../CONTRIBUTING.md#testing) must also
be green, and every commit must carry a DCO sign-off per
[`CONTRIBUTING.md` → DCO](../CONTRIBUTING.md#developer-certificate-of-origin-dco).
Together these four — qualified +2 approval, green CI, DCO sign-off,
and no protected-branch bypass — form the PLC release-readiness gate
for FlashDreams.

## Why this matters

The +2 review gate exists for four reasons, in order of practical
impact:

- **Correctness.** A second pair of qualified eyes catches the
  mistakes tests don't: a misplaced `assert`, a config default that
  changes behaviour on a different GPU, a logging line that leaks
  something it shouldn't.
- **Security.** The standing checklist makes common vulnerability
  classes harder to ship by accident.
- **Shared ownership.** Each subsystem has named maintainers who
  understand its trade-offs, so changes don't fall into review limbo
  and aren't merged by people unfamiliar with the area.
- **Auditability.** Authenticated approvals on a protected branch
  give the project a clean audit trail, which matters for shipping
  FlashDreams as a supported NVIDIA product alongside the public OSS
  release.

We have kept the policy simple enough that it does not get in a
community contributor's way: contributors open PRs as normal, and the
qualified reviewer comes from CODEOWNERS automatically.

## How ownership is organised

GitHub honours exactly one CODEOWNERS file per repository, so all of
the rules live in [`/.github/CODEOWNERS`](../.github/CODEOWNERS). To
keep that file readable, it is divided into **sections by subsystem**,
with each section behaving like its own delegated CODEOWNERS for the
subtree it covers. The table below mirrors that structure so you can
find the right reviewer without reading the whole file.

### Subsystems

| Area | Paths | Qualified +2 reviewers |
|------|-------|------------------------|
| **Core library** | `flashdreams/flashdreams/core/`, `plugins/` | `@NVIDIA/flashdreams-core` |
| **Infra** | `flashdreams/flashdreams/infra/` | `@NVIDIA/flashdreams-infra` |
| **Recipes & configs** | `flashdreams/flashdreams/recipes/`, `configs/` | `@NVIDIA/flashdreams-recipes` |
| **Tests & pytest plugins** | `flashdreams/tests/`, `_pytest_plugins/` | `@NVIDIA/flashdreams-maintainers` |
| **Integrations** | `integrations/<recipe>/` | `@NVIDIA/flashdreams-integrations` + per-recipe owners |
| **Documentation** | `docs/` | `@NVIDIA/flashdreams-docs` |
| **Container & CI** | `docker/`, `.github/` | `@NVIDIA/flashdreams-ci` |
| **Workspace metadata** | `pyproject.toml`, `uv.lock`, per-package `pyproject.toml` | `@NVIDIA/flashdreams-maintainers` |
| **Licensing & security** | `LICENSE`, `LICENSES/`, `NOTICE`, `reuse.toml`, `CODEOWNERS` itself, `docs/code_review.md` | `@NVIDIA/flashdreams-maintainers` + `@NVIDIA/flashdreams-security` |
| **Fallback (anything else)** | `*` | `@NVIDIA/flashdreams-maintainers` |

Each integration under `integrations/` additionally has a per-recipe
owners team (for example, `@NVIDIA/flashdreams-cosmos-owners` for
`integrations/cosmos_predict2/`). The integrations team is added as a
backstop so no recipe is orphaned if its lead is unavailable.

### Reading the file

CODEOWNERS rules are applied in order, with the **last matching line
winning**. A more specific path overrides a more general one — for
instance, `integrations/cosmos_predict2/` overrides the catch-all
`integrations/` line above it. If you are unsure who owns a file:

```bash
# Validate the CODEOWNERS file itself
gh api repos/NVIDIA/flashdreams/codeowners/errors

# Inspect recent reviewers for a path
git log -1 --format=%H -- <path>
```

When in doubt, open the PR and request a review from
`@NVIDIA/flashdreams-maintainers`; they will route it to the right
qualified reviewer.

## Security-sensitive paths

A few paths require an additional approval from
`@NVIDIA/flashdreams-security` on top of the subsystem CODEOWNER:

- `LICENSE`, `LICENSES/`, `NOTICE`, `reuse.toml` — Apache-2.0
  attribution and third-party license bookkeeping.
- `.github/CODEOWNERS` itself — changes to who can approve what.
- `docs/code_review.md` (this document) — changes to the review
  policy.

Mistakes in those files are unusually expensive to roll back, so the
project's Security PIC (or a delegate from the security team) signs
off on them as a matter of routine.

For **vulnerability reports**, do *not* open a public issue or PR.
Follow NVIDIA's coordinated disclosure process at
<https://www.nvidia.com/en-us/security/>, as described in
[`CONTRIBUTING.md` → Filing issues and security reports](../CONTRIBUTING.md#filing-issues-and-security-reports).

## What this looks like from a contributor's perspective

You don't need to think about most of this when opening a PR:

1. Fork, branch, code, sign off, push, open a PR — same flow as in
   [`CONTRIBUTING.md` → Submitting a pull request](../CONTRIBUTING.md#submitting-a-pull-request).
2. GitHub automatically tags the right CODEOWNERS based on the paths
   you touched. They appear in the PR sidebar under "Reviewers
   requested by Code owners".
3. A qualified +2 reviewer works through the checklist below and
   approves; if you touched a security-sensitive path above, a
   security reviewer also approves.
4. The squash-merge button lights up, and a maintainer presses it.

If your PR spans multiple subsystems (say, a core change that also
updates an integration), you will see review requests from each
affected CODEOWNERS team. That is intentional: cross-subsystem
changes benefit from a qualified reviewer in each area.

We aim for first review within two business days; if a PR is quieter
than that, please leave a friendly ping comment.

## Becoming a +2 reviewer

CODEOWNERS membership is how the project formalises long-term
ownership of a subsystem, and it's the main mechanism by which
governance opens up over time
(see [`CONTRIBUTING.md` → Project governance](../CONTRIBUTING.md#project-governance)).
Contributors — NVIDIA employee or not — who consistently land
high-quality work in an area, participate in reviews, and engage with
the issue tracker can be invited onto the relevant CODEOWNERS team
once any required NVIDIA PLC training is complete.

There is no fixed time bar; we look for sustained good judgment about
when to ship, when to push back, and when to ask for help. If you'd
like to grow into that role, please say so in a Discussion or to any
maintainer — we'd rather hear it than not.

## No bypass, no waiver

The +2 gate is not waivable. Branch protection on `main` is
configured so that:

- Administrators are *not* allowed to bypass review or status checks.
- Direct pushes to `main` are blocked.
- Force-pushes to `main` are blocked.

If an operational situation genuinely requires an exception — for
example, an emergency security patch where the usual reviewer is
unavailable — the exception must be documented as an issue or
incident record, approved by the Security PIC in writing, and the
audit trail retained. We have not needed to use this path, and we
would prefer to keep it that way.

## +2 Security Review Checklist

Reviewers — human or LLM agent — run this checklist on every PR
before approving. It is the standing checklist from NVIDIA's *PLC:
Code Review Excellence* training; the items below are the literal
questions a +2 reviewer is expected to answer.

**How to use the table.** For each item, set **Status** to one of:

- `Pass` — the change clearly satisfies the item.
- `Fail` — the change clearly violates the item; block until fixed.
- `N/A` — the item does not apply to this PR (e.g. crypto items on a
  pure-docs change). Explain why in the *Evidence / finding* column.
- `Human review` — the item requires judgment beyond what an LLM
  agent should make alone; flag it for the human +2 reviewer.

**Evidence / finding** should cite the relevant `path/to/file.py:LN`
or quote the specific lines or commit hash that justify the status.
For `Pass` on a non-trivial item, briefly say *why* (one short
sentence) rather than just `Pass`. LLM agents should err toward
`Human review` whenever they are uncertain, and should never mark an
item `Pass` without concrete evidence.

| # | Category | Item | Question | Status | Evidence / finding |
|---|----------|------|----------|--------|--------------------|
| 1 | Process & Static Analysis | SCA Clean | Have all critical and high-severity issues identified by Static Code Analysis (SCA) tools been resolved? | | |
| 2 | Process & Static Analysis | Risk Assessed | Has the code been evaluated against the product's Threat and Vulnerability Analysis (TAVA)? | | |
| 3 | Input Validation | Trust Boundaries | Is all input crossing the trust boundary strictly validated? | | |
| 4 | Input Validation | Length & Range | Are the lengths and ranges of the data explicitly checked? | | |
| 5 | Input Validation | Failures | Does any input validation failure result in an error? | | |
| 6 | Safe Function Usage | C/C++ APIs | Have unsafe functions been replaced with secure alternatives (e.g. `snprintf` vs `sprintf`, `strlcpy` vs `strcpy`, `fgets` vs `gets`)? | | |
| 7 | Safe Function Usage | Python APIs | Are unsafe executions avoided (e.g. `literal_eval` vs `eval`, `shell=False` in `subprocess`)? | | |
| 8 | Variable Management | Initialization | Are all variables initialized before use with a deny-by-default approach? | | |
| 9 | Variable Management | Typing | Are unsigned types used unless negative values are required? | | |
| 10 | Variable Management | Scope | Are variables scoped minimally and not reused improperly? | | |
| 11 | Compilation & Resilience | Compiler Flags | Are strict security compiler flags enabled (e.g. `-Werror`, `-Wall`, `-fstack-protector-strong`)? | | |
| 12 | Compilation & Resilience | Fault Injection (if applicable) | Are mitigations in place for critical low-level code (e.g. redundancy, glitch resistance)? | | |
| 13 | Error Handling | Action Taken | Are errors properly handled and propagated (logging alone is insufficient)? | | |
| 14 | Error Handling | No Leakage | Do error flows prevent exposure of sensitive data (stack traces, memory, secrets)? | | |
| 15 | Error Handling | Resource Cleanup | Are resources properly freed during error handling? | | |
| 16 | Cryptography & Secrets | No Hardcoded Secrets | Is the code free of plaintext secrets (API keys, passwords, tokens, internal IPs)? | | |
| 17 | Cryptography & Secrets | Strong Crypto | Are only modern, vetted cryptographic algorithms used (e.g. CNSA Suite 2.0)? | | |
| 18 | Cryptography & Secrets | Authentication First | Is data authenticated before decryption? | | |
| 19 | Cryptography & Secrets | Safe Randomness | Are cryptographically secure RNGs used (e.g. `/dev/urandom` vs `rand()`)? | | |
| 20 | Access Control & Concurrency | Least Privilege | Are access controls enforced using allowlists vs blocklists? | | |
| 21 | Access Control & Concurrency | Shared Resources | Are shared resources protected against DoS or privilege abuse? | | |
| 22 | Access Control & Concurrency | Race Conditions | Is the code protected against TOCTOU vulnerabilities? | | |
| 23 | Access Control & Concurrency | No Backdoors | Are there no intentional or accidental bypass mechanisms? | | |

The wording of the items above is reproduced from NVIDIA's *PLC:
Code Review Excellence* training; it is the authoritative version,
and changes to it should track the upstream training rather than
diverging here.

## Changing this document or the CODEOWNERS file

Both this file and `.github/CODEOWNERS` are themselves owned by
`@NVIDIA/flashdreams-maintainers` and `@NVIDIA/flashdreams-security`.
If you want to propose a change — new subsystem owners, a different
ownership boundary, a refinement to the +2 policy — please open an
issue or Discussion first so we can talk through it before the PR.
The goal is for the review policy to be predictable; we change it
deliberately rather than incrementally.

---

Thanks for reading this far. The project is healthier when reviewers
and contributors share a clear picture of how decisions get made; if
anything above is unclear, that's a bug in the documentation and
we'd appreciate the issue.
