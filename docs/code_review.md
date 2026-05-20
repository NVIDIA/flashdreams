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
repository, *how* code review works in practice, and *what* a
contributor or reviewer can expect day-to-day.

It is a companion to [`CONTRIBUTING.md`](../CONTRIBUTING.md), not a
replacement. If you have not read CONTRIBUTING yet, start there for
the fork-and-PR mechanics, DCO sign-off, testing markers, and the
SPDX header convention. This document picks up where CONTRIBUTING's
[*Code review and merge*](../CONTRIBUTING.md#code-review-and-merge)
section leaves off.

## At a glance

- Every PR to `main` requires approval from a **code owner** for the
  touched paths. The code owners for each area of the repository are
  listed in [`/.github/CODEOWNERS`](../.github/CODEOWNERS).
- A single approval from a code owner is the bar. GitHub treats the
  teams listed on a CODEOWNERS line as alternatives — *any one* of
  them can approve.
- Code owners run the [security review checklist](#security-review-checklist)
  at the bottom of this document on every review.
- Outside contributors are welcome everywhere. CODEOWNERS controls
  who must *approve* a change, not who may *open* one.
- The review requirement cannot be bypassed; any operational
  exception requires a documented audit trail.

## How review and merge work on GitHub

There is no special button for code-owner approval — it's the
standard GitHub PR flow:

1. **A contributor opens a pull request.** GitHub reads
   [`.github/CODEOWNERS`](../.github/CODEOWNERS), computes which
   teams own the paths touched, and adds them to the *Reviewers*
   sidebar under "Reviewers requested by Code owners".
2. **A code owner clicks "Review changes → Approve"** in the GitHub
   UI (or runs `gh pr review --approve` from the CLI). That single
   approval, from a member of a CODEOWNERS team for the touched
   paths, satisfies the "Require review from Code Owners" branch
   protection check.
3. **CI and DCO must also be green** — formatting, type-checks, CI
   tier markers per [`CONTRIBUTING.md` → Testing](../CONTRIBUTING.md#testing),
   and a `Signed-off-by` trailer on every commit per
   [`CONTRIBUTING.md` → DCO](../CONTRIBUTING.md#developer-certificate-of-origin-dco).
4. **A maintainer squash-merges.** The PR title and description
   become the squash commit message.

A non-CODEOWNER approval is welcome — review feedback from anyone
helps — but on its own it does not satisfy the protected-branch
check.

Branch protection on `main` is configured so that:

- **Require pull request reviews before merging** — enabled.
- **Require review from Code Owners** — enabled.
- **Dismiss stale approvals when new commits are pushed** — enabled.
- **Restrict who can push to matching branches** — enabled; direct
  pushes are not allowed.
- **Do not allow bypassing the above settings** — enabled, including
  for administrators.

## How ownership is organised

GitHub honours exactly one CODEOWNERS file per repository, so all of
the rules live in [`/.github/CODEOWNERS`](../.github/CODEOWNERS). The
file is divided into sections by subsystem, with each section
behaving like its own delegated CODEOWNERS for the subtree it
covers. The table below mirrors that structure so you can find the
right reviewer without reading the whole file.

### Subsystems

We keep ownership intentionally lean — three teams cover the whole
repository.

| Area | Paths | Code owners |
|------|-------|-------------|
| **Default** (core, infra, recipes, plugins, configs, tests, docs, CI, container) | everything not listed below | `@NVIDIA/flashdreams-maintainers` |
| **Integrations** | `integrations/` (all recipes) | `@NVIDIA/flashdreams-integrations` *or* `@NVIDIA/flashdreams-maintainers` |
| **Security-sensitive files** | `LICENSE`, `reuse.toml`, `uv.lock`, `flashdreams/pyproject.toml` | `@NVIDIA/flashdreams-security` (sole owner) |
| **Licensing & dependency manifests with security review** | `LICENSES/`, `NOTICE`, root `pyproject.toml` | `@NVIDIA/flashdreams-maintainers` *or* `@NVIDIA/flashdreams-security` |
| **Per-integration packaging** | `integrations/*/pyproject.toml` | `@NVIDIA/flashdreams-integrations` *or* `@NVIDIA/flashdreams-security` |
| **Review-policy files** | `.github/CODEOWNERS`, `docs/code_review.md` | `@NVIDIA/flashdreams-maintainers` *or* `@NVIDIA/flashdreams-security` |

`@NVIDIA/flashdreams-maintainers` is the default owner for any path
not explicitly matched. We can split out additional teams later if
any subsystem grows enough to deserve its own ownership boundary,
but the current shape keeps the review surface small and easy to
staff.

**A note on "or" in the table above.** When CODEOWNERS lists
multiple owners on the same line, GitHub treats them as
alternatives — approval from *any one* of the listed teams
satisfies the required-review-from-Code-Owners check. If we ever
need a path to require sign-off from *both* the subsystem team
*and* the security team (rather than either), we will raise the
branch-protection "required approvals" count and split the rule
across two lines.

### Reading the file

CODEOWNERS rules are applied in order, with the **last matching
line winning**. A more specific path overrides a more general one.
If you are unsure who owns a file:

```bash
# Validate the CODEOWNERS file itself
gh api repos/NVIDIA/flashdreams/codeowners/errors

# Inspect recent reviewers for a path
git log -1 --format=%H -- <path>
```

When in doubt, open the PR and request a review from
`@NVIDIA/flashdreams-maintainers`; they will route it to the right
person.

## Security-sensitive paths

A small set of paths is owned by `@NVIDIA/flashdreams-security`,
either as sole owner or as an additional approver:

- `LICENSE`, `LICENSES/`, `NOTICE`, `reuse.toml` — Apache-2.0
  attribution and third-party license bookkeeping.
- `pyproject.toml`, `flashdreams/pyproject.toml`,
  `integrations/*/pyproject.toml`, `uv.lock` — dependency manifests
  and lock files.
- `.github/CODEOWNERS` and `docs/code_review.md` — the review
  policy itself.

Mistakes in these files are unusually expensive to roll back, so the
security team signs off on them as a matter of routine.

For **vulnerability reports**, do *not* open a public issue or PR.
Follow NVIDIA's coordinated disclosure process at
<https://www.nvidia.com/en-us/security/>, as described in
[`CONTRIBUTING.md` → Filing issues and security reports](../CONTRIBUTING.md#filing-issues-and-security-reports).

## What this looks like from a contributor's perspective

Here's what to expect when you open a pull request — the
mechanics are straightforward, and you'll see each step on the
PR page as it happens:

1. Fork, branch, code, sign off, push, open a PR — same flow as in
   [`CONTRIBUTING.md` → Submitting a pull request](../CONTRIBUTING.md#submitting-a-pull-request).
2. GitHub automatically tags the right code owners based on the
   paths you touched.
3. A code owner works through the checklist below and approves.
4. The squash-merge button lights up, and a maintainer presses it.

If your PR spans multiple subsystems (say, a core change that also
updates an integration), you will see review requests from each
affected CODEOWNERS team. That is intentional: cross-subsystem
changes benefit from a reviewer in each area.

We aim for first review within two business days; if a PR is
quieter than that, please leave a friendly ping comment.

## Becoming a code owner

CODEOWNERS membership is how the project formalises long-term
ownership of a subsystem, and it's the main mechanism by which
governance opens up over time (see
[`CONTRIBUTING.md` → Project governance](../CONTRIBUTING.md#project-governance)).
Contributors — NVIDIA employee or not — who consistently land
high-quality work in an area, participate in reviews, and engage
with the issue tracker can be invited to become code owners.
CODEOWNERS supports two ways to be listed: team references such as
`@NVIDIA/flashdreams-integrations`, and individual GitHub handles
such as `@username`. The `@NVIDIA/*` teams require NVIDIA
organization membership, so contributors outside NVIDIA are
recognised by adding their individual handle to the relevant
CODEOWNERS rule — they still count as required reviewers for the
paths they own.

There is no fixed time bar; we look for sustained good judgment
about when to ship, when to push back, and when to ask for help. If
you'd like to grow into that role, please say so in a Discussion or
to any maintainer — we'd rather hear it than not.

## No bypass, no waiver

Branch protection on `main` is configured so that:

- Administrators are *not* allowed to bypass review or status checks.
- Direct pushes to `main` are blocked.
- Force-pushes to `main` are blocked.

If an operational situation genuinely requires an exception — for
example, an emergency security patch where the usual reviewer is
unavailable — the exception must be documented as an issue and the
audit trail of the incident retained. We have not needed to use
this path, and we would prefer to keep it that way.

## Security review checklist

Code owners — human or LLM agent — run this checklist on every PR
before approving. It covers the most common ways a change can
introduce a security regression.

**How to use the table.** For each item, set **Status** to one of:

- `Pass` — the change clearly satisfies the item.
- `Fail` — the change clearly violates the item; block until fixed.
- `N/A` — the item does not apply to this PR (e.g. crypto items on
  a pure-docs change). Explain why in the *Evidence / finding*
  column.
- `Human review` — the item requires judgment beyond what an LLM
  agent should make alone; flag it for a human code owner.

**Evidence / finding** should cite the relevant `path/to/file.py:LN`
or quote the specific lines or commit hash that justify the status.
For `Pass` on a non-trivial item, briefly say *why* (one short
sentence) rather than just `Pass`. LLM agents should err toward
`Human review` whenever they are uncertain, and should never mark
an item `Pass` without concrete evidence.

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

## NVIDIA PLC compliance note

FlashDreams is released alongside an NVIDIA Product Lifecycle (PLC)
"+2 Code Review" compliance requirement. The CODEOWNERS-based
review described above — one approval from a qualified code owner,
checklist applied, security team on sensitive paths, no bypass —
is what satisfies that requirement. Code owners are trained as
required by NVIDIA PLC before being added to a CODEOWNERS team, and
the checklist questions above are reused from NVIDIA's *PLC: Code
Review Excellence* training. No separate, parallel process is
needed.

## Changing this document or the CODEOWNERS file

Both this file and `.github/CODEOWNERS` are themselves owned by
`@NVIDIA/flashdreams-maintainers` and `@NVIDIA/flashdreams-security`.
If you want to propose a change — new subsystem owners, a different
ownership boundary, a refinement to how review works — please open
an issue or Discussion first so we can talk through it before the
PR. The goal is for the review policy to be predictable; we change
it deliberately rather than incrementally.

---

Thanks for reading this far. The project is healthier when
reviewers and contributors share a clear picture of how decisions
get made; if anything above is unclear, that's a bug in the
documentation and we'd appreciate the issue.
