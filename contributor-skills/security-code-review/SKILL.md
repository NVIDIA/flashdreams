---
name: security-code-review
description: Review security-sensitive FlashDreams changes or an explicitly requested security review, including untrusted inputs, model loading, serving endpoints, credentials and CI permissions. Produce findings and evidence for human review.
---

# Security code review

Read [SECURITY.md](../../SECURITY.md) for disclosure and
[CONTRIBUTING.md](../../CONTRIBUTING.md#code-review-and-merge) for review policy.
Use the current `.github/CODEOWNERS` and branch rules for ownership and approval;
do not reproduce reviewer rosters or change repository settings as part of a review.

Trace the changed behavior through its callers and actual trust boundaries.
Focus on relevant paths: checkpoint/deserialization inputs, subprocess arguments,
paths and downloads, serving access, credential handling, and CI permissions.
Use existing tests and scans as evidence; inspect what they do not cover.

Report concrete findings with location, triggering conditions, impact and a
suggested repair. Separate demonstrated defects from optional hardening, and
state which checks ran. Follow the private disclosure route in SECURITY.md for
potential vulnerabilities. An agent's findings do not replace the maintainer
approval required by the contribution policy.
