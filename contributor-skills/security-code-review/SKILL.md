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

## Security review checklist

A +2 code review (the final, accountable approval; on GitHub, the approving
maintainer or CODEOWNER review) follows this checklist.
Use this checklist actively; record evidence or a finding rather than silently
assuming an item passed. Mark an item not applicable only with a short reason.

### Process and static analysis

- [ ] **SCA clean:** all critical and high-severity findings identified by
  static code analysis have been resolved.

### Input validation

- [ ] **Trust boundaries:** every input crossing a trust boundary is strictly
  validated.
- [ ] **Length and range:** data lengths and ranges are explicitly checked.
- [ ] **Failures:** input-validation failures return an error.

### Safe function usage

- [ ] **C/C++ APIs:** unsafe functions are replaced with secure alternatives,
  such as snprintf rather than sprintf, strlcpy rather than strcpy, and fgets
  rather than gets.
- [ ] **Python APIs:** unsafe execution is avoided; for example, use literal_eval
  rather than eval and shell=False with subprocess.

### Variable management

- [ ] **Initialization:** variables are initialized before use with a deny-by-
  default approach.
- [ ] **Typing:** unsigned types are used unless negative values are required.
- [ ] **Scope:** variables have minimal scope and are not reused improperly.

### Compilation and resilience

- [ ] **Compiler flags:** strict security flags are enabled where supported,
  including applicable equivalents of -Werror, -Wall, and
  -fstack-protector-strong.
- [ ] **Fault injection, when applicable:** critical low-level code has the
  necessary mitigations, such as redundancy and glitch resistance.

### Error handling

- [ ] **Action taken:** errors are handled and propagated; logging alone is not
  error handling.
- [ ] **No leakage:** error paths do not expose stack traces, memory, secrets,
  or other sensitive data.
- [ ] **Resource cleanup:** error handling releases acquired resources.

### Cryptography and secrets

- [ ] **No hardcoded secrets:** code contains no plaintext API keys, passwords,
  tokens, or similar credentials.
- [ ] **Strong cryptography:** only modern, publicly vetted algorithms suitable
  for the application are used.
- [ ] **Authentication first:** data is authenticated before decryption.
- [ ] **Safe randomness:** security decisions use a cryptographically secure
  random-number generator, such as /dev/urandom rather than rand().

### Access control and concurrency

- [ ] **Least privilege:** access is enforced with allowlists rather than
  blocklists.
- [ ] **Shared resources:** shared resources are protected from denial-of-
  service and privilege abuse.
- [ ] **Race conditions:** code is protected against time-of-check/time-of-use
  vulnerabilities.
- [ ] **No backdoors:** there is no intentional or accidental bypass mechanism.
