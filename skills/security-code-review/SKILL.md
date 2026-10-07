---
name: security-code-review
description: Review FlashDreams changes for production, policy, and security risk. Use for security-sensitive changes, release-critical paths, or final merge approval.
---

# Security code review

Treat review as a release control, not a formality. The objective is to establish
correctness, security, and maintainability; enforce organizational standards;
create shared ownership and accountability; and stop defects before release.
A final approving reviewer takes ownership of whether the code is
production-ready, policy-compliant, and secure enough to ship. If the
reviewer would not stand behind the outcome, they must not approve.

## Review posture

- Process varies by SCM, but accountability does not. Preserve peer validation,
  separation of duties, an authenticated reviewer identity, and a traceable
  approval.
- Read the change in context. Trace affected behavior and trust boundaries;
  inspect callers, configuration, tests, deployment, and dependency effects as
  needed. Never approve from the diff alone or assume somebody else checked it.
- Treat automated scans as evidence, not proof. Passing CI and security scans do
  not make unsafe logic safe.
- AI may suggest questions, find common bugs, explain unfamiliar code, or
  propose tests. It cannot supply the accountable final approval. A qualified
  human must independently validate its output.
- Give extra scrutiny to authentication and authorization, cryptography,
  infrastructure and deployment pipelines, and AI/ML model or data handling.
- Review deletion of controls as carefully as adding sensitive code. Confirm a
  deletion does not remove required validation, access control, logging, or
  auditing. Test affected trust boundaries against documented security
  expectations and known abuse cases.
- Do not rubber-stamp to unblock a pipeline, ignore scan results, or trade the
  release gate for schedule pressure. Known or incomplete risk needs traceable
  owner acceptance that records its approver, rationale,
  and time-bound remediation; it is not an ordinary approval.

## General MUST checks

Verify each applicable item before final approval:

- **Code quality:** logic is clear and maintainable; dead or redundant code is
  absent; errors are handled properly.
- **Testing:** unit or integration coverage is adequate and relevant edge cases
  are handled.
- **Build and integration:** required CI passes cleanly and dependencies are not
  broken.
- **Documentation:** comments explain non-obvious intent where needed, and API
  or interface changes are documented.

## Security MUST checks

Verify each applicable item before final approval:

- **Secrets and credentials:** no hardcoded tokens, keys, passwords, or other
  secrets; the change uses a protected secret manager or runtime injection.
- **Input handling:** untrusted input is validated and sanitized, with no SQL,
  command, deserialization, or other injection path.
- **Dependencies:** no known-vulnerable package is introduced and the SBOM
  impact is understood when applicable.
- **Access control:** authorization checks are correct and no privilege-
  escalation path is introduced.
- **Sensitive data:** PII and regulated data are handled correctly and are not
  logged unintentionally.

## Security review checklist

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

## Findings and approval

Lead with concrete, fixable findings. For each one, identify the affected path
or trust boundary, the unsafe behavior, realistic conditions, and required
repair. Separate blocking security or policy findings from optional hardening
and ordinary maintainability comments. Record which checks were exercised and
what evidence supports the final judgment.

Final approval is an accountable role, not just the presence of a generic
review. The reviewer must use this checklist and understand the changed area
and its security boundaries. The change author or submitter is not the final
reviewer.

The GitHub users or teams on the catch-all line in `.github/CODEOWNERS` are
FlashDreams' final security-review delegates. Treat that file as the sole
reviewer roster; do not duplicate names in this skill. Keep enough active
owners so review coverage does not depend on one person, and route
security-sensitive changes to an owner with relevant context.

## Merge enforcement

- Put each final reviewer's `@github-username` or valid team handle on one
  catch-all line in `.github/CODEOWNERS`. GitHub treats an approval from any
  listed owner as sufficient. Do not split owners across duplicate `*` rules
  because only the last matching rule applies.
- In the `main` branch rule or ruleset, require a pull request, one approval,
  and a Code Owner review. A Code Owner's approval can satisfy the one-approval
  count and ownership gate together.
- Leave required-pull-request bypass disabled and ensure the protection applies
  to admins. Each listed owner must have Write or higher access to the
  repository; a team must also be visible.
- A Developer/Write review is useful but satisfies the final gate only when the
  reviewer is listed by the matching CODEOWNERS rule.
