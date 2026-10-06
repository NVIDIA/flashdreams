---
name: security-code-review
description: Review FlashDreams changes for production, policy, and security risk. Use for security-sensitive changes, release-critical paths, final merge approval, or checking whether review controls satisfy the +2 requirement.
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
  auditing. Validate sensitive changes against applicable threat models,
  security design patterns, and known abuse cases.
- Do not rubber-stamp to unblock a pipeline, ignore scan results, or trade the
  release gate for schedule pressure. Known or incomplete risk needs the formal,
  traceable acceptance required by policy, including its approver, rationale,
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
  secrets; the change uses the approved secret-management mechanism.
- **Input handling:** untrusted input is validated and sanitized, with no SQL,
  command, deserialization, or other injection path.
- **Dependencies:** no known-vulnerable package is introduced and the SBOM
  impact is understood when applicable.
- **Access control:** authorization checks are correct and no privilege-
  escalation path is introduced.
- **Sensitive data:** PII and regulated data are handled correctly and are not
  logged unintentionally.

## +2 security review checklist

Use this checklist actively; record evidence or a finding rather than silently
assuming an item passed. Mark an item not applicable only with a short reason.

### Process and static analysis

- [ ] **SCA clean:** all critical and high-severity findings identified by
  static code analysis have been resolved.
- [ ] **Risk assessed:** the change has been evaluated against the applicable
  Threat and Vulnerability Analysis (TAVA).

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
  tokens, internal IP addresses, or similar secrets.
- [ ] **Strong cryptography:** only modern, vetted algorithms are used, such as
  those in CNSA Suite 2.0.
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

The +2 Security Reviewer is a qualified approval role, not a number of generic
reviews. The reviewer must have completed the approved training, use this
checklist, and be the designated Security PIC or trained delegate when required.
The change author or submitter is not a reviewer.

Verify enforcement separately from reviewing code. Protected target branches
must require an `APPROVED` pull-request review from an authenticated human with
a Developer/Write, Maintainer/Maintain, or Admin role. For FlashDreams, this
role-based approval is accepted as the +2 SCM enforcement; a separate required
CODEOWNER or Security PIC rule is not needed. Confirm the rule applies to every
target branch and leaves no unaudited bypass. Reviewer training, checklist use,
and any Security PIC sign-off remain procedural evidence for the review.
