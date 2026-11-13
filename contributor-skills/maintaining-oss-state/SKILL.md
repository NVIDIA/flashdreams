---
name: maintaining-oss-state
description: Maintain FlashDreams licensing, dependency attribution, SPDX headers and REUSE annotations when changing dependencies, vendoring source or investigating reuse-lint failures. Ordinary model use does not require this contributor workflow.
---

# Maintain FlashDreams OSS state

Read [CONTRIBUTING.md](../../CONTRIBUTING.md#licensing-of-contributions) and its
coding-convention and IP-review sections for the current policy. Inspect
[the reuse-lint workflow](../../.github/workflows/reuse-lint.yml) for the check
that actually failed, rather than relying on historical release instructions.

For dependency changes, keep the owning `pyproject.toml`, `uv.lock` and applicable
entries in `THIRD-PARTY-NOTICES` consistent. For vendored source, preserve upstream
attribution, record provenance, and check the relevant `LICENSES/`, `REUSE.toml`,
`LICENSE` and `NOTICE` entries. Do not replace third-party terms with project defaults.

Use the actual copyright holder and preserve existing notices. Files without
inline headers use REUSE annotations; skill YAML frontmatter must remain first.
Run the applicable checks, report unresolved attribution or approval questions,
and follow the current maintainer review process. Historical approval for one
release does not establish approval for a new dependency or use.
