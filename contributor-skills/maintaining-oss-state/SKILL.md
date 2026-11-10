---
name: maintaining-oss-state
description: Maintain FlashDreams repository licensing, dependency attribution, SPDX headers, REUSE annotations, and OSS-release collateral. Use when changing dependencies, vendoring source, or investigating reuse-lint failures; ordinary model use does not require this contributor workflow.
---

# Maintain FlashDreams OSS state

Work from the FlashDreams repository root. Read `CONTRIBUTING.md` for current
contribution and IP-review policy, and `.github/workflows/reuse-lint.yml` for the
checks actually enforced today.

For a dependency or vendoring change, read the applicable section of
[the maintenance reference](references/oss-maintenance.md) before editing:

- Sections 1-4: ownership of LICENSE/NOTICE/THIRD-PARTY-NOTICES/REUSE files and
  aggregate versus override annotations.
- Sections 5-6: dependency updates and physically redistributed third-party code.
- Sections 7-10: source headers, release records, DCO text and CI checks.
- Sections 11-13: common failures, focused operation checklists and file lookup.

Keep dependency declarations, lockfile and attribution consistent. Preserve
upstream notices when moving vendored files; do not replace their licensing
with project defaults. Documentation and config without inline headers are
covered by `REUSE.toml`. A Markdown skill starts with YAML frontmatter, not a
license comment.

The reference preserves historical release-review records and examples. Verify
current source and approval requirements for the actual change rather than
assuming those historical examples grant permission or describe every current
third-party component. Report required external paperwork separately from local
file changes; do not submit it unless the user authorized that action.
