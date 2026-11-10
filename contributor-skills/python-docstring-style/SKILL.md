---
name: python-docstring-style
description: Write or revise FlashDreams Python docstrings and comments when adding an API, documenting fields or polishing documentation. Match nearby code and the current Google-style documentation renderer without expanding unrelated code changes.
---

# Python docstrings and comments

Read [Coding conventions](../../CONTRIBUTING.md#coding-conventions), inspect the
surrounding module, and check [the docs configuration](../../docs/zensical.toml)
when changing rendered API documentation. The current Python renderer uses
Google-style docstrings.

- Keep the existing SPDX notices; new files use the actual copyright holder and
  current year as described in CONTRIBUTING.md. Put the module docstring before imports.
- Use a concise summary. For functions, describe the operation with an imperative
  verb; add `Args`, `Returns` and `Raises` only where they explain semantics,
  shapes, defaults or caller-visible failure conditions beyond the signature.
- Match nearby attribute docstrings for class/dataclass fields and non-obvious
  module constants. Explain invariants once; avoid boilerplate for `_target` or
  values already clear from their names and types.
- Keep comments that explain intent, constraints or help readers scan a complex
  block. Remove stale history and literal restatements of adjacent code, without
  stripping useful comments wholesale.
- Use the current renderer's supported code/reference syntax. Do not introduce
  Sphinx-only roles or configuration assumptions into the Zensical docs.

Run the applicable formatter and documentation checks for the files changed.
Keep API behavior unchanged during documentation-only edits.
