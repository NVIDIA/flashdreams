---
name: maintaining-oss-state
description: Maintain FlashDreams's OSS-release state — the LICENSE / NOTICE / THIRD-PARTY-NOTICES / REUSE.toml / LICENSES/ / CONTRIBUTING.md collateral that satisfies OSRB Bug 6107043, the per-file SPDX headers, the third-party dependency manifest in THIRD-PARTY-NOTICES, and the pyproject.toml + uv.lock dependency pins. Use when adding or upgrading a runtime dependency, vendoring third-party source into the repo, adding a new first-party source file (any .py / .pyx / .pyi / .c / .cc / .cpp / .h / .hpp / .cu / .cuh / .sh / .proto / Dockerfile), reviewing whether a change requires reopening an OSRB bug or filing a self-cert, or triaging a reuse-lint CI failure.
---

# Maintaining FlashDreams's OSS state

FlashDreams is released to the public under Apache-2.0 (OSRB Bug
[6107043](https://nvbugswb.nvidia.com/NVBugs5/redir.aspx?url=/6107043)).
The repo carries a fixed set of collateral that the OSRB approved on, and
a `reuse-lint` CI workflow that fails the build if that collateral drifts.
This skill is the map for keeping the collateral consistent — what each
file is for, what edits trigger which downstream paperwork, and which CI
gates catch what.

> The reference design here was landed across PRs #54 (CONTRIBUTING.md),
> #55 (Apache-2.0 collateral), #111 (cudaraster + LodePNG disclosure),
> #119 (strict inline SPDX CI), and the `alpadreams → omnidreams` rename
> (#128 / #132). The git history of those commits is the canonical
> example for every operation described below.

## TL;DR

- **Six files at repo root + one CI workflow define OSS state:**
  `LICENSE`, `LICENSES/`, `NOTICE`, `THIRD-PARTY-NOTICES`,
  `REUSE.toml`, `CONTRIBUTING.md`, and
  `.github/workflows/reuse-lint.yml`. Touch any of them with the same
  care you'd give to a public API.
- **`LICENSE` has a multi-license preamble** explaining that the bulk
  of the repo is Apache-2.0 and that two subtrees
  (cudaraster → BSD-3-Clause, LodePNG → Zlib) carry different OSI
  licenses, then reproduces the full Apache-2.0 text. CI verifies the
  canonical Apache-2.0 sentinel strings are present; the structural
  cross-references in the preamble are not lint-checked (these files
  change rarely — see the change-log review path instead).
- **`NOTICE` is the minimal Apache 2.0 §4(d) notice** — NVIDIA
  copyright + pointers to `LICENSE`, `LICENSES/`, and
  `THIRD-PARTY-NOTICES`. It is *not* the full attribution document.
- **`THIRD-PARTY-NOTICES` is the full per-dependency attribution
  document** — direct runtime deps, reference architectures, optional
  integrations, and source-level redistributions, each with SPDX
  identifier and upstream URL.
- **Every first-party source file carries an inline SPDX header.**
  `REUSE.toml`'s `**` aggregate keeps the lint green for files that
  can't carry one (config, assets, binaries), but `reuse-lint`'s
  "Inline SPDX headers on first-party source files" step rejects any
  new `.py` / `.c` / `.cpp` / `.cu` / `.sh` / `.proto` / `Dockerfile`
  / etc. without the inline tag.
- **Direct deps are mirrored in three places:** the workspace member's
  `pyproject.toml` `dependencies`, the resolved `uv.lock` pin, and the
  `THIRD-PARTY-NOTICES` "Direct runtime dependencies" table. All
  three must agree.
- **Third-party source physically present in the repo** (cudaraster,
  LodePNG) lives in `THIRD-PARTY-NOTICES` "Source-level
  redistributions", carries the full license text under
  `LICENSES/<SPDX>.txt`, has a matching `REUSE.toml` `override`
  annotation, and is cross-referenced from the `LICENSE` preamble.
- **OSRB Bug 6107043 §14 is the source of truth for which deps are
  "covered" by the contribution approval.** Adding a new direct dep =
  reopen 6107043 and amend §14. Adding a transitive that only matters
  because the SBOM scanner flagged it = either reopen + amend §14, or
  file a self-cert under `osrb/`. Dev-only transitives → SBOM
  correction (not shipped).

## 1. The OSS-collateral file set

| File / path | Role | OSRB anchor |
|---|---|---|
| `LICENSE` | Multi-license preamble (Apache-2.0 + BSD-3 + Zlib pointer) followed by the canonical Apache-2.0 v2.0 text. CI verifies the canonical Apache-2.0 sentinel strings are present (preamble structure is not lint-gated). | 6107043 item #3 + OSRB unified-posture review |
| `LICENSES/Apache-2.0.txt` | REUSE 3.3 license-bundle copy of the canonical Apache-2.0 text (no preamble — must remain reusable verbatim by REUSE tooling). | 6107043 item #3 |
| `LICENSES/BSD-3-Clause.txt` | Full BSD-3 text covering the in-source cudaraster port. | 6107043 Cmt #5 (2) |
| `LICENSES/Zlib.txt` | Full Zlib text covering the embedded LodePNG codec. | 6107043 Cmt #5 (2) |
| `NOTICE` | Apache 2.0 §4(d) minimal notice — NVIDIA copyright + pointers to `LICENSE`, `LICENSES/`, and `THIRD-PARTY-NOTICES`. Carried forward verbatim by downstream redistributions. | Apache-2.0 §4(d) |
| `THIRD-PARTY-NOTICES` | Full per-dependency attribution: Direct runtime deps + Reference architectures + Optional-integration deps + Source-level redistributions. The source of truth for the third-party manifest. | OSRB review (canonical attribution doc) |
| `REUSE.toml` | REUSE 3.3 aggregate / override annotations for files without inline SPDX. | 6107043 item #2 |
| `CONTRIBUTING.md` | Apache-2.0-only contribution statement + DCO v1.1 reproduction + "Signing Your Work" subsection (with `git commit -s` and `--signoff`) + IP-review reference. | 6107043 item #6 + OSRB DCO template |
| `.github/workflows/reuse-lint.yml` | CI gate enforcing REUSE 3.3 compliance, presence of the five core OSRB collateral files (`LICENSE`, `LICENSES/Apache-2.0.txt`, `CONTRIBUTING.md`, `NOTICE`, `REUSE.toml`), canonical Apache-2.0 text in `LICENSE` and `LICENSES/Apache-2.0.txt`, `CONTRIBUTING.md` DCO/sign-off reference, inline SPDX headers (incl. `.sh` / `.proto` / Dockerfile), and no legacy proprietary banners. Content-shape of `LICENSE` preamble / `CONTRIBUTING.md` policy text / `THIRD-PARTY-NOTICES` sections is reviewed manually, not lint-checked. | (enforces the above) |

The CI workflow runs on every PR, every push to `main`, and inside the
GitHub merge queue. A failed `reuse-lint` blocks merge — never bypass it,
fix the underlying file.

## 2. Per-file SPDX headers

Every first-party source file starts with the inline SPDX header. The
exact wording is enforced by `reuse-lint`'s "Inline SPDX headers on
first-party source files" step (looks for `SPDX-License-Identifier` in
the first 20 lines).

**Python / shell / TOML / YAML** (`#` line comments):

```python
# SPDX-FileCopyrightText: Copyright (c) <YEAR> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
```

**C / C++ / CUDA** (`//` line comments): same two SPDX tags + the same
Apache-2.0 preamble, with `//` swapped for `#`.

Rules:

- `<YEAR>` is the **current calendar year** for newly created files
  (use the system clock — *not* a model-training-cutoff year). For
  files being **edited**, leave the year alone — it reflects original
  authorship, not last-touched.
- The two SPDX tags (`SPDX-FileCopyrightText` + `SPDX-License-Identifier`)
  are the load-bearing part. The Apache-2.0 preamble is house style;
  the CI gate only checks for `SPDX-License-Identifier` in the first 20
  lines, but the long form is what every existing file carries, so
  match it.
- External contributors add their **own** copyright line *above* the
  NVIDIA line — keep both. See `CONTRIBUTING.md:200-235`.
- The Cosmos-Drive-Dreams files
  (`integrations/omnidreams/omnidreams/conditioning/world_scenario/{camera_base,ftheta,pinhole}.py`)
  carry two `SPDX-FileCopyrightText` lines (NVIDIA + Cosmos-Drive-Dreams
  contributors). Mirror that pattern when redistributing other modified
  upstream Apache-2.0 source.

### What's exempt (handled by `REUSE.toml`)

- Vendored upstream under `integrations/omnidreams/ludus-renderer/ludus_renderer/_cpp/cudaraster/**`
  (carries its own BSD-3 / Zlib banners; `override` annotation in REUSE.toml).
- Generated protobuf stubs under
  `integrations/omnidreams/omnidreams/grpc/protos/*_pb2*` (regeneration
  script is project-owned; aggregate annotation covers them).
- Binary assets (`assets/**.png`, `.jpg`, `.jpeg`, `.webp`, `.mp4`,
  `.gif`, `.svg`) and lock files (`uv.lock`).
- Documentation (`**.md`, `**.rst`, `docs/**`).

If a tracked source file genuinely cannot carry an inline header (a
tooling-generated artifact, an asset, a config file), extend `REUSE.toml`
rather than fighting the lint.

## 3. `REUSE.toml` — aggregate vs override

`REUSE.toml` is the REUSE 3.3 manifest that fills gaps the inline SPDX
header convention can't.

Precedence rules (from `REUSE.toml`):

- **`precedence = "aggregate"`** — declared license merges with any
  inline SPDX header the file carries. Used for the project-wide default
  (`path = "**"` → Apache-2.0) and for documentation / config /
  build / asset blocks.
- **`precedence = "override"`** — declared license replaces any inline
  SPDX header. Used for vendored upstream (`cudaraster/**` →
  BSD-3-Clause; `lodepng/**` → Zlib) to keep the upstream banners
  authoritative while still surfacing the SPDX identifier REUSE needs.

When to add an annotation block:

| Scenario | Block style | precedence |
|---|---|---|
| Add a new first-party source-file type covered by the default `**` rule | (nothing — default covers it) | n/a |
| Add a new asset / config / generated-output path that can't carry an inline header | new `[[annotations]]` block, copyright + Apache-2.0 | `aggregate` |
| Add a new third-party source subtree (different license, banners we want to keep) | new `[[annotations]]` block, copyright = upstream, SPDX = upstream's license | `override` |
| Add a redistributed-and-modified upstream Apache-2.0 file | new block listing dual `SPDX-FileCopyrightText` (NVIDIA + upstream) | `aggregate` |

More specific paths win — the cudaraster `override` covers everything
under that subtree, then the lodepng `override` overrides the cudaraster
block for the lodepng leaf. Order the annotations so specific paths come
*after* general ones.

## 4. `NOTICE` vs `THIRD-PARTY-NOTICES` — what goes where

Two distinct files. Mixing them up is the most common OSS-state mistake.

### `NOTICE` — minimal, downstream-propagated

`NOTICE` exists to satisfy **Apache 2.0 §4(d)**: any derivative work must
carry a readable copy of the upstream `NOTICE` text. Keep it small so
downstream consumers do not pay an unreasonable carry-forward cost.
Shape:

```
NVIDIA FlashDreams
Copyright (c) <YEAR> NVIDIA CORPORATION & AFFILIATES. All rights reserved.

This product is licensed under the Apache License, Version 2.0; the
full license text is reproduced in LICENSE at the repository root and
in LICENSES/Apache-2.0.txt.

Two subtrees physically vendored into this repository carry
different OSI-approved licenses; full texts are reproduced under
LICENSES/:

  - integrations/omnidreams/ludus-renderer/ludus_renderer/_cpp/cudaraster/
        BSD-3-Clause  (see LICENSES/BSD-3-Clause.txt)
  - integrations/omnidreams/ludus-renderer/ludus_renderer/_cpp/
    cudaraster/framework/3rdparty/lodepng/{lodepng.h,lodepng.cpp}
        Zlib          (see LICENSES/Zlib.txt)

Third-party software attributions, source-level redistribution
disclosures, and the full per-dependency license inventory are
documented in THIRD-PARTY-NOTICES at the repository root.
```

Do not enumerate transitive dependencies, SPDX tables, or per-package
attributions in `NOTICE`. Those go in `THIRD-PARTY-NOTICES`.

### `THIRD-PARTY-NOTICES` — full per-dependency manifest

`THIRD-PARTY-NOTICES` is the consumer-facing attribution document and
the source of truth for the third-party manifest. It has four named
sections; do not invent new ones without a corresponding `REUSE.toml`
change.

```
NVIDIA FlashDreams — Third-Party Notices
Copyright (c) <YEAR> NVIDIA CORPORATION & AFFILIATES. All rights reserved.

  Preamble explaining the dynamic-import / no-source-redistribution
  default and pointing readers at the Source-level redistributions
  section at the bottom.

================================================================================
Direct runtime dependencies
================================================================================

  <name>  <SPDX>  <upstream-URL>
  ... one row per direct dep ...

================================================================================
Reference architectures
================================================================================

  Wan 2.1 / Wan 2.2   Apache-2.0   https://github.com/Wan-Video/Wan2.1
      <one paragraph explaining the reference-architecture relationship
       and confirming weights/sources aren't redistributed>

================================================================================
Optional integration: integrations/<name>
================================================================================

  <one block per integration that pulls in deps not used by the root
   package — currently omnidreams (mediapy, opencv-python-headless,
   grpcio, shapely, ludus-renderer) and lingbot (aiohttp, aiortc,
   opencv-python-headless)>

================================================================================
Source-level redistributions
================================================================================

  <one block per subtree of third-party source physically in the repo>
```

Rules:

- **Column 1 = exact PyPI / upstream name.** Match
  `flashdreams/pyproject.toml`'s `dependencies =` spelling
  (e.g., `opencv-python-headless`, not `opencv`).
- **Column 2 = SPDX identifier** (from <https://spdx.org/licenses/>). For
  dual-licensed packages use comma-separated SPDX IDs in alphabetical
  order, e.g., `MIT, MPL-2.0` for tqdm.
- **Column 3 = upstream source URL**, not the PyPI page.
- **Only direct deps go in the top table.** Transitives stay out unless
  they're material enough to flag separately under "Optional
  integration" blocks.
- **Reference architectures get their own block** with a 2–4 line
  explanation (we implement the architecture; we don't redistribute the
  upstream code or weights).
- **The "Source-level redistributions" section is the only place
  physically-present third-party source is acknowledged.** Each block:
  Path (absolute from repo root), License (SPDX + pointer to
  `LICENSES/<SPDX>.txt`), Upstream URL, and a paragraph explaining what
  was modified vs. what's upstream code.

## 5. Adding or upgrading a runtime dependency

This is the highest-frequency OSS-state edit. It touches **four** places:

1. `flashdreams/pyproject.toml` (or the workspace member that needs
   the dep) — add to `dependencies = [...]`. Pin a floor (`>=`) on
   semver-stable packages; pin tightly (`==`) only when the upstream
   API is known-unstable across minor versions.
2. `uv.lock` — regenerate with `uv lock` so the hash-pinned resolved
   version lands in the lockfile.
3. `THIRD-PARTY-NOTICES` "Direct runtime dependencies" table — add a row with
   `name  SPDX  upstream-URL`.
4. OSRB Bug 6107043 §14 — **reopen the bug and amend §14** with the
   new (name, version, license, URL) row. Per OSRB policy, for ongoing
   contributions the previously-approved contribution bug must be
   reopened whenever a new package is added to the product delivery,
   a previously-approved package changes its license, or the use of a
   previously-approved component changes.

### Pre-add checklist

- [ ] **Confirm the SPDX license**. Read the upstream LICENSE file
      (don't trust GitHub's auto-detected badge). MPL-2.0, LGPL,
      AGPL, GPL, EPL, MS-PL all carry copyleft conditions; loop in
      legal *before* adding.
- [ ] **Confirm the dep is published from PyPI**, not from a private
      index. If it isn't on PyPI, write `[tool.uv.sources]` with care
      and flag for OSRB review.
- [ ] **Check for security advisories** (Snyk, BDSA, NVD). The repo
      has carried floor pins for security reasons before
      (`urllib3>=2.7.0` for the botocore/requests CVE chain). If a
      floor is needed, leave a one-line comment in `pyproject.toml`
      explaining why.
- [ ] **MPL-2.0 / weak-copyleft**: confirm dynamic import only, no
      modifications, no source redistribution. If any of those don't
      hold, the dep needs to be vendored (see §6) and OSRB review is
      mandatory.
- [ ] **Codec / crypto**: if the new dep implements an audio/video
      codec or encryption, the entry belongs in the corresponding
      Optional-integration block of `THIRD-PARTY-NOTICES`, and the
      OSRB bug Q11 / Q12 answers may need re-confirming.

### Upgrading an existing dep version

- **Same license, same SPDX, version-bump only** → update
  `pyproject.toml` floor (if needed), regen `uv.lock`.
  `THIRD-PARTY-NOTICES` may not need a touch (we don't pin exact
  versions there). OSRB bug does **not** need to be reopened (policy
  explicit: "version updates without licensing changes don't require
  reopening").
- **License change between versions** → treat as a new dep:
  reopen 6107043 §14, update `THIRD-PARTY-NOTICES`, possibly re-check
  codec/crypto questions.
- **Major version bump that changes the dep's *use*** (e.g.,
  switching from sync-only to async-only, or adding a new transitive
  family) — reopen 6107043 §14 even if the SPDX hasn't changed
  ("The use of a previously approved component changed").

### `uv.lock` hygiene

- Always regenerate the lock from the workspace root: `uv lock`.
- Commit `pyproject.toml` and `uv.lock` together — they are
  jointly maintained (see commit `9480367` ownership notes).
- If a dep's transitives shift in a way that drops or adds a
  *direct-of-direct* (e.g., `httpx` → drops `certifi`), the new
  closure is what the OSRB SBOM scanner will see — re-run the
  scanner after the merge so anything new gets caught.

## 6. Adding a third-party source-level redistribution

Vendoring upstream source into the repo (the cudaraster + LodePNG
pattern, PR #111) is heavier — it touches **six** places:

1. **Physically place the source** under an `integrations/.../` subtree
   that signals it's third-party (e.g.,
   `integrations/omnidreams/ludus-renderer/ludus_renderer/_cpp/cudaraster/`).
   Keep the upstream banners verbatim in the file headers — don't
   replace them with NVIDIA SPDX headers.
2. **Add the full license text** under `LICENSES/<SPDX>.txt`.
3. **Extend `REUSE.toml`** with an `override` annotation for the
   subtree (license = upstream SPDX, copyright = upstream copyright).
4. **Add a "Source-level redistributions" block in
   `THIRD-PARTY-NOTICES`** — path, license + pointer to
   `LICENSES/<SPDX>.txt`, upstream URL, one paragraph describing what
   we modified vs. upstream. **Also update `NOTICE`** to add a
   one-line entry under the existing two-subtree bullet list, since
   physically-redistributed third-party source is one of the things
   downstream consumers must see when carrying our Apache 2.0
   §4(d) notice forward. **Also update the `LICENSE` preamble** to
   cross-reference the new `LICENSES/<SPDX>.txt`.
5. **OSRB Bug 6107043** — reopen + add a §14 row *and* note the
   source-level redistribution in a comment (per Cmt #5 (2) workflow).
   Filing an OSRB Bug for the upstream project itself may be required
   if it has its own OSRB process (ludus-renderer = bug 6105127).
6. **`reuse-lint` exclusion** in `.github/workflows/reuse-lint.yml`'s
   "Inline SPDX headers on first-party source files" step — extend
   the `excludes` regex to skip the new subtree, since upstream
   banners use the upstream license, not Apache-2.0.

Reference: commits `100c0f8` (initial collateral), `1d8c9ed`
(cudaraster + LodePNG vendoring), `8f2aedf` (ludus-renderer REUSE 3.3
compliance for sub-bug 6105127).

## 7. Adding a new first-party source file

Easy path — `reuse-lint` will fail the PR if you skip a step.

1. Open the file with the SPDX header (see §2). The current calendar
   year for newly authored files.
2. Save / `git add`. No `REUSE.toml` change needed — the `**`
   default rule covers it.
3. If the file lives under a path that has an `override` annotation
   (cudaraster, lodepng), it inherits the upstream license — only do
   this when the file genuinely *is* upstream-derived, not because
   it's convenient.

The `reuse-lint` "No NVIDIA proprietary banners" step will reject any
file that still carries the legacy NVIDIA-CONFIDENTIAL banner. If you
ported source from an internal repo, strip the old banner and replace
with the Apache-2.0 SPDX header.

## 8. OSRB bug interaction matrix

| Change | OSRB action |
|---|---|
| Bump version, same license | None (policy explicit). |
| Bump version, license changed | Reopen 6107043, amend §14. |
| Add new direct dep | Reopen 6107043, amend §14, update `THIRD-PARTY-NOTICES`. |
| Add new transitive flagged by SBOM scanner | Reopen 6107043 §14 (preferred), OR file self-cert under `osrb/`. |
| Add dev/test-only transitive (`[dev]` extra) flagged by scanner | File **SBOM correction** — not in product delivery. Self-cert as fallback. |
| Vendor third-party source physically into repo | Reopen 6107043 + Cmt thread; possibly file a sub-OSRB bug for the upstream project (cf. 6105127 for ludus-renderer). |
| Remove a dep | Update `pyproject.toml`, `uv.lock`, `THIRD-PARTY-NOTICES`. No OSRB action — removal doesn't add new attack surface. |
| Drop a previously-approved transitive (closure shift) | Update `THIRD-PARTY-NOTICES` if it was listed; no OSRB action required. |

OSRB self-cert templates live under `osrb/` (e.g.,
`osrb/selfcert-certifi-2026.4.22.md`). Mirror the OSS-USE form
shape — see prior tickets for the field list.

### MPL-2.0 in particular

NVIDIA accepts MPL-2.0 use when **all three** hold:

1. **No modifications** to the upstream MPL-2.0 source.
2. **No source redistribution** — the dep is consumed from PyPI at
   install time, not vendored.
3. **Dynamic linking only** (Python `import`).

Document this trio explicitly on every MPL-2.0 self-cert ticket. If any
of the three fails, the dep needs a regular Use bug at
<https://nvbugs/5443768>.

## 9. Updating CONTRIBUTING.md

The DCO v1.1 text is reproduced verbatim in `CONTRIBUTING.md:108-133`.
**Do not paraphrase, summarize, or "modernize" it** — the `reuse-lint`
collateral step looks for the exact pattern
`Developer.{1,40}Certificate.{1,10}of.{1,10}Origin|Signed-off-by|sign-off`,
and OSRB approval is on the *verbatim* text.

When extending CONTRIBUTING.md:

- Keep the DCO section anchored at `## Developer Certificate of Origin
  (DCO)` — the README and external docs link to it by anchor.
- The SPDX header preamble at `CONTRIBUTING.md:200-235` doubles as the
  agent-and-human source for what every new source file's header should
  look like. Update it and `python-docstring-style/SKILL.md` together.
- The IP-review-process reference in `CONTRIBUTING.md` is an OSRB
  pointer — don't change it without OSRB sign-off.

## 10. CI gates — what `reuse-lint` enforces

The workflow has two jobs and five checks. Read
`.github/workflows/reuse-lint.yml` if you need to add a new gate.

The gate set is intentionally narrow — these files change rarely and
the cost of over-fitted CI (false positives, sweeping rewrites
needed when wording shifts) exceeds the benefit. Treat the lint as a
backstop for structural regressions (missing collateral file, missing
SPDX header, legacy banner) and trust human review for content shape
(preamble references, contribution policy wording, attribution-table
sections).

**`reuse` job** (`fsfe/reuse-action@v5`):

- REUSE 3.3 lint: every tracked file has an SPDX identifier, either
  inline or via `REUSE.toml`. New files without coverage fail.

**`collateral` job** (custom bash):

1. `LICENSE` and `LICENSES/Apache-2.0.txt` both contain the canonical
   Apache-2.0 sentinel strings. (Byte-identical equality is no longer
   required — `LICENSE` carries a multi-license preamble in front of
   the Apache-2.0 body.)
2. The five core OSRB collateral files exist at the repo root:
   `LICENSE`, `LICENSES/Apache-2.0.txt`, `CONTRIBUTING.md`, `NOTICE`,
   `REUSE.toml`. (`LICENSES/BSD-3-Clause.txt`, `LICENSES/Zlib.txt`,
   and `THIRD-PARTY-NOTICES` also need to be present per OSRB
   approval, but are not lint-gated — they change slowly and a
   manual review catches drift sooner than the cost of over-fitted
   CI would justify.)
3. `CONTRIBUTING.md` references the DCO / sign-off. (The explicit
   "Apache-2.0-only" sentence, "Signing Your Work" subsection, and
   `git commit -s` short-form example are required by the OSRB
   template but are not separately lint-asserted.)
4. Every tracked source file (`.py`, `.pyx`, `.pyi`, `.c`, `.cc`,
   `.cpp`, `.cxx`, `.h`, `.hh`, `.hpp`, `.hxx`, `.cu`, `.cuh`,
   `.inl`, `.sh`, `.proto`, `Dockerfile` / `*.dockerfile`) carries
   an inline `SPDX-License-Identifier` in its first 20 lines — with
   the documented exclusions (`cudaraster/**`, generated protobuf
   stubs).
5. No file contains the legacy NVIDIA proprietary-banner sentinel
   phrases checked by `.github/workflows/reuse-lint.yml`.

Triggers: every PR, every push to `main`, and every merge-queue group
(see `b84fe7d` for the `merge_group` trigger landing).

## 11. Common pitfalls

- **Editing `LICENSE` without mirroring into `LICENSES/Apache-2.0.txt`**
  — the collateral step compares them byte-for-byte. If you fix a typo
  in one, fix it in both.
- **Adding a new direct dep and forgetting `THIRD-PARTY-NOTICES`**.
  The lint won't catch this (the file is free-form prose). Add the
  attribution row in the same commit that touches `pyproject.toml` /
  `uv.lock` and the matching PR.
- **Touching `NOTICE` when only `THIRD-PARTY-NOTICES` should
  change**. `NOTICE` is the small Apache 2.0 §4(d) file that
  downstream consumers carry forward verbatim — keep it minimal.
  Routine dep additions belong in `THIRD-PARTY-NOTICES`. Touch
  `NOTICE` only when (a) the year on line 2 rolls forward, (b) a new
  source-level redistribution subtree appears (rare), or (c) the
  pointer text needs to mention a new top-level OSS file.
- **Renaming an integration (e.g., alpadreams → omnidreams)** —
  remember to update the matching path in `REUSE.toml`'s annotations,
  the path in `THIRD-PARTY-NOTICES`'s "Source-level redistributions"
  block, the path in the `NOTICE` two-subtree bullet list, the path
  in the `LICENSE` preamble, and the exclusion regex in
  `.github/workflows/reuse-lint.yml`. The rename in PRs #128 / #132
  is the reference.
- **Using single-quoted SPDX-FileCopyrightText**. REUSE is forgiving;
  the rest of the repo uses double-quoted strings. Mismatch breaks
  human grep, not the lint.
- **Year stamp drift**. The SPDX header year is *when the file was
  first authored*, not when it was last touched — never mass-rewrite
  the year across the tree. The two exceptions are line 2 of `NOTICE`
  and line 2 of `THIRD-PARTY-NOTICES` (and the project copyright in
  the `LICENSE` preamble), which are the project's overall copyright
  year and are allowed to roll forward annually.
- **Adding an OSRB self-cert ticket to a public branch.** OSRB tickets
  are NVIDIA-internal — file them on the `gitlab` remote
  (`gitlab-master.nvidia.com/sil/flashdreams`), not on `origin`
  (github.com/NVIDIA/flashdreams). Public-facing skill, drafts, and
  process docs are fine on `origin`.
- **Skipping `uv lock` after a `pyproject.toml` edit.** The lockfile
  is the source of truth for what consumers actually install; an
  out-of-sync `uv.lock` is a real bug, not a cosmetic one.
- **Adding a dep "just for tests" without `[dev]` extra placement.**
  If a dep lives in the `dev` extra, it's not in the product delivery
  and is out of OSRB scope per the policy bullet. If it's in
  `dependencies = [...]`, it ships to every consumer — OSRB-scoped.
  The `[dev]` extras in `flashdreams/pyproject.toml` and
  `integrations/*/pyproject.toml` are the seams.
- **Forgetting that `gitlab/main` and `origin/main` have diverged.**
  Internal `gitlab` `main` carries 10+ commits not on `origin`
  (`Add --offload-text-encoder for batch run`, etc.) and is missing
  ~77 from `origin`. When opening MRs against gitlab, base your branch
  on `gitlab/main`; when opening PRs against github, base on
  `origin/main`. Tooling-and-doc branches like this one belong on
  github (canonical project home); OSRB-ticket-draft branches belong
  on gitlab (NVIDIA-internal).

## 12. Scaffolding checklist — full operations

**Add a new direct runtime dep `foo` (semver-stable, Apache/MIT/BSD):**

1. Add `"foo>=X.Y"` to the right workspace's `pyproject.toml`
   `dependencies = [...]`.
2. `uv lock` from the workspace root; commit `pyproject.toml` +
   `uv.lock` together.
3. Add row to `THIRD-PARTY-NOTICES` "Direct runtime dependencies":
   `foo  <SPDX>  <upstream-URL>`.
4. Reopen OSRB Bug 6107043, amend §14 with the new row, and ping for
   re-approval.
5. PR + CI passes `reuse-lint` → merge.

**Add a new direct runtime dep `bar` (MPL-2.0 / LGPL / other
weak-copyleft):**

1. Verify dynamic-import / no-mod / no-redistribute trio (§8). If any
   fails, stop and engage OSRB.
2. Same four steps above.
3. *Also* file a self-cert ticket under `osrb/selfcert-bar-<ver>.md`
   on the `gitlab` remote (cf. existing certifi / regendoc drafts).

**Bump dep `baz` from 2.x to 3.x (same license):**

1. Verify SPDX unchanged at the new version.
2. Update `pyproject.toml` floor if API contract requires.
3. `uv lock`; commit `pyproject.toml` + `uv.lock`.
4. No OSRB action.
5. (Optional) Update `THIRD-PARTY-NOTICES` if its row drifts
   (e.g., URL changed).

**Vendor upstream `qux` (BSD-3) into `integrations/foo/qux/`:**

1. Drop source in with upstream banners preserved.
2. Add `LICENSES/BSD-3-Clause.txt` if not already present.
3. New `[[annotations]]` block in `REUSE.toml` with
   `precedence = "override"`, BSD-3 SPDX, upstream copyright, path
   `"integrations/foo/qux/**"`.
4. New block in `THIRD-PARTY-NOTICES` "Source-level
   redistributions" — path,
   `License: BSD-3-Clause (see LICENSES/BSD-3-Clause.txt)`,
   upstream URL, modification paragraph. Also add a bullet to the
   `NOTICE` two-subtree list and a pointer in the `LICENSE`
   preamble.
5. Extend the `excludes` regex in
   `.github/workflows/reuse-lint.yml`'s inline-SPDX step.
6. Reopen OSRB Bug 6107043; file a sub-OSRB if `qux` has its own
   project-level OSRB process.

**Triage a `reuse-lint` failure:**

1. Read the failed step name.
2. `REUSE 3.3 compliance` → run `pipx run reuse lint` locally; add
   inline SPDX or extend `REUSE.toml`.
3. `LICENSE / LICENSES/Apache-2.0.txt are byte-identical` →
   `diff LICENSE LICENSES/Apache-2.0.txt`, restore parity.
4. `Required OSRB collateral present` → recreate the missing file
   from history (`git log -- <file>` to find the original commit).
5. `CONTRIBUTING.md references the DCO` → restore the DCO section
   anchor and verbatim text.
6. `Inline SPDX headers on first-party source files` → the step
   prints every offending path as a GitHub annotation. Add the header
   to each.
7. `No NVIDIA proprietary banners` → strip the legacy banner from
   the listed file(s), replace with the Apache-2.0 SPDX header.

## 13. Where this maps in the codebase

| Question | File / pointer |
|---|---|
| What's the canonical Apache-2.0 text? | `LICENSE` (= `LICENSES/Apache-2.0.txt`) |
| What deps does FlashDreams ship? | `THIRD-PARTY-NOTICES` "Direct runtime dependencies" + `flashdreams/pyproject.toml` |
| What does a SPDX header look like? | `CONTRIBUTING.md:215-232` |
| Where do I declare a config / asset file's license? | `REUSE.toml` |
| Where do I record vendored upstream source? | NOTICE "Source-level redistributions" + `REUSE.toml` `override` block |
| What does the CI gate enforce? | `.github/workflows/reuse-lint.yml` (read top to bottom) |
| Which deps did OSRB approve under 6107043? | The §14 table on the bug itself (kept in sync with NOTICE) |
| Where do OSRB self-cert drafts live? | `osrb/` on the `gitlab` remote |
| Who do I tag if OSRB needs reopening? | The reviewer on 6107043 (Michael Hasper / MHASPER for FlashDreams) |
