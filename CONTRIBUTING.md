# Contributing to FlashDreams

Thanks for your interest in contributing to **FlashDreams**. This project
is developed openly on GitHub and released under the
[Apache License 2.0](https://github.com/NVIDIA/flashdreams/blob/main/LICENSE). Outside contributions — bug reports,
feature requests, performance improvements, new model integrations,
documentation fixes — are genuinely welcome, and this guide explains how
they fit in alongside the project's day-to-day work.

We have intentionally kept the process light. If something below is
unclear or feels heavier than it should be, that's a bug; please file an
issue and we'll fix it.

Setup and build instructions:

- [install and run FlashDreams](https://nvidia.github.io/flashdreams/main/quickstart/);
- [work on one integration package](https://nvidia.github.io/flashdreams/main/quickstart/#install-one-integration);
- [build and preview the website](https://nvidia.github.io/flashdreams/main/documentation/tools/build_website/).

## Table of contents

1. [Ways to contribute](#ways-to-contribute)
2. [Project governance](#project-governance)
3. [Developer Certificate of Origin (DCO)](#developer-certificate-of-origin-dco)
4. [Submitting a pull request](#submitting-a-pull-request)
5. [Code review and merge](#code-review-and-merge)
6. [File Tree Of FlashDreams](#file-tree-of-flashdreams)
7. [Adding documentation](#adding-documentation)
8. [Coding conventions](#coding-conventions)
9. [Testing](#testing)
10. [Dependency version bounds](#dependency-version-bounds)
11. [Working with a single integration package](#working-with-a-single-integration-package)
12. [Licensing of contributions](#licensing-of-contributions)
13. [Reporting issues](#reporting-issues)
14. [Code of Conduct](#code-of-conduct)

## Ways to contribute

There are several useful ways to help out, ordered roughly from "low
overhead" to "high overhead":

- **Try FlashDreams and tell us what broke.** A clear bug report — what
  you ran, what you expected, what you saw — is one of the most valuable
  contributions a project of this kind can receive.
- **Improve documentation.** Website articles, integration walkthroughs,
  performance notes, and troubleshooting entries all land easily and benefit
  every future user.
- **Fix bugs.** Issues labelled `good first issue` are a friendly
  starting point. Larger fixes are welcome too — please leave a comment
  on the relevant issue first so we can avoid duplicate work.
- **Add or extend integrations.** New video-generation models, new schedulers,
  new integrations. For non-trivial features, please open a design issue
  before sending the PR (see [Submitting a pull request](#submitting-a-pull-request)).
- **Performance work.** FlashDreams cares about latency and throughput
  on NVIDIA GPUs. Numbers and reproducible benchmarks make these PRs
  easy to evaluate.

## Project governance

FlashDreams was developed inside NVIDIA's Simulation & Imitation
Learning group, and at the time of release NVIDIA holds the
maintainer and admin roles on the
[`NVIDIA/flashdreams`](https://github.com/NVIDIA/flashdreams) repository.
That includes the `main` branch protections, release tags, the package
publishing keys, and the right to merge.

We treat that as a starting point, not an endpoint. Our intent — modelled
on projects like [Slang](https://github.com/shader-slang/slang), which
moved from a single-vendor home to community governance once it grew an
external user base — is to open governance up as a contributor community
develops. Concretely, that means:

- **CODEOWNERS is the source of truth for review responsibility.** As
  contributors take long-term ownership of a subsystem, they get added
  there and become required reviewers for changes in that area, NVIDIA
  employee or not.
- **Decisions happen in public.** Significant design changes are
  discussed in GitHub issues, pull requests, or
  [Discussions](https://github.com/NVIDIA/flashdreams/discussions).
  Internal NVIDIA roadmap planning that touches the public project will
  surface as a public issue before it lands.
- **Release notes credit external contributors** by name and PR.
- **Open path to maintainer.** Contributors who consistently land
  high-quality work in an area, participate in reviews, and engage with
  the issue tracker can be invited to become maintainers. There is no
  fixed time bar; sustained good judgment is what we look for.

If you have feedback on governance — including things you'd like to see
formalised faster — please open a Discussion. We'd rather hear it than
not.

## Developer Certificate of Origin (DCO)

**This project will only accept contributions under the Apache-2.0
license.** By submitting a pull request you agree that your
contribution is licensed under the Apache License, Version 2.0 (see
[LICENSE](https://github.com/NVIDIA/flashdreams/blob/main/LICENSE)).

All contributions to FlashDreams are made under the
[Developer Certificate of Origin](https://developercertificate.org/).
This is a lightweight, well-understood mechanism (used by the Linux
kernel, GitLab, NVIDIA TensorRT, and many other projects) that lets you
attest that you have the right to submit your contribution under the
project's license — without requiring a separate Contributor License
Agreement.

Full text of the [Developer Certificate of Origin](https://developercertificate.org/):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Pull requests without DCO sign-off will be asked to rebase before merge.
This is a hard gate; please don't take a polite ping personally.

NVIDIA contributors and external contributors follow the *same* DCO
process. NVIDIA reviewers follow the
[NVIDIA IP review process](https://nv/ip_review_process) for ongoing project
modifications and third-party contributions. Reviewers handle this internal
process on your behalf — you do not need to engage with it as an outside
contributor.

### Signing Your Work

We require that all contributors sign-off on their commits. This
certifies that the contribution is your original work, or you have
rights to submit it under the same license, or a compatible license.
Any contribution which contains commits that are not Signed-Off will
not be accepted.

To sign off on a commit, use the `--signoff` (or `-s`) option when
committing your changes:

```bash
$ git commit -s -m "Add cool feature."
```

This will append the following trailer to your commit message:

```
Signed-off-by: Your Name <your@email.com>
```

The `user.name` and `user.email` git config values must be set to your
real name and a verifiable email address — sign-offs from anonymous or
pseudonymous identities cannot be accepted.

## Submitting a pull request

The short version:

1. Fork the repo on GitHub and create a feature branch from `main`.
2. Make your changes. Keep PRs small and focused; a 200-line PR that
   does one thing reviews much faster than a 2000-line PR that does ten.
3. Add or update tests where it makes sense. Every test must carry
   exactly one CI tier marker (`@pytest.mark.ci_cpu`,
   `@pytest.mark.ci_gpu`, or `@pytest.mark.manual`); see
   [Testing](#testing) below. The project enforces this at collection
   time -- pytest will error if a test is missing a marker.
4. Run the project's checks locally:

   ```bash
   uv run --group lint pre-commit run -a       # format + lint + type-check
   uv run --group test pytest -m ci_cpu         # CPU tests (no GPU required)
   ```

5. Sign off your commits (`git commit --signoff`) and push to your fork.
6. Open a pull request against `main`. Fill in the PR template, include
   any context a reviewer would need that isn't obvious from the diff,
   and link the issue your PR resolves if there is one.

For larger features (a new integration, a substantial refactor, a new
integration), please open an issue first to discuss the design. This
saves everyone time and gives you a chance to surface trade-offs before
investing implementation effort.

## Code review and merge

- Every pull request requires CI to pass and at least one approving
  review from a maintainer or designated CODEOWNER for the touched area.
- For changes that span multiple subsystems, expect reviews from each
  affected CODEOWNER.
- We squash-merge to keep `main`'s history readable. The PR title and
  description become the squash commit message — please make them
  descriptive and reviewer-facing.
- `main` is gated by GitHub's **merge queue**. Once your PR is approved
  and CI is green, click "Merge when ready" — GitHub will rebase your
  branch on top of `main` plus any earlier queued PRs, re-run the
  required checks against that combined state, and only land the merge
  if everything is still green. You do not need to manually rebase or
  re-run CI when another PR lands first. PRs that would conflict or
  fail after rebase are kicked back out of the queue automatically.
  Maintainers: do not enable "Require branches to be up to date before
  merging" alongside the queue — they're redundant, and enabling both
  reintroduces the rebase-storm problem the queue exists to solve.
- If a review comment is unclear, ask. We'd rather have a 30-second
  clarifying exchange than a misunderstanding turning into rework.

### Contributor expectations

We welcome community pull requests. Review timing depends on project
priorities and maintainer availability. Contributions should be
useful, tested, and easy to maintain. Contributors are responsible for
understanding and checking all code they submit, including generated
code, and providing relevant tests or steps to verify the change.
Maintainers should provide an initial review and respond to updates
within 7 calendar days. Contributors should respond to review feedback
within 7 calendar days. If either side needs more time, they should
say so in the PR. Maintainers may close PRs that remain inactive after
a reminder or do not meet the project’s needs or quality standards.
Before closing a public issue or PR, maintainers will check whether
users or documentation reference it. An issue or PR should not be
closed simply because it seems unnecessary. When closing, maintainers
will explain the reason and point to an alternative approach or
tracking issue where available.

## File Tree Of FlashDreams

Where a new app or model goes:

```text
apps/                           # reusable, model-independent applications
integrations_v2/                # model packages and application adapters
docs/source/demos/              # canonical demo articles
docs/source/models/             # canonical model articles
```

The framework package, and the test and doc trees:

```text
flashdreams/flashdreams/        # the framework package
  core/                         # numerical primitives, checkpoint loading, attention, I/O
  infra/                        # framework contracts: configs, pipelines, encoders/decoders, schedulers, runners
  recipes/                      # built-in reusable recipe code (WAN, Cosmos, TAEHV, ...)
  api_v2/                       # protocols an application implements
  runtime_v2/                   # the two-thread loop that runs an application
  runtime/                      # experimental inference runtime API envelope (v0, pre-v2)
  serving/                      # optional serving utilities (WebRTC, network, launch)
  demo/                         # transport-neutral application hosting and I/O API
  accelerated/                  # accelerated kernels (quantization, multi-head attention)
  quality/                      # output-quality regression utilities (video, CLIP compare)
  configs/                      # runner registry and CLI aggregator
  plugins/                      # external-runner plugin layer (RunnerConfig discovery)
  scripts/                      # console-script entry points (flashdreams-run)
  _pytest_plugins/              # pytest plugins (e.g. CI-tier marker enforcement)

flashdreams/test_v2/            # FlashDreams Runtime/Protocol tests (window, run_session, threads)
flashdreams/tests/              # framework tests not yet migrated to test_v2/
tests/                          # repo-wide test-runner scripts + meta checks, not package tests
docs/source/                    # Zensical Markdown sources
```

## Adding documentation

Write documentation for users and contributors as a website article under
`docs/source/`, not as a package `README.md`. Update an existing article when
it already owns the topic. Otherwise, add a Markdown file in the section that
owns it:

| Article | Website source | Example |
| --- | --- | --- |
| Install or first run | `docs/source/quickstart/` | `docs/source/quickstart/index.md` |
| Model requirements, setup, commands, or limitations | `docs/source/models/` | `docs/source/models/my_model.md` |
| Demo controls, options, or behavior | `docs/source/demos/` | `docs/source/demos/my_demo.md` |
| API concept or development guide | `docs/source/documentation/<area>/guides/` | `docs/source/documentation/demo_api/guides/my_topic.md` |
| API reference | `docs/source/documentation/<area>/api_reference/` | `docs/source/documentation/inferencing_api/api_reference/my_api.md` |
| Project or community process | `docs/source/community/` | `docs/source/community/my_process.md` |
| Common user failure | `docs/source/troubleshooting.md` | Add a focused section to the existing page |

For example, a new demo article starts at `docs/source/demos/my_demo.md`:

```markdown
---
title: 'My Demo'
---

<!-- SPDX-FileCopyrightText: Copyright (c) 2026 YOUR NAME. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

Explain what the demo does, how to launch it, its controls, and its options.
```

Register a new page in the `.nav.yml` beside it; for this example, add
`My Demo: my_demo.md` to `docs/source/demos/.nav.yml`. Use a lowercase
snake-case filename that matches the model or demo package when applicable.

Files outside `docs/source/` should link to the published website article
instead of copying its instructions. `README.md` outside website, should try to point to the canonical website as much as possible.

Build or preview the site with the commands in
[Build Website](https://nvidia.github.io/flashdreams/main/documentation/tools/build_website/#build-the-website).

## Coding conventions

- Python 3.10+. Type-annotate new code; the project type-checks with
  [`ty`](https://docs.astral.sh/ty/).
- Formatting is enforced by `ruff` via pre-commit (`uv run pre-commit
  run -a`). The CI will reject unformatted code; running pre-commit
  locally is the easiest way to avoid surprises.
- Prefer small, well-named functions over long functions with comments
  explaining each block. Comments should explain *why*, not *what*.
- Tests live next to the thing they validate — see the File Tree Of
  FlashDreams above. Use `pytest` and prefer existing fixtures over
  hand-rolled setup. See
  [Testing](#testing) for discovery and marker requirements.
- Every source file added by a contribution must include this concise
  two-line SPDX header:

  ```python
  # SPDX-FileCopyrightText: Copyright (c) <YEAR> <COPYRIGHT HOLDER>
  # SPDX-License-Identifier: Apache-2.0
  ```

  Use the current calendar year and the name of the actual copyright
  holder, which may be the contributor, their organization, or NVIDIA.
  Preserve existing copyright notices when modifying a file; multiple
  `SPDX-FileCopyrightText` lines are allowed. The longer Apache-2.0
  boilerplate is accepted but not preferred.

## Testing

See [Testing](https://nvidia.github.io/flashdreams/main/documentation/tools/testing/) for test discovery,
CI tier markers, and commands.

## Dependency version bounds

The `flashdreams/pyproject.toml` declares minimum version bounds for all
runtime dependencies. These bounds reflect the oldest versions we believe
are compatible based on API analysis.

**CI tests run against the pinned versions in `uv.lock`**, not against
the declared minimums. This means:

- We guarantee correctness at the locked versions.
- We expect the package to work at the declared minimum bounds, but do
  not continuously validate this in CI.
- If you encounter breakage with a version that satisfies the declared
  bounds but differs from the lock file, please
  [open an issue](https://github.com/NVIDIA/flashdreams/issues). We will
  either fix compatibility or bump the bound in `pyproject.toml`.

## Working with a single integration package

See our [model gallery](https://nvidia.github.io/flashdreams/main/models/) page.

## Licensing of contributions

By submitting a pull request to this repository, you agree that your
contribution is licensed under the
[Apache License, Version 2.0](https://github.com/NVIDIA/flashdreams/blob/main/LICENSE), the same license under which
FlashDreams is distributed. The DCO sign-off described above is your
attestation that you have the right to make that grant.

Third-party code (i.e. code you did not write yourself, but that you
have the right to redistribute under a compatible license) may be
contributed only if:

1. its license is compatible with Apache-2.0;
2. its origin and license are clearly recorded in
   [`REUSE.toml`](https://github.com/NVIDIA/flashdreams/blob/main/REUSE.toml) and
   [`THIRD-PARTY-NOTICES`](https://github.com/NVIDIA/flashdreams/blob/main/THIRD-PARTY-NOTICES);
3. its files retain whatever attribution headers the upstream license
   requires.

If you are not sure whether something is contributable, please ask in
an issue before sending the code — it's much easier to sort out
upfront.

## Reporting issues

Use [GitHub Issues](https://github.com/NVIDIA/flashdreams/issues) to report
functional defects and to request improvements. Please do not include
confidential or customer information.

Do not file security vulnerabilities as public issues. Follow the coordinated
disclosure process in
[SECURITY.md](https://github.com/NVIDIA/flashdreams/blob/main/SECURITY.md).

## Code of Conduct

This project follows the
[Code of Conduct](https://github.com/NVIDIA/flashdreams/blob/main/CODE_OF_CONDUCT.md).
By participating in this project — including issues, discussions, and
pull requests — you agree to abide by it. Please report concerns to the
maintainers via the address listed in the Code of Conduct.

---

Thanks again for contributing. The project is more useful, more correct,
and more interesting because outside contributors take the time to send
their work upstream. We appreciate it.
