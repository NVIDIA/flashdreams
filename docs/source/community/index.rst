.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
.. http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

Community
=========

FlashDreams is built in the open on
`GitHub <https://github.com/NVIDIA/flashdreams>`__ under the Apache-2.0
license. This section is the wayfinding hub for everyone who wants to
participate — whether that means filing a bug, asking a question,
sending a pull request, or just keeping up with releases.

.. toctree::
   :hidden:
   :maxdepth: 1

   contributing
   support
   faq

Channels
--------

.. grid:: 1 2 2 2
   :gutter: 3
   :margin: 0 0 4 0

   .. grid-item-card:: GitHub Issues
      :class-card: fd-feature
      :link: https://github.com/NVIDIA/flashdreams/issues

      File bug reports and feature requests. The fastest way to reach
      the maintainers about something concrete and reproducible.

   .. grid-item-card:: GitHub Discussions
      :class-card: fd-feature

      Open-ended questions, design proposals, and show-and-tell. See
      the placeholder below for the canonical link once Discussions is
      enabled on the repository.

   .. grid-item-card:: Discord
      :class-card: fd-feature

      Real-time chat with maintainers and other users. Invite link
      pending — see the placeholder below.

   .. grid-item-card:: Email / mailing list
      :class-card: fd-feature

      Low-volume announcements (releases, security advisories).
      Subscription details pending — see the placeholder below.

.. admonition:: PLACEHOLDER — Discord invite URL
   :class: placeholder

   **What goes here:** The permanent (non-expiring) Discord server
   invite URL for the FlashDreams community server. Replace the
   "Discord" card above with a ``:link:`` to that URL.

   **Owner:** project lead / community manager.
   **Tracking:** TBD.

.. admonition:: PLACEHOLDER — GitHub Discussions
   :class: placeholder

   **What goes here:** Confirmation that GitHub Discussions is enabled
   on ``NVIDIA/flashdreams`` and the canonical URL
   (``https://github.com/NVIDIA/flashdreams/discussions``). Once
   enabled, add ``:link: https://github.com/NVIDIA/flashdreams/discussions``
   to the Discussions card above.

   **Owner:** repo admin.
   **Tracking:** TBD.

.. admonition:: PLACEHOLDER — mailing list / announcement channel
   :class: placeholder

   **What goes here:** Either the subscribe URL for a low-volume
   announcement list, or a decision to remove this card. If the
   project will rely on GitHub Releases + Discord only, drop the
   "Email / mailing list" card and re-balance the grid to
   ``1 2 2 3``.

   **Owner:** project lead.
   **Tracking:** TBD.

Contributing
------------

.. admonition:: New contributor? Start here.
   :class: fd-callout

   The contribution flow is small and well-trodden:

   1. **Fork** ``NVIDIA/flashdreams`` and create a feature branch off
      ``main``.
   2. **Set up** your environment with ``uv sync --extra dev`` (see
      :doc:`contributing` for the full local-dev walkthrough).
   3. **Run** ``uv run pre-commit run -a`` and ``uv run pytest -m ci_cpu``
      before you push.
   4. **Sign off** every commit with ``git commit --signoff`` (DCO is a
      hard gate).
   5. **Open a PR** against ``main`` and fill in the template.

   For the full, authoritative version of this flow — DCO details,
   review expectations, CI tier markers, the SPDX header template —
   read :doc:`contributing` and the canonical
   `CONTRIBUTING.md <https://github.com/NVIDIA/flashdreams/blob/main/CONTRIBUTING.md>`__
   in the repo root.

Code of conduct
---------------

This project follows the
`NVIDIA Open Source Code of Conduct <https://github.com/NVIDIA/.github/blob/main/CODE_OF_CONDUCT.md>`__.
By participating — issues, discussions, pull requests, chat — you
agree to abide by it. Concerns can be reported to the maintainers via
the address listed in that document.

.. admonition:: PLACEHOLDER — project-local CODE_OF_CONDUCT.md
   :class: placeholder

   **What goes here:** Decide whether the project keeps deferring to
   the NVIDIA org-wide Code of Conduct, or adds a project-local
   ``CODE_OF_CONDUCT.md`` at the repo root (with a project-specific
   reporting address). If a local file is added, link to it here.

   **Owner:** project lead.
   **Tracking:** TBD.

Maintainers and governance
--------------------------

FlashDreams is currently maintained by NVIDIA's Simulation & Imitation
Learning group, which holds admin rights on the repository, the
``main`` branch protections, and the package publishing keys.
``CODEOWNERS`` is the source of truth for per-subsystem review
responsibility, and the path to becoming a maintainer is sustained,
high-quality contribution in an area — see the *Project governance*
section of
`CONTRIBUTING.md <https://github.com/NVIDIA/flashdreams/blob/main/CONTRIBUTING.md#project-governance>`__
for the full statement.

.. admonition:: PLACEHOLDER — MAINTAINERS.md
   :class: placeholder

   **What goes here:** A ``MAINTAINERS.md`` file at the repo root that
   lists active maintainers (GitHub handle, area of ownership, contact
   preference) and the criteria for promotion. Until that file exists,
   ``CODEOWNERS`` is the de-facto record.

   **Owner:** project lead.
   **Tracking:** TBD.

Releases and versioning
-----------------------

FlashDreams follows semantic versioning. The canonical version lives
in ``flashdreams/flashdreams/_version.py`` and is synced into every
integration package by a pre-commit hook. CI currently publishes
wheels to `Test PyPI <https://test.pypi.org/project/flashdreams/>`__
on every push to ``main`` (see ``DEV.md`` for the temporary-by-design
note); once the package graduates to real PyPI, tagged release notes
will appear at
`GitHub Releases <https://github.com/NVIDIA/flashdreams/releases>`__.

The documentation site is structured to support version-pinned
hosting once releases land: the ``doc.yml`` workflow already deploys
release builds to ``flashdreams.org/versions/<x.y.z>/`` alongside the
rolling ``main`` build, and ``versions/index.txt`` will enumerate
them. There are no release tags in the repo today, so this layout is
on the runway rather than in production.

.. admonition:: PLACEHOLDER — release cadence statement
   :class: placeholder

   **What goes here:** A short, public statement of the project's
   intended release cadence (e.g. "minor release every ~6 weeks,
   patch releases as needed"). Until that is decided, point readers
   to the Releases page above.

   **Owner:** project lead.
   **Tracking:** TBD.

Frequently asked questions
--------------------------

A growing list of FAQ entries lives on the :doc:`faq` page. The
current entries are placeholders that future maintainers will fill in
as questions recur on issues, Discord, and Discussions. If a question
keeps coming up in support channels, propose an FAQ entry via a pull
request — see :doc:`contributing`.

Where to next
-------------

.. container:: fd-cta-row

   .. button-ref:: contributing
      :ref-type: doc
      :color: primary

      Read the contribution guide

   .. button-ref:: support
      :ref-type: doc
      :color: secondary
      :outline:

      Get help

   .. button-link:: https://github.com/NVIDIA/flashdreams
      :color: secondary
      :outline:

      Browse the repo
