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

.. container:: fd-hero fd-hero-band

   .. container:: fd-hero-eyebrow

      Apache-2.0, developed in the open

   .. rubric:: Build streaming diffusion with us.
      :class: fd-hero-title

   .. container:: fd-hero-lede

      FlashDreams is built in the open at
      `NVIDIA/flashdreams <https://github.com/NVIDIA/flashdreams>`__.
      File a bug, ask a question, send a pull request, or just keep
      up with releases — every channel that matters lives here.

   .. container:: fd-cta-row

      .. button-link:: https://github.com/NVIDIA/flashdreams
         :color: primary

         Browse the repo

      .. button-ref:: contributing
         :ref-type: doc
         :color: secondary
         :outline:

         Contribute

      .. button-ref:: support
         :ref-type: doc
         :color: secondary
         :outline:

         Get help

.. toctree::
   :hidden:
   :maxdepth: 1

   contributing
   support
   faq

Where to find us
----------------

.. container:: fd-eyebrow

   Pick the channel that matches the question

.. container:: fd-lede

   Maintainers monitor GitHub Issues and Discussions first; Discord
   is the place for real-time conversation; the mailing list is the
   slow lane for announcements.

.. container:: fd-media-rail fd-media-rail-4

   .. container:: fd-media-tile

      .. container:: fd-media-tile-body

         .. container:: fd-media-tile-title

            GitHub Issues

         File bug reports and feature requests. The fastest way to
         reach the maintainers with something concrete and
         reproducible.

         `Open an issue
         <https://github.com/NVIDIA/flashdreams/issues>`__

   .. container:: fd-media-tile

      .. admonition:: PLACEHOLDER — GitHub Discussions
         :class: placeholder

         **Tile content once enabled:**

         GitHub Discussions
         — open-ended questions, design proposals, show-and-tell.

         Confirm Discussions is enabled on
         ``NVIDIA/flashdreams`` and link to
         ``/discussions``.

   .. container:: fd-media-tile

      .. admonition:: PLACEHOLDER — Discord invite URL
         :class: placeholder

         **Tile content once approved:**

         Discord — real-time chat with maintainers and other users.

         Drop the permanent (non-expiring) FlashDreams Discord
         server invite URL into this tile.

   .. container:: fd-media-tile

      .. admonition:: PLACEHOLDER — mailing list / announcements
         :class: placeholder

         **Tile content once decided:**

         Email / mailing list — low-volume announcements (releases,
         security advisories).

         Either ship the subscribe URL, or remove this tile and
         re-balance the rail to a 3-up if the project relies on
         GitHub Releases + Discord only.

Contributing
------------

.. container:: fd-eyebrow

   New contributor? Start here

.. container:: fd-split fd-split-asymmetric-reverse

   .. container:: fd-split-text

      The flow is small, well-trodden, and documented in full on the
      :doc:`contributing` page. The five steps on the side are the
      short version — every one is the same step you'd take on any
      other Apache-licensed NVIDIA repo.

      The authoritative version of this flow — DCO details, review
      expectations, CI tier markers, the SPDX header template — lives
      in :doc:`contributing` and the canonical `CONTRIBUTING.md
      <https://github.com/NVIDIA/flashdreams/blob/main/CONTRIBUTING.md>`__
      in the repo root.

   .. container:: fd-split-visual

      .. container:: fd-info-card

         .. container:: fd-info-card-title

            Five-step flow

         1. **Fork** ``NVIDIA/flashdreams`` and create a feature
            branch off ``main``.
         2. **Set up** your local environment — see
            :doc:`/quickstart/installation`.
         3. **Run** the linters and CPU-tier tests before you push;
            the :doc:`contributing` page has the exact commands.
         4. **Sign off** every commit with ``git commit --signoff``
            (DCO is a hard gate).
         5. **Open a PR** against ``main`` and fill in the template.

Code of conduct
---------------

.. container:: fd-eyebrow

   Participation is bound by the NVIDIA OSS code of conduct

.. container:: fd-lede

   FlashDreams follows the
   `NVIDIA Open Source Code of Conduct <https://github.com/NVIDIA/.github/blob/main/CODE_OF_CONDUCT.md>`__.
   By participating — issues, discussions, pull requests, chat —
   you agree to abide by it. Concerns can be reported to the
   maintainers via the address listed in that document.

.. admonition:: PLACEHOLDER — project-local CODE_OF_CONDUCT.md
   :class: placeholder

   **What goes here:** Decide whether the project keeps deferring to
   the NVIDIA org-wide Code of Conduct, or adds a project-local
   ``CODE_OF_CONDUCT.md`` at the repo root (with a project-specific
   reporting address). If a local file is added, link to it here.

Maintainers
-----------

.. container:: fd-eyebrow

   Who owns the repo

.. container:: fd-lede

   FlashDreams is currently maintained by NVIDIA's Simulation &
   Imitation Learning group, which holds admin rights on the
   repository, the ``main`` branch protections, and the package
   publishing keys. The path to becoming a maintainer is sustained,
   high-quality contribution in an area — see the *Project
   governance* section of `CONTRIBUTING.md
   <https://github.com/NVIDIA/flashdreams/blob/main/CONTRIBUTING.md#project-governance>`__
   for the full statement.

.. admonition:: PLACEHOLDER — MAINTAINERS.md
   :class: placeholder

   **What goes here:** A ``MAINTAINERS.md`` file at the repo root
   that lists active maintainers (GitHub handle, area of ownership,
   contact preference) and the criteria for promotion.

Releases
--------

.. container:: fd-eyebrow

   Semver, Test PyPI today, real PyPI on the runway

.. container:: fd-lede

   FlashDreams follows semantic versioning. CI currently publishes
   wheels to `Test PyPI <https://test.pypi.org/project/flashdreams/>`__
   on every push to ``main``; once the package graduates to real
   PyPI, tagged release notes will appear at `GitHub Releases
   <https://github.com/NVIDIA/flashdreams/releases>`__.

The documentation site is set up for version-pinned hosting once
releases land: release builds will be archived under
``flashdreams.org/versions/<x.y.z>/`` alongside the rolling ``main``
build, with a switcher in the navbar. There are no release tags in
the repo today, so the per-version layout is on the runway rather
than in production.

.. admonition:: PLACEHOLDER — release cadence statement
   :class: placeholder

   **What goes here:** A short, public statement of the project's
   intended release cadence (e.g. "minor release every ~6 weeks,
   patch releases as needed"). Until that is decided, point readers
   to the Releases page above.

Frequently asked
----------------

.. container:: fd-eyebrow

   Recurring questions, all in one place

.. container:: fd-lede

   Answers to questions that come up repeatedly in issues, Discord, and
   Discussions live on the :doc:`faq` page. If a question keeps coming
   up in support channels, propose an FAQ entry via a pull request —
   see :doc:`contributing`.

.. rst-class:: fd-band-accent fd-cta-banner

Where to next
-------------

.. container:: fd-eyebrow

   Three doors

.. container:: fd-lede

   Pick the path that matches what you're here to do.

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
