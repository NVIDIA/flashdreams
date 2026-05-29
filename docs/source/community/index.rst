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

   .. rubric:: Community and contribution
      :class: fd-hero-title

   .. container:: fd-hero-lede

      FlashDreams is developed in the open at
      `NVIDIA/flashdreams <https://github.com/NVIDIA/flashdreams>`__.
      This section collects the channels for filing bugs, asking
      questions, sending pull requests, and tracking releases.

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

Maintainers monitor the GitHub issue tracker first. Discord is the
venue for real-time conversation with maintainers and other users.

.. container:: fd-media-rail fd-media-rail-2

   .. container:: fd-media-tile

      .. container:: fd-media-tile-body

         .. container:: fd-media-tile-title

            GitHub Issues

         File bug reports and feature requests. This is the fastest way
         to reach the maintainers with something concrete and
         reproducible.

         `Open an issue
         <https://github.com/NVIDIA/flashdreams/issues>`__

   .. container:: fd-media-tile

      .. container:: fd-media-tile-body

         .. container:: fd-media-tile-title

            Discord

         Real-time chat with maintainers and other users. The
         FlashDreams community lives on the NVIDIA Omniverse server.

         `Join the Discord
         <https://discord.com/invite/nvidiaomniverse>`__

Contributing
------------

.. container:: fd-split fd-split-asymmetric-reverse

   .. container:: fd-split-text

      The contribution flow is documented in full on the
      :doc:`contributing` page. The five steps on the side are the
      short version.

      The authoritative version — DCO details, review expectations,
      CI tier markers, the SPDX header template — lives in
      :doc:`contributing` and the canonical `CONTRIBUTING.md
      <https://github.com/NVIDIA/flashdreams/blob/main/CONTRIBUTING.md>`__
      in the repo root.

   .. container:: fd-split-visual

      .. container:: fd-info-card

         .. container:: fd-info-card-title

            Five-step flow

         1. **Fork** ``NVIDIA/flashdreams`` and create a feature
            branch off ``main``.
         2. **Set up** your local environment — see the
            :doc:`/quickstart/index`.
         3. **Run** the linters and CPU-tier tests before you push;
            the :doc:`contributing` page has the exact commands.
         4. **Sign off** every commit with ``git commit --signoff``
            (DCO is a hard gate).
         5. **Open a PR** against ``main`` and fill in the template.

Code of conduct
---------------

Participation in the project is bound by the
`NVIDIA Open Source Code of Conduct <https://github.com/NVIDIA/.github/blob/main/CODE_OF_CONDUCT.md>`__.
By participating — issues, discussions, pull requests, chat — you
agree to abide by it. Concerns can be reported to the maintainers via
the address listed in that document.

Maintainers
-----------

The repository is maintained by the team that owns admin rights,
``main`` branch protections, and the package publishing keys. The
path to becoming a maintainer is sustained, high-quality contribution
in an area — see the *Project governance* section of `CONTRIBUTING.md
<https://github.com/NVIDIA/flashdreams/blob/main/CONTRIBUTING.md#project-governance>`__
for the full statement.

Releases
--------

FlashDreams follows semantic versioning. CI currently publishes
wheels to `Test PyPI <https://test.pypi.org/project/flashdreams/>`__
on every push to ``main``; once the package graduates to real PyPI,
tagged release notes will appear at `GitHub Releases
<https://github.com/NVIDIA/flashdreams/releases>`__.

The documentation site is configured for version-pinned hosting once
releases land: release builds will be archived under
``flashdreams.org/versions/<x.y.z>/`` alongside the rolling ``main``
build, with a switcher in the navbar. There are no release tags in
the repo today, so the per-version layout is configured but not yet
populated.

Frequently asked
----------------

Answers to questions that come up repeatedly in the issue tracker
live on the :doc:`faq` page. If a question keeps recurring in support
channels, propose an FAQ entry via a pull request — see
:doc:`contributing`.

.. rst-class:: fd-band-accent fd-cta-banner

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
