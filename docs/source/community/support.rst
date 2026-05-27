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

Getting help
============

FlashDreams is an open-source project with a small maintainer team.
Picking the right channel up-front gets you a useful answer fastest.

Choose a channel
----------------

.. grid:: 1 2 2 2
   :gutter: 3
   :margin: 0 0 4 0

   .. grid-item-card:: I think I found a bug
      :class-card: fd-feature
      :link: https://github.com/NVIDIA/flashdreams/issues

      File a GitHub issue with the smallest reproducer you can manage.
      See the checklist below for what makes a bug report easy to act
      on.

   .. grid-item-card:: I have a question about how to do X
      :class-card: fd-feature

      Use GitHub Discussions or Discord for open-ended "how do I…"
      questions.

   .. grid-item-card:: I have a feature idea
      :class-card: fd-feature
      :link: https://github.com/NVIDIA/flashdreams/issues/new

      Open an issue describing the use case, what you'd want the API
      to look like, and the trade-offs you can think of. For larger
      features, please discuss before sending a PR.

   .. grid-item-card:: I found a security issue
      :class-card: fd-feature
      :link: https://www.nvidia.com/en-us/security/

      Do **not** file as a public issue. Follow NVIDIA's coordinated
      disclosure process.

.. admonition:: PLACEHOLDER — GitHub Discussions link
   :class: placeholder

   **What goes here:** Once Discussions is enabled on the repo, link
   the "I have a question" card above to
   ``https://github.com/NVIDIA/flashdreams/discussions``.


.. admonition:: PLACEHOLDER — Discord invite
   :class: placeholder

   **What goes here:** Replace this with a permanent Discord invite
   URL once the server is live, and link the "I have a question" card
   to it.


Before you file an issue
------------------------

A 30-second check saves everyone time:

- **Search existing issues** — open and closed — for your error
  message or symptom. Most "is this a bug?" questions already have an
  answer.
- **Check the** :doc:`faq` **page.** If your question is there,
  great; if a related question is there, link to it in your issue.
- **Confirm your version.** A bug fixed in ``main`` looks identical to
  a fresh bug if you're on an older tagged release. Reproduce against
  the latest ``main`` or note your version in the report.
- **Try with the smallest possible inputs.** A 5-minute repro on a
  single GPU is more actionable than a multi-node training job.

What makes a good bug report
----------------------------

.. admonition:: Aim for a copy-pastable reproducer.
   :class: fd-callout

   The best bug reports are the ones a maintainer can run, see fail,
   and start debugging in under five minutes.

Please include:

- **What you ran.** The exact command, recipe name, or Python snippet.
- **What you expected.** One sentence.
- **What you saw.** Full stack trace or output. Wrap it in a code
  fence; don't paste a screenshot of text.
- **Your environment.** Python version, CUDA version, GPU model,
  FlashDreams version (``python -c "import flashdreams;
  print(flashdreams.__version__)"``), and how you installed it
  (workspace checkout, ``pip install``, container image).
- **What you've already tried.** Workarounds, related issues, debug
  prints — any of these speed up triage.

Response times
--------------

The maintainers aim for a first review on every PR within
**two business days** (see ``CONTRIBUTING.md`` for the canonical
statement). Issues have no formal SLA — the project is small and
volunteer-staffed, so polite pings on quiet threads are welcome.

Triage and labels
-----------------

.. admonition:: PLACEHOLDER — label vocabulary
   :class: placeholder

   **What goes here:** the actual label set on
   ``https://github.com/NVIDIA/flashdreams/labels`` (no
   ``.github/labels.yml`` in the repo today, so the live taxonomy is
   the source of truth). A *suggested* starter taxonomy if the project
   doesn't yet have one: ``bug``, ``enhancement``, ``question``,
   ``good first issue``, ``needs-info``, ``perf``.


Commercial / NVIDIA-internal support
------------------------------------

FlashDreams is offered as-is under the Apache-2.0 license. There is
no commercial support agreement attached to the open-source project.
NVIDIA-internal users with production deployment needs should contact
their usual NVIDIA solutions architect rather than the public issue
tracker.

.. admonition:: PLACEHOLDER — NVIDIA-internal contact
   :class: placeholder

   **What goes here:** If there is a specific internal mailing list or
   Slack channel for NVIDIA employees deploying FlashDreams, link it
   here. Otherwise, remove this section.

