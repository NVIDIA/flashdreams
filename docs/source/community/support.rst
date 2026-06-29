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

FlashDreams is an open-source project. Picking the right channel
up-front gets you a useful answer fastest.

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
      :link: discord
      :link-type: doc

      :doc:`Discord <discord>` is the venue for open-ended "how do I…"
      questions, sharing results, and office hours with maintainers and
      other users. The issue tracker with the ``question`` label is
      fine too.

   .. grid-item-card:: I have a feature idea
      :class-card: fd-feature
      :link: https://github.com/NVIDIA/flashdreams/issues/new

      Open an issue describing the use case, what you'd want the API
      to look like, and the trade-offs you can think of. For larger
      features, please discuss before sending a PR.

   .. grid-item-card:: I found a security issue
      :class-card: fd-feature
      :link: https://github.com/NVIDIA/flashdreams/blob/main/SECURITY.md

      Do **not** file as a public issue. Follow the coordinated
      disclosure process in ``SECURITY.md``.


Before you file an issue
------------------------

- **Search existing issues** (open and closed) for your error
  message or symptom. Most "is this a bug?" questions already have an
  answer.
- **Check the** :doc:`faq` **page.** If your question is there,
  great; if a related question is there, link to it in your issue.
- **Check the** :doc:`/troubleshooting` **page.** It lists common
  first-run failures (e.g. CUDA build mismatches, disk and cache limits,
  Hugging Face auth, GPU memory) with a likely
  cause and next step for each.
- **Confirm your version.** A bug fixed in ``main`` looks identical to
  a fresh bug if you're on an older tagged release. Reproduce against
  the latest ``main`` or note your version in the report.
- **Try with the smallest possible inputs.** A 5-minute repro on a
  single GPU is more actionable than a multi-node training job.

What makes a good bug report
----------------------------

- **What you ran.** The exact command, runner slug, or Python snippet.
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
**two business days** (see the :doc:`/community/index` guide for the
canonical statement). Issues have no formal service-level agreement.

Commercial support
-------------------

FlashDreams is offered as-is under the Apache-2.0 license. There is
no commercial support agreement attached to the open-source project.
