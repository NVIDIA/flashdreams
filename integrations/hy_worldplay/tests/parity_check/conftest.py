# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Pytest collection guard for the parity-check directory.

This directory hosts two things pytest must NOT recurse into:

* ``HY-WorldPlay/`` -- the cloned vendor tree. It contains internal
  ``test_*.py`` files (e.g.
  ``worldcompass/reward_function/HunyuanWorldMirror/test_setup.py``)
  that import vendor-internal deps (``gsplat``, ``sageattention``)
  which only live in the parity sub-venv. Letting pytest collect
  those from the main flashdreams venv breaks ``pytest
  integrations/hy_worldplay/tests/``.

* ``.venv/`` -- the parity sub-venv, with its own installed packages
  that show up as importable test modules under
  ``.venv/lib/python*/site-packages/``.

The CPU tests that exercise this directory (e.g. the phase 2b.6
``test_parity_helper.py`` for the ``use_kv_cache=True`` helper)
live one level up under ``integrations/hy_worldplay/tests/`` so
the main pytest discovery picks them up naturally.
"""

from __future__ import annotations

collect_ignore_glob = [
    "HY-WorldPlay/**",
    ".venv/**",
]
