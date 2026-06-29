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

"""Cheap checks for the ``omnidreams-prepare`` setup helper."""

from __future__ import annotations

import pytest
from omnidreams.config import AVAILABLE_OMNIDREAMS_CHECKPOINT_PATHS
from omnidreams.prepare import hf_prewarm_urls

pytestmark = pytest.mark.ci_cpu


def test_hf_prewarm_urls_includes_world_model_checkpoint() -> None:
    # Regression guard: this used to return () and the checkpoint 401'd lazily
    # at runtime instead of being staged here.
    urls = hf_prewarm_urls()

    assert any("omni-dreams-models" in url and url.endswith(".pt") for url in urls), (
        f"expected an omni-dreams-models checkpoint in {urls!r}"
    )


def test_hf_prewarm_urls_only_returns_hf_file_urls() -> None:
    urls = hf_prewarm_urls()

    assert urls
    assert all(url.startswith("https://huggingface.co/") for url in urls)
    assert "MISSING" not in urls
    assert len(urls) == len(set(urls))


def test_hf_prewarm_urls_match_available_checkpoint_paths() -> None:
    # Stays in lockstep with the recipe config, minus the "MISSING" sentinels.
    real_urls = {
        value
        for value in AVAILABLE_OMNIDREAMS_CHECKPOINT_PATHS.values()
        if value.startswith("https://huggingface.co/")
    }

    assert set(hf_prewarm_urls()) == real_urls
