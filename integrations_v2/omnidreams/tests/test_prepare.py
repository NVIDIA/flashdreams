# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cheap checks for the ``omnidreams-prepare`` setup helper."""

from __future__ import annotations

import pytest
from omnidreams.config import AVAILABLE_OMNIDREAMS_CHECKPOINT_PATHS
from omnidreams.impl.tools.prepare import hf_prewarm_urls

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
