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

from __future__ import annotations

import pytest

from flashdreams.core.io import hf_org


@pytest.fixture(autouse=True)
def _scrub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with the env var unset so the precedence checks
    are insulated from whatever the developer has in their shell."""
    monkeypatch.delenv(hf_org.OMNI_DREAMS_HF_ORG_ENV_VAR, raising=False)


def test_resolve_default() -> None:
    assert (
        hf_org.resolve_omni_dreams_hf_org()
        == hf_org.DEFAULT_OMNI_DREAMS_HF_ORG
        == "nvidia"
    )


def test_resolve_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(hf_org.OMNI_DREAMS_HF_ORG_ENV_VAR, "nvidia-omni-dreams-lha")
    assert hf_org.resolve_omni_dreams_hf_org() == "nvidia-omni-dreams-lha"


def test_rewrite_no_op_for_default_org() -> None:
    """No env var set -> URL passes through unchanged."""
    url = (
        "https://huggingface.co/nvidia/omni-dreams-models/resolve/main/"
        "single_view/foo.pt"
    )
    assert hf_org.rewrite_omni_dreams_hf_url(url) == url


def test_rewrite_no_op_when_env_explicitly_nvidia(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Setting the env var to the default org is also a no-op."""
    monkeypatch.setenv(hf_org.OMNI_DREAMS_HF_ORG_ENV_VAR, "nvidia")
    url = "https://huggingface.co/nvidia/omni-dreams-models/resolve/main/foo.pt"
    assert hf_org.rewrite_omni_dreams_hf_url(url) == url


def test_rewrite_swaps_models_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """A flashdreams alpadreams DiT URL flips to the configured mirror."""
    monkeypatch.setenv(hf_org.OMNI_DREAMS_HF_ORG_ENV_VAR, "nvidia-omni-dreams-lha")
    url = (
        "https://huggingface.co/nvidia/omni-dreams-models/resolve/main/"
        "single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt"
    )
    expected = (
        "https://huggingface.co/nvidia-omni-dreams-lha/omni-dreams-models/resolve/main/"
        "single_view/2b_res720p_30fps_i2v_hdmap_distilled.pt"
    )
    assert hf_org.rewrite_omni_dreams_hf_url(url) == expected


def test_rewrite_swaps_scenes_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same hook also covers the omni-dreams scenes dataset, even
    though flashdreams doesn't fetch scenes itself -- exposing the
    behaviour means callers can route both kinds with one helper."""
    monkeypatch.setenv(hf_org.OMNI_DREAMS_HF_ORG_ENV_VAR, "nvidia-omni-dreams-lha")
    url = "https://huggingface.co/datasets/nvidia/omni-dreams-scenes/resolve/main/foo.usdz"
    expected = "https://huggingface.co/datasets/nvidia-omni-dreams-lha/omni-dreams-scenes/resolve/main/foo.usdz"
    assert hf_org.rewrite_omni_dreams_hf_url(url) == expected


def test_rewrite_passes_through_unrelated_nvidia_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """nvidia/* URLs that aren't omni-dreams must NOT be rewritten."""
    monkeypatch.setenv(hf_org.OMNI_DREAMS_HF_ORG_ENV_VAR, "nvidia-omni-dreams-lha")
    for url in (
        "https://huggingface.co/nvidia/Cosmos-Reason1-7B/resolve/main/config.json",
        "https://huggingface.co/nvidia/omni-dreams-other/resolve/main/foo.pt",
        "somethingnvidia/omni-dreams-models/foo",
    ):
        assert hf_org.rewrite_omni_dreams_hf_url(url) == url


def test_rewrite_handles_bare_repo_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pattern matches a bare ``nvidia/omni-dreams-*`` repo id too,
    not just full HF URLs -- callers may pass either."""
    monkeypatch.setenv(hf_org.OMNI_DREAMS_HF_ORG_ENV_VAR, "nvidia-omni-dreams-lha")
    assert (
        hf_org.rewrite_omni_dreams_hf_url("nvidia/omni-dreams-models")
        == "nvidia-omni-dreams-lha/omni-dreams-models"
    )
    assert (
        hf_org.rewrite_omni_dreams_hf_url("nvidia/omni-dreams-scenes")
        == "nvidia-omni-dreams-lha/omni-dreams-scenes"
    )
