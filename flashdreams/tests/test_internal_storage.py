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

import importlib

import pytest

from flashdreams.core.io import internal
from flashdreams.recipes.alpadreams import hf


@pytest.fixture(autouse=True)
def _scrub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Insulate from the developer's shell env."""
    monkeypatch.delenv(internal.INTERNAL_STORAGE_ENV_VAR, raising=False)


def test_use_internal_storage_default_unset() -> None:
    assert internal.use_internal_storage() is False


@pytest.mark.parametrize("value", ["1", "true", "True", "TRUE"])
def test_use_internal_storage_truthy(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(internal.INTERNAL_STORAGE_ENV_VAR, value)
    assert internal.use_internal_storage() is True


@pytest.mark.parametrize(
    "value", ["", "0", "false", "no", "off", "yes", "anything-else"]
)
def test_use_internal_storage_falsy(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Only ``"1"`` / ``"true"`` flip the toggle on; everything else
    (including "yes") leaves the public/HF default in place."""
    monkeypatch.setenv(internal.INTERNAL_STORAGE_ENV_VAR, value)
    assert internal.use_internal_storage() is False


def _reload(module_path: str):
    """Re-import a module so its ``AVAILABLE_*`` re-reads the env var.
    Doesn't work on alpadreams config (re-registers runners on import)."""
    return importlib.reload(importlib.import_module(module_path))


def test_omni_dreams_hf_org_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(hf.OMNI_DREAMS_HF_ORG_ENV_VAR, raising=False)

    assert hf.omni_dreams_hf_repo("omni-dreams-models") == ("nvidia/omni-dreams-models")


def test_omni_dreams_hf_org_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(hf.OMNI_DREAMS_HF_ORG_ENV_VAR, "nvidia-omni-dreams-lha")

    assert hf.omni_dreams_hf_repo("omni-dreams-models") == (
        "nvidia-omni-dreams-lha/omni-dreams-models"
    )
    assert hf.omni_dreams_hf_url(
        "omni-dreams-samples",
        "tree/main/data/single_view",
        repo_type="dataset",
    ) == (
        "https://huggingface.co/datasets/nvidia-omni-dreams-lha/"
        "omni-dreams-samples/tree/main/data/single_view"
    )


def test_wan_vae_paths_respect_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(internal.INTERNAL_STORAGE_ENV_VAR, raising=False)
    vae = _reload("flashdreams.recipes.wan.autoencoder.vae")
    for url in vae.AVAILABLE_WAN_VAE_CHECKPOINT_PATHS.values():
        assert url.startswith("https://huggingface.co/lightx2v/"), url

    monkeypatch.setenv(internal.INTERNAL_STORAGE_ENV_VAR, "1")
    vae = _reload("flashdreams.recipes.wan.autoencoder.vae")
    for url in vae.AVAILABLE_WAN_VAE_CHECKPOINT_PATHS.values():
        assert url.startswith("s3://flashdreams/"), url


def test_taehv_paths_respect_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(internal.INTERNAL_STORAGE_ENV_VAR, raising=False)
    taehv = _reload("flashdreams.recipes.taehv")
    for url in taehv.AVAILABLE_TAEHV_CHECKPOINT_PATHS.values():
        assert url.startswith("https://huggingface.co/lightx2v/"), url

    monkeypatch.setenv(internal.INTERNAL_STORAGE_ENV_VAR, "1")
    taehv = _reload("flashdreams.recipes.taehv")
    for url in taehv.AVAILABLE_TAEHV_CHECKPOINT_PATHS.values():
        assert url.startswith("s3://flashdreams/"), url


def test_alpadreams_paths_at_current_env() -> None:
    """Check the default/public alpadreams paths for the scrubbed test env.

    ``alpadreams.config`` can't be reloaded mid-test because importing it
    re-registers runners, and the autouse fixture clears the internal-storage
    env var before this test runs.
    """
    from flashdreams.recipes.alpadreams import config

    # Mirrored chunk2 -> HF; unmirrored slugs fall through to s3.
    assert config.AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS["1view-vae-chunk2"].startswith(
        "https://huggingface.co/nvidia/"
    )
    assert config.AVAILABLE_ALPADREAMS_CHECKPOINT_PATHS["1view-vae-chunk3"].startswith(
        "s3://"
    )
