# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import pytest
from omnidreams import hf_org


@pytest.fixture(autouse=True)
def _scrub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with the env var unset so the precedence checks
    are insulated from whatever the developer has in their shell."""
    monkeypatch.delenv(hf_org.ENV_VAR, raising=False)


def test_resolve_default() -> None:
    assert hf_org.resolve_hf_org() == hf_org.DEFAULT_HF_ORG == "nvidia"


def test_resolve_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(hf_org.ENV_VAR, "example-omni-dreams-mirror")
    assert hf_org.resolve_hf_org() == "example-omni-dreams-mirror"


def test_resolve_cli_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(hf_org.ENV_VAR, "example-omni-dreams-mirror")
    assert hf_org.resolve_hf_org(cli_value="some-other-org") == "some-other-org"


def test_resolve_cli_alone() -> None:
    assert hf_org.resolve_hf_org(cli_value="some-other-org") == "some-other-org"


def test_hf_repo_default() -> None:
    assert hf_org.hf_repo(kind="scenes") == "nvidia/omni-dreams-scenes"


def test_hf_repo_explicit_org() -> None:
    assert (
        hf_org.hf_repo(kind="scenes", org="example-omni-dreams-mirror")
        == "example-omni-dreams-mirror/omni-dreams-scenes"
    )


def test_hf_repo_invalid_kind() -> None:
    with pytest.raises(ValueError, match="unknown omni-dreams repo kind"):
        hf_org.hf_repo(kind="bogus")  # type: ignore[arg-type]


def test_apply_cli_to_env_writes_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    assert hf_org.ENV_VAR not in os.environ
    result = hf_org.apply_cli_to_env("example-omni-dreams-mirror")
    assert result == "example-omni-dreams-mirror"
    assert os.environ[hf_org.ENV_VAR] == "example-omni-dreams-mirror"


def test_apply_cli_to_env_falls_through_to_default() -> None:
    import os

    result = hf_org.apply_cli_to_env(None)
    assert result == "nvidia"
    assert os.environ[hf_org.ENV_VAR] == "nvidia"


def test_rewrite_url_no_op_for_default_org() -> None:
    text = (
        "https://huggingface.co/nvidia/omni-dreams-scenes/resolve/main/foo.usdz\n"
        "https://huggingface.co/nvidia/Cosmos-Reason1-7B\n"
    )
    assert hf_org.rewrite_omni_dreams_urls(text, org="nvidia") == text


def test_rewrite_url_swaps_omni_dreams_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Other HF URLs are NOT omni-dreams scene mirrors and must pass through untouched."""
    monkeypatch.setenv(hf_org.ENV_VAR, "example-omni-dreams-mirror")
    yaml_text = (
        "checkpoint_path_dir: https://huggingface.co/nvidia/omni-dreams-models/resolve/main/foo.pt\n"
        "scenes_repo_url: https://huggingface.co/nvidia/omni-dreams-scenes\n"
        "reason1: https://huggingface.co/nvidia/Cosmos-Reason1-7B\n"
    )
    out = hf_org.rewrite_omni_dreams_urls(yaml_text)
    # The scene repo flips to the selected org.
    assert "example-omni-dreams-mirror/omni-dreams-scenes" in out
    # Unrelated URLs untouched.
    assert "nvidia/omni-dreams-models" in out
    assert "nvidia/Cosmos-Reason1-7B" in out


def test_rewrite_url_explicit_org_wins_over_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(hf_org.ENV_VAR, "env-org")
    text = "nvidia/omni-dreams-scenes"
    assert (
        hf_org.rewrite_omni_dreams_urls(text, org="cli-org")
        == "cli-org/omni-dreams-scenes"
    )


def test_rewrite_url_handles_word_boundaries() -> None:
    """Don't rewrite non-scene repos or substring matches across boundaries."""
    # Genuine match -- should rewrite.
    assert (
        hf_org.rewrite_omni_dreams_urls(
            "nvidia/omni-dreams-scenes foo", org="example-org"
        )
        == "example-org/omni-dreams-scenes foo"
    )
    # Looks similar but isn't an omni-dreams kind we know about.
    assert (
        hf_org.rewrite_omni_dreams_urls(
            "nvidia/omni-dreams-other foo", org="example-org"
        )
        == "nvidia/omni-dreams-other foo"
    )
    assert (
        hf_org.rewrite_omni_dreams_urls(
            "somethingnvidia/omni-dreams-scenes foo", org="example-org"
        )
        == "somethingnvidia/omni-dreams-scenes foo"
    )
