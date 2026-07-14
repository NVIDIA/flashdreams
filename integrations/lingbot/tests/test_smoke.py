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

"""Cheap import-time checks for the ``lingbot`` plugin."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import pytest
import tomli as tomllib
from lingbot import config as config_mod
from lingbot import runner as runner_mod
from lingbot.config import (
    LINGBOT_WORLD_V2_CHECKPOINT_PATH,
    PIPELINE_CONFIGS,
    PIPELINE_LINGBOT_WORLD_FAST,
    PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST,
    RUNNER_CONFIGS,
)
from lingbot.pipeline import LingbotWorldInferencePipelineConfig
from lingbot.runner import (
    EXAMPLE_DATA_AVAILABLE_IDXS,
    LingbotWorldRunner,
    LingbotWorldRunnerConfig,
    example_data_dirname,
)
from lingbot.transformer import (
    LINGBOT_WORLD_MIN_CHECKPOINT_FREE_GB,
    LingbotWorldTransformer,
    LingbotWorldTransformerConfig,
)

from flashdreams.infra.config import derive_config
from flashdreams.infra.runner import RunnerConfig

pytestmark = pytest.mark.ci_cpu

ENTRY_POINT_GROUP = "flashdreams.runner_configs"


def test_all_upstream_example_indices_are_available() -> None:
    """Accept every upstream example folder from ``00`` through ``05``."""
    assert EXAMPLE_DATA_AVAILABLE_IDXS == tuple(range(6))
    assert [example_data_dirname(idx) for idx in range(6)] == [
        "00",
        "01",
        "02",
        "03",
        "04",
        "05",
    ]


@pytest.mark.parametrize("example_idx", [3, 4])
def test_promptless_examples_skip_the_prompt_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    example_idx: int,
) -> None:
    """Skip ``prompt.txt`` for scenes without an upstream prompt."""
    downloads: list[tuple[str, Path, str]] = []

    def _record_download(url: str, *, cache_dir: Path, filename: str) -> None:
        downloads.append((url, cache_dir, filename))

    monkeypatch.setattr(runner_mod, "EXAMPLE_DATA_DIR_LOCAL", tmp_path)
    monkeypatch.setattr(runner_mod, "download_to_cache", _record_download)

    cache_dir = runner_mod.ensure_example_data_downloaded(
        is_rank_zero=True,
        example_idx=example_idx,
    )

    assert cache_dir == tmp_path / f"{example_idx:02d}"
    assert [filename for _, _, filename in downloads] == [
        "image.jpg",
        "poses.npy",
        "intrinsics.npy",
    ]
    assert all(download_cache == cache_dir for _, download_cache, _ in downloads)
    assert all(f"/{example_idx:02d}/" in url for url, _, _ in downloads)


def test_promptless_example_resolves_to_empty_string(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use an empty model prompt when the example provides no prompt."""
    (tmp_path / "prompt.txt").write_text("stale fallback prompt", encoding="utf-8")
    runner = object.__new__(LingbotWorldRunner)
    runner_config = cast(
        LingbotWorldRunnerConfig,
        derive_config(
            RUNNER_CONFIGS["lingbot-world-v2-14b-causal-fast"],
            prompt="",
            prompt_path=None,
            example_idx=3,
        ),
    )
    runner.config = runner_config
    runner.is_rank_zero = True
    monkeypatch.setattr(
        runner_mod,
        "ensure_example_data_downloaded",
        lambda **_kwargs: tmp_path,
    )
    warnings: list[str] = []
    monkeypatch.setattr(
        runner_mod.logger,
        "warning",
        lambda message, *args: warnings.append(message.format(*args)),
    )

    runner._fill_example_data_defaults()

    assert runner_config.prompt_path is None
    assert runner._resolve_prompt() == ""
    assert warnings == [
        "LingBot prompt.txt is missing; proceeding with an empty prompt."
    ]


def test_runners_dict_is_non_empty() -> None:
    """Plugin must expose at least one runner."""
    assert RUNNER_CONFIGS, "RUNNER_CONFIGS is empty"


def test_runner_name_mirrors_pipeline_name() -> None:
    """``runner_name`` must equal ``pipeline.name`` per the CLI contract."""
    drifted = {
        slug: (cfg.runner_name, cfg.pipeline.name)
        for slug, cfg in RUNNER_CONFIGS.items()
        if cfg.runner_name != cfg.pipeline.name
    }
    assert not drifted, f"runner_name != pipeline.name: {drifted}"


def test_runners_have_descriptions() -> None:
    """Every shipped runner needs a non-empty CLI description."""
    empty = [
        slug for slug, cfg in RUNNER_CONFIGS.items() if not cfg.description.strip()
    ]
    assert not empty, f"runners missing description: {empty}"


def test_lingbot_configs_carry_documented_checkpoint_disk_requirement() -> None:
    """All LingBot checkpoints should preflight the documented first-run budget."""
    for cfg in RUNNER_CONFIGS.values():
        transformer = cfg.pipeline.diffusion_model.transformer
        assert isinstance(transformer, LingbotWorldTransformerConfig)
        assert (
            transformer.checkpoint_min_free_gb == LINGBOT_WORLD_MIN_CHECKPOINT_FREE_GB
        )


def test_v2_only_replaces_the_v1_checkpoint() -> None:
    """Derive the v2 model by replacing only the v1 checkpoint and slug."""
    expected = derive_config(
        PIPELINE_LINGBOT_WORLD_FAST,
        name="lingbot-world-v2-14b-causal-fast",
        diffusion_model=dict(
            transformer=dict(checkpoint_path=LINGBOT_WORLD_V2_CHECKPOINT_PATH),
        ),
    )

    assert PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST == expected


@pytest.mark.parametrize(
    "slug",
    ["lingbot-world-fast", "lingbot-world-v2-14b-causal-fast"],
)
def test_model_versions_share_text_event_capable_pipeline(slug: str) -> None:
    """Expose the same text encoder and transformer runtime for v1 and v2."""
    pipeline = PIPELINE_CONFIGS[slug]
    assert isinstance(pipeline, LingbotWorldInferencePipelineConfig)
    assert pipeline.text_encoder is not None
    transformer = pipeline.diffusion_model.transformer
    assert isinstance(transformer, LingbotWorldTransformerConfig)
    assert transformer._target is LingbotWorldTransformer


def test_entry_points_match_module_literals() -> None:
    """The entry points in ``pyproject.toml`` must resolve to module attrs.

    Catches the common drift where someone adds a runner literal but
    forgets to wire it into the entry-point group (or vice versa);
    discovery would silently miss the new slug at the user's terminal.
    """
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as fh:
        meta = tomllib.load(fh)
    entries = meta["project"]["entry-points"][ENTRY_POINT_GROUP]
    declared_slugs = set(entries)
    module_slugs = set(RUNNER_CONFIGS)
    assert declared_slugs == module_slugs, (
        f"entry-point slugs ({sorted(declared_slugs)}) "
        f"!= module runners ({sorted(module_slugs)})"
    )

    for slug, target in entries.items():
        module_name, attr = target.split(":", 1)
        # Resolve the entry-point target the same way importlib.metadata
        # would, but skip the actual ``entry_points()`` call so the test
        # passes even when the plugin isn't pip-installed yet.
        assert module_name == "lingbot.config", (
            f"unexpected module in entry point {slug!r}: {module_name}"
        )
        cfg = cast(RunnerConfig, getattr(config_mod, attr))
        assert cfg.runner_name == slug, (
            f"entry point {slug!r} -> {attr} resolves to "
            f"runner_name={cfg.runner_name!r}"
        )


@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="entry-point discovery test relies on ``importlib.metadata`` 3.10+ shape",
)
def test_entry_points_discoverable_when_installed() -> None:
    """``importlib.metadata.entry_points`` finds the plugin's slugs.

    Requires the package to be installed (``uv sync`` from the repo
    root suffices since the plugin is a workspace member). Skipped
    automatically when running from a clean checkout. This is the
    integration check that mirrors what ``flashdreams-run``'s
    discovery layer actually does.
    """
    from importlib.metadata import entry_points

    eps = entry_points(group=ENTRY_POINT_GROUP)
    discovered = {ep.name for ep in eps if ep.value.startswith("lingbot.")}
    if not discovered:
        pytest.skip("plugin not installed; run `uv sync` from the repo root first")
    assert discovered == set(RUNNER_CONFIGS), (
        f"discovered slugs ({sorted(discovered)}) != "
        f"plugin runners ({sorted(RUNNER_CONFIGS)})"
    )
