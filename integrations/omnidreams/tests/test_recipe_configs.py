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

"""Cheap import-time checks for the ``omnidreams`` plugin.

The full numerics / GPU tests live alongside this file (they need GPU
+ checkpoints). These smoke tests just confirm the plugin is wired
correctly: importable, public slugs map to their intended internal
pipeline presets, descriptions are non-empty, and the
entry-point declarations in ``pyproject.toml`` match the
``omnidreams.config`` ``RUNNER_*`` literals exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import pytest
import tomli as tomllib
from omnidreams import config as config_mod
from omnidreams.config import OMNIDREAMS_RUNNERS
from omnidreams.transformer import CosmosTransformerConfig

from flashdreams.infra.runner import RunnerConfig
from integrations.omnidreams.benchmarks.cases import BENCHMARK_CASES

pytestmark = pytest.mark.ci_cpu

ENTRY_POINT_GROUP = "flashdreams.runner_configs"


def test_runners_dict_is_non_empty() -> None:
    """Plugin must expose at least one runner."""
    assert OMNIDREAMS_RUNNERS, "OMNIDREAMS_RUNNERS is empty"


def test_public_runner_slugs_map_to_internal_pipeline_presets() -> None:
    """Short public slugs must keep selecting the intended model presets."""
    expected = {
        "omnidreams": "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae",
        "omnidreams-perf": "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf",
        "omnidreams-optimized-gb300": (
            "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-optimized-gb300"
        ),
        "omnidreams-optimized-rtx-pro-6000": (
            "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-optimized-rtx-pro-6000"
        ),
        "omnidreams-cuda-cudnn": (
            "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-cuda-cudnn"
        ),
        "omnidreams-cuda-sparge": (
            "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-cuda-sparge"
        ),
        "omnidreams-cuda-sage3": (
            "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-cuda-sage3"
        ),
        "omnidreams-cuda-sage3fp8": (
            "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-cuda-sage3-fp8"
        ),
    }
    actual = {slug: cfg.pipeline.name for slug, cfg in OMNIDREAMS_RUNNERS.items()}
    assert actual == expected
    assert all(slug == cfg.runner_name for slug, cfg in OMNIDREAMS_RUNNERS.items())


def test_accelerated_runners_skip_finalize_kv_cache() -> None:
    """Only accelerated public runners skip the finalize cache refresh."""
    skipped = set()
    for slug, cfg in OMNIDREAMS_RUNNERS.items():
        transformer = cfg.pipeline.diffusion_model.transformer
        assert isinstance(transformer, CosmosTransformerConfig)
        if transformer.skip_finalize_kv_cache:
            skipped.add(slug)

    assert skipped == {
        "omnidreams-perf",
        "omnidreams-optimized-gb300",
        "omnidreams-optimized-rtx-pro-6000",
        "omnidreams-cuda-cudnn",
        "omnidreams-cuda-sparge",
        "omnidreams-cuda-sage3",
        "omnidreams-cuda-sage3fp8",
    }


def test_accelerated_runner_configs_match_benchmark_cases() -> None:
    """Public acceleration presets must mirror the benchmark matrix."""
    implementation_by_runner = {
        "omnidreams-optimized-gb300": (
            "optimized_cudnn_fp8_self_full_no_tma_cross_none_tma"
        ),
        "omnidreams-optimized-rtx-pro-6000": (
            "optimized_fa2_quantized_sdpa_self_full_tma_cross_none_tma"
        ),
        "omnidreams-cuda-cudnn": "cuda",
        "omnidreams-cuda-sparge": "cuda_sparge",
        "omnidreams-cuda-sage3": "cuda_sage3",
        "omnidreams-cuda-sage3fp8": "cuda_sage3_fp8",
    }
    cases = {case.implementation: case for case in BENCHMARK_CASES}

    for runner_name, implementation in implementation_by_runner.items():
        transformer = OMNIDREAMS_RUNNERS[
            runner_name
        ].pipeline.diffusion_model.transformer
        assert isinstance(transformer, CosmosTransformerConfig)
        network = transformer.network
        case = cases[implementation]

        assert network.self_attention_backend is case.self_attention_backend
        assert network.cross_attention_backend is case.cross_attention_backend
        assert (
            network.self_attn_optimized_impl_config
            == case.self_attn_optimized_impl_config
        )
        assert (
            network.cross_attn_optimized_impl_config
            == case.cross_attn_optimized_impl_config
        )

        if case.native_dit:
            assert transformer.native_dit_acceleration == "required"
            assert transformer.native_dit_backend == case.native_dit_backend
            assert (
                transformer.native_dit_attention_backend
                == case.native_attention_backend
            )
        else:
            assert transformer.native_dit_acceleration == "disabled"


def test_runners_have_descriptions() -> None:
    """Every shipped runner needs a non-empty CLI description."""
    empty = [
        slug for slug, cfg in OMNIDREAMS_RUNNERS.items() if not cfg.description.strip()
    ]
    assert not empty, f"runners missing description: {empty}"


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
    module_slugs = set(OMNIDREAMS_RUNNERS)
    assert declared_slugs == module_slugs, (
        f"entry-point slugs ({sorted(declared_slugs)}) "
        f"!= module runners ({sorted(module_slugs)})"
    )

    for slug, target in entries.items():
        module_name, attr = target.split(":", 1)
        # Resolve the entry-point target the same way importlib.metadata
        # would, but skip the actual ``entry_points()`` call so the test
        # passes even when the plugin isn't pip-installed yet.
        assert module_name == "omnidreams.config", (
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
    discovered = {ep.name for ep in eps if ep.value.startswith("omnidreams.")}
    if not discovered:
        pytest.skip("plugin not installed; run `uv sync` from the repo root first")
    assert discovered == set(OMNIDREAMS_RUNNERS), (
        f"discovered slugs ({sorted(discovered)}) != "
        f"plugin runners ({sorted(OMNIDREAMS_RUNNERS)})"
    )
