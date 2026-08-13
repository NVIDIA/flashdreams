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

"""CPU tests for shared YAML pipeline-preset parsing and materialization."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from flashdreams.core.pipeline_presets import (
    ObjectGraphPipelineProvider,
    load_pipeline_preset_catalog,
    load_pipeline_provider,
    materialize_object_graph,
)

pytestmark = pytest.mark.ci_cpu


def _parse_runtime_options(value: object, *, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a mapping.")
    return dict(cast(Mapping[str, object], value))


def test_load_pipeline_preset_catalog_parses_generic_schema(tmp_path: Path) -> None:
    path = tmp_path / "presets.yaml"
    path.write_text(
        """
schema_version: 1
default_preset_id: example
presets:
  example:
    provider: flashdreams.core.pipeline_presets:ObjectGraphPipelineProvider
    runtime:
      fps: 12
    pipeline:
      enabled: true
""".strip(),
        encoding="utf-8",
    )

    catalog = load_pipeline_preset_catalog(
        path,
        runtime_options_parser=_parse_runtime_options,
    )
    preset_id, preset = catalog.resolve(None)

    assert preset_id == "example"
    assert preset.runtime == {"fps": 12}
    assert preset.pipeline == {"enabled": True}


def test_load_pipeline_preset_catalog_rejects_unknown_root_field(
    tmp_path: Path,
) -> None:
    path = tmp_path / "presets.yaml"
    path.write_text(
        """
schema_version: 1
default_preset_id: example
presets: {}
unexpected: true
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown.*unexpected"):
        load_pipeline_preset_catalog(
            path,
            runtime_options_parser=_parse_runtime_options,
        )


def test_object_graph_provider_materializes_shared_declarative_nodes() -> None:
    provider = load_pipeline_provider(
        "flashdreams.core.pipeline_presets:ObjectGraphPipelineProvider"
    )

    config = provider.create_pipeline_config(
        preset_id="example",
        options={
            "_target": "types:SimpleNamespace",
            "callback": {"_ref": "builtins:len"},
            "shape": {"_tuple": [1, 2, 3]},
        },
    )

    assert isinstance(provider, ObjectGraphPipelineProvider)
    assert isinstance(config, SimpleNamespace)
    assert config.callback is len
    assert config.shape == (1, 2, 3)


def test_materialize_object_graph_rejects_mixed_reference_node() -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        materialize_object_graph(
            {"_ref": "builtins:len", "other": True},
            path="preset.pipeline.callback",
        )
