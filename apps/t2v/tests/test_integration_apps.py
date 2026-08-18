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

from pathlib import Path

import pytest
import tomli as tomllib

pytestmark = pytest.mark.ci_cpu

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_t2v_app_installs_dependencies_for_documented_outputs() -> None:
    manifest = tomllib.loads(
        (_REPO_ROOT / "apps" / "t2v" / "pyproject.toml").read_text()
    )

    assert "flashdreams[local-window,serving]" in manifest["project"]["dependencies"]


@pytest.mark.parametrize(
    ("project", "module", "slug"),
    [
        ("cosmos_predict2", "cosmos_predict2", "t2v-cosmos-predict2"),
        ("causal_forcing", "causal_forcing", "t2v-causal-forcing"),
        (
            "fastvideo_causal_wan22",
            "fastvideo_causal_wan22",
            "t2v-fastvideo-causal-wan22",
        ),
        ("self_forcing", "self_forcing", "t2v-self-forcing"),
        ("wan21", "wan21", "t2v-wan21"),
    ],
)
def test_concrete_t2v_app_is_owned_by_integration(
    project: str,
    module: str,
    slug: str,
) -> None:
    project_dir = _REPO_ROOT / "integrations" / project
    app_path = project_dir / module / "t2v" / "app.py"
    manifest = tomllib.loads((project_dir / "pyproject.toml").read_text())

    assert app_path.is_file()
    assert manifest["project"]["entry-points"]["flashdreams.applications"][slug] == (
        f"{module}.t2v.app:create_app"
    )
    assert "flashdreams-t2v" in manifest["project"]["dependencies"]
