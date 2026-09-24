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

"""CPU checks for preload static analysis."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

from flashdreams.runtime_v2 import cli

pytestmark = pytest.mark.ci_cpu


def test_preload_static_analysis_finds_application_and_framework_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_path = tmp_path / "app.py"
    app_path.write_text(
        "import huggingface_hub as hub\n"
        "def prepare():\n"
        "    hub.snapshot_download('org/model')\n",
        encoding="utf-8",
    )
    dependency_path = tmp_path / "dependency.py"
    dependency_path.write_text(
        "from urllib.request import urlretrieve as download\n"
        "def prepare():\n"
        "    download('https://example.invalid/file')\n",
        encoding="utf-8",
    )
    framework_path = tmp_path / "framework.py"
    framework_path.write_text(
        "import subprocess\nsubprocess.Popen(['prepare'])\n",
        encoding="utf-8",
    )
    modules = {
        "example_app": ModuleType("example_app"),
        "example_dependency": ModuleType("example_dependency"),
        "flashdreams.example_framework": ModuleType("flashdreams.example_framework"),
    }
    modules["example_app"].__file__ = str(app_path)
    modules["example_dependency"].__file__ = str(dependency_path)
    modules["flashdreams.example_framework"].__file__ = str(framework_path)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    with pytest.warns(RuntimeWarning) as recorded:
        cli._run_preload_static_analysis(set(modules))

    assert [str(item.message) for item in recorded] == [
        "huggingface_hub.snapshot_download may perform preparation outside "
        "IApplication.init",
        "urllib.request.urlretrieve may perform preparation outside IApplication.init",
        "subprocess.Popen may perform preparation outside IApplication.init",
    ]
