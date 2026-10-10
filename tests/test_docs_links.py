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

"""Check local media and script paths in the documentation sources."""

import re
from pathlib import Path
from urllib.parse import urlsplit

import pytest

pytestmark = pytest.mark.ci_cpu

_DOCS_SOURCE = Path(__file__).parents[1] / "docs" / "source"
_SRC_RE = re.compile(r'\b(?:src|poster)="([^"]+)"')


def test_local_html_assets_resolve_from_source() -> None:
    missing = []
    for page in _DOCS_SOURCE.rglob("*.md"):
        for reference in _SRC_RE.findall(page.read_text(encoding="utf-8")):
            url = urlsplit(reference)
            if url.scheme or url.netloc or url.path.startswith("/"):
                continue
            if not (page.parent / url.path).resolve().is_file():
                missing.append(f"{page.relative_to(_DOCS_SOURCE)}: {reference}")

    assert not missing, "Missing local documentation assets:\n" + "\n".join(missing)
