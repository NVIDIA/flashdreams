# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
