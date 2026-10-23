# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check local links and media paths in the documentation sources."""

import re
from pathlib import Path
from urllib.parse import urlsplit

import pytest

pytestmark = pytest.mark.ci_cpu

_DOCS_SOURCE = Path(__file__).parents[1] / "docs" / "source"
_REPO_ROOT = _DOCS_SOURCE.parents[1]
_MARKDOWN_LINK_RE = re.compile(r"\[[^]]*\]\(([^)#]+)(?:#([a-z0-9-]+))?\)")
_CUSTOM_ANCHOR_RE = re.compile(r'<a\s+id="([^"]+)"')
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.MULTILINE)
_CONTRIBUTING_URL = "https://github.com/NVIDIA/flashdreams/blob/main/CONTRIBUTING.md"
_SRC_RE = re.compile(r'\b(?:src|poster)="([^"]+)"')
_LEGACY_VIDEO_RE = re.compile(
    r'<video\b|<source\b[^>]*\btype="video/'
    r'|(?:src|poster)="[^"]+\.(?:mp4|webm|mov|mkv)(?:[?#][^"]*)?"',
    re.IGNORECASE,
)
_LEGACY_VIDEO_SUFFIXES = {".mkv", ".mov", ".mp4", ".webm"}


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


def test_documentation_motion_media_uses_avif() -> None:
    legacy = [
        str(path.relative_to(_DOCS_SOURCE))
        for path in (_DOCS_SOURCE / "_static").rglob("*")
        if path.suffix.lower() in _LEGACY_VIDEO_SUFFIXES
    ]

    for page in _DOCS_SOURCE.rglob("*.md"):
        source = page.read_text(encoding="utf-8")
        for match in _LEGACY_VIDEO_RE.finditer(source):
            line = source.count("\n", 0, match.start()) + 1
            legacy.append(f"{page.relative_to(_DOCS_SOURCE)}:{line}: {match.group(0)}")

    assert not legacy, "Documentation motion media must use AVIF:\n" + "\n".join(legacy)


def _anchors(path: Path) -> set[str]:
    source = path.read_text(encoding="utf-8")
    anchors = set(_CUSTOM_ANCHOR_RE.findall(source))
    for heading in _HEADING_RE.findall(source):
        slug = re.sub(r"[^\w -]", "", heading.lower()).strip().replace(" ", "-")
        anchors.add(slug)
    return anchors


def _assert_local_section_link(source: Path, target: str, fragment: str) -> None:
    page = (source.parent / target).resolve()
    assert page.is_relative_to(_DOCS_SOURCE), (
        f"{source.relative_to(_REPO_ROOT)}: canonical link leaves docs/source: "
        f"{target}#{fragment}"
    )
    assert page.is_file(), (
        f"{source.relative_to(_REPO_ROOT)}: missing canonical page: {target}"
    )
    assert not fragment or fragment in _anchors(page), (
        f"{source.relative_to(_REPO_ROOT)}: missing section #{fragment} in "
        f"{page.relative_to(_REPO_ROOT)}"
    )


def test_contributing_stays_on_github() -> None:
    source = (_REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    headings = set(_HEADING_RE.findall(source))
    assert {
        "Ways to contribute",
        "Submitting a pull request",
        "Adding documentation",
    } <= headings
    assert not (_DOCS_SOURCE / "community/contributing.md").exists()
    assert _CONTRIBUTING_URL in (_DOCS_SOURCE / "community/index.md").read_text()


def test_repository_links_resolve_to_canonical_sections() -> None:
    sources = [_REPO_ROOT / "README.md", _REPO_ROOT / "tests/README.md"]
    sources.extend(sorted((_REPO_ROOT / "skills").glob("*/SKILL.md")))

    found = 0
    for source in sources:
        for target, fragment in _MARKDOWN_LINK_RE.findall(
            source.read_text(encoding="utf-8")
        ):
            if target == _CONTRIBUTING_URL:
                assert not fragment or fragment in _anchors(
                    _REPO_ROOT / "CONTRIBUTING.md"
                )
                found += 1
                continue
            website = "https://nvidia.github.io/flashdreams/main/"
            if target.startswith(website):
                route = target.removeprefix(website).rstrip("/")
                page = _DOCS_SOURCE / route
                assert page.suffix != ".md", (
                    f"Website link must use a rendered route: {target}"
                )
                if page.suffix == ".html":
                    page = page.with_suffix(".md")
                elif page.is_dir():
                    page = page / "index.md"
                elif not page.suffix:
                    page = page.with_suffix(".md")
                _assert_local_section_link(source, str(page), fragment)
                found += 1
                continue
            if urlsplit(target).scheme:
                continue
            page = (source.parent / target).resolve()
            if source.name == "README.md":
                assert page.exists(), f"Missing README link: {target}"
            if page.is_relative_to(_DOCS_SOURCE):
                _assert_local_section_link(source, target, fragment)
                found += 1

    assert found, "expected canonical links from repository guidance to docs/source"


def test_readme_reference_links_have_definitions() -> None:
    source = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    references = set(re.findall(r"\[[^]]+\]\[([^]]+)\]", source))
    definitions = set(re.findall(r"^\[([^]]+)\]:", source, re.MULTILINE))
    assert references <= definitions, (
        f"Undefined README links: {references - definitions}"
    )


def test_demo_guide_navigation_matches_pages_and_cards() -> None:
    navigation = (_DOCS_SOURCE / ".nav.yml").read_text(encoding="utf-8")
    overview = (_DOCS_SOURCE / "documentation/demo_api/index.md").read_text(
        encoding="utf-8"
    )
    entries = re.findall(
        r"^\s+- ([^:]+): (documentation/demo_api/guides/[^\s]+\.md)$",
        navigation,
        re.MULTILINE,
    )
    assert entries, "expected Demo API guides in the sidebar"
    for label, target in entries:
        page = _DOCS_SOURCE / target
        assert page.is_file(), f"Sidebar links to missing guide: {target}"
        title = re.search(r"^title: ['\"](.+?)['\"]$", page.read_text(), re.MULTILINE)
        assert title and title[1] == label, f"Sidebar title differs from {target}"
        route = f"guides/{page.stem}/"
        assert f'href="{route}"' in overview, f"Missing guide card: {route}"

    routes = re.findall(r'href="(guides/[^\"]+/)"', overview)
    expected = {f"guides/{Path(target).stem}/" for _, target in entries}
    assert set(routes) == expected, "Guide cards and sidebar link to different pages"
