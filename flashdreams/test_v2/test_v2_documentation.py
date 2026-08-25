# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that the v2 documentation still points at things that exist.

Nothing else reads the v2 Markdown, so a renamed module leaves a reference
behind and nobody finds out until someone follows it. This checks the two kinds
of reference that can be checked mechanically: relative links, and backticked
file names.
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.ci_cpu

_FENCED_BLOCK = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
"""Fenced code block, stripped before extraction.

Examples are allowed to name files that do not exist, such as the placeholder
``test_<name>.py`` in the integration guide's directory listing.
"""

_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
"""Markdown inline link, capturing its target."""

_BACKTICKED = re.compile(r"`([^`\n]+)`")
"""Single-backticked span, capturing its contents."""

_FILE_SUFFIXES = (".py", ".md", ".json", ".toml", ".yml", ".sh")
"""Suffixes that make a backticked span worth resolving as a file."""


def _repository_root() -> Path:
    """Return the repository root, found by looking for ``integrations_v2``."""
    for directory in Path(__file__).resolve().parents:
        if (directory / "integrations_v2").is_dir():
            return directory
    raise AssertionError("Could not find the repository root from this test file.")


_ROOT = _repository_root()

_DOCUMENTS = (
    "ARCHITECTURE.md",
    "flashdreams/flashdreams/api_v2/README.md",
    "flashdreams/flashdreams/runtime_v2/README.md",
    "flashdreams/flashdreams/t2v_v2/README.md",
    "integrations_v2/README.md",
    *sorted(
        str(path.relative_to(_ROOT))
        for path in (_ROOT / "integrations_v2").glob("*/README.md")
    ),
)
"""Every v2 Markdown document, listed so a missing one fails rather than passes."""


def _prose(document: Path) -> str:
    """Return ``document`` with its fenced code blocks removed."""
    return _FENCED_BLOCK.sub("", document.read_text(encoding="utf-8"))


def _link_resolves(target: str, document: Path) -> bool:
    """Return whether a link reaches a file, resolved the way a reader's does.

    Against the document's own directory and nowhere else, because that is what
    a Markdown renderer does with a relative link. A link is clickable, so a
    target that only happens to exist somewhere else in the repository is
    broken for the reader and fails here. A target climbing out of the
    repository fails too, having nothing to reach.
    """
    destination = (document.parent / target).resolve()
    return destination.is_relative_to(_ROOT) and destination.exists()


def _file_name_resolves(span: str) -> bool:
    """Return whether a backticked file name matches something that exists.

    Looser than a link, deliberately. Prose names a file the way the sentence
    around it reads rather than relative to the document, so
    ``runtime_v2/client_window_factory.py`` appears in documents at three
    different depths and none of them mean it as a path. Anything whose path
    ends with the span therefore counts, which still catches the thing these
    get wrong: a module renamed, or never written.
    """
    return any(
        str(path.relative_to(_ROOT)).endswith(span)
        for path in _ROOT.rglob(span.rsplit("/", 1)[-1])
    )


def _links(document: Path) -> list[str]:
    """Return the in-repository link targets in ``document``, anchors stripped."""
    return [
        stripped
        for target in _MARKDOWN_LINK.findall(_prose(document))
        if not target.startswith(("http://", "https://", "#"))
        for stripped in [target.split("#", 1)[0]]
        if stripped
    ]


def _file_names(document: Path) -> list[str]:
    """Return the backticked spans in ``document`` that name a file."""
    return [
        span
        for span in _BACKTICKED.findall(_prose(document))
        if span.endswith(_FILE_SUFFIXES) and " " not in span
    ]


@pytest.mark.parametrize("relative_path", _DOCUMENTS)
def test_every_v2_document_exists(relative_path: str) -> None:
    """The documents this checks are the documents that are there."""
    assert (_ROOT / relative_path).is_file(), (
        f"{relative_path} is listed here but not in the repository."
    )


@pytest.mark.parametrize("relative_path", _DOCUMENTS)
def test_v2_documentation_links_resolve(relative_path: str) -> None:
    """Every link reaches a file from where the document that carries it sits."""
    document = _ROOT / relative_path
    broken = sorted(
        {target for target in _links(document) if not _link_resolves(target, document)}
    )

    assert not broken, (
        f"{relative_path} links to targets that do not resolve from "
        f"{document.parent.relative_to(_ROOT) or '.'}: {broken}"
    )


@pytest.mark.parametrize("relative_path", _DOCUMENTS)
def test_v2_documentation_file_names_exist(relative_path: str) -> None:
    """Every backticked file name matches a file somewhere in the repository."""
    document = _ROOT / relative_path
    broken = sorted(
        {span for span in _file_names(document) if not _file_name_resolves(span)}
    )

    assert not broken, f"{relative_path} names files that do not exist: {broken}"
