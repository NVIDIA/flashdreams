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


def _resolves(reference: str, document: Path) -> bool:
    """Return whether ``reference`` names something that exists.

    A reference carrying a directory is resolved against the document's own
    directory and each of its ancestors, so both ``../api_v2/README.md`` and
    ``runtime_v2/client_window_factory.py`` reach their target. A bare file name
    is accepted if anything in the repository is called that, which is enough to
    catch a module that was renamed or never existed.
    """
    if "/" in reference:
        for directory in (document.parent, *document.parent.parents):
            if (directory / reference).exists():
                return True
            if directory == _ROOT:
                break
        return False
    return any(_ROOT.rglob(reference))


def _file_references(document: Path) -> list[str]:
    """Return the relative links and backticked file names in ``document``."""
    prose = _prose(document)
    references = [
        target.split("#", 1)[0]
        for target in _MARKDOWN_LINK.findall(prose)
        if not target.startswith(("http://", "https://", "#"))
    ]
    references += [
        span
        for span in _BACKTICKED.findall(prose)
        if span.endswith(_FILE_SUFFIXES) and " " not in span
    ]
    return [reference for reference in references if reference]


@pytest.mark.parametrize("relative_path", _DOCUMENTS)
def test_every_v2_document_exists(relative_path: str) -> None:
    """The documents this checks are the documents that are there."""
    assert (_ROOT / relative_path).is_file(), (
        f"{relative_path} is listed here but not in the repository."
    )


@pytest.mark.parametrize("relative_path", _DOCUMENTS)
def test_v2_documentation_references_resolve(relative_path: str) -> None:
    """Every link and backticked file name reaches something."""
    document = _ROOT / relative_path
    broken = sorted(
        {
            reference
            for reference in _file_references(document)
            if not _resolves(reference, document)
        }
    )

    assert not broken, f"{relative_path} refers to files that do not exist: {broken}"
