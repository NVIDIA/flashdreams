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

"""Validate repository Agent Skills against the core agentskills.io spec."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.ci_cpu

_ROOT = Path(__file__).resolve().parents[1]
_IGNORED_SKILL_PARTS = frozenset({".agents", ".claude", ".codex", ".cursor"})
_NAME_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def _is_repo_skill_file(path: Path) -> bool:
    parts = path.relative_to(_ROOT).parts
    return not _IGNORED_SKILL_PARTS.intersection(parts)


def _repo_skill_files() -> list[Path]:
    return sorted(path for path in _ROOT.rglob("SKILL.md") if _is_repo_skill_file(path))


def _frontmatter_lines(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines and lines[0] == "---", f"{path} is missing YAML frontmatter"

    for index, line in enumerate(lines[1:], start=1):
        if line == "---":
            return lines[1:index]

    raise AssertionError(f"{path} frontmatter is not closed")


def _dedent_block(lines: list[str]) -> list[str]:
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    if not indents:
        return []
    indent = min(indents)
    return [line[indent:] if len(line) >= indent else "" for line in lines]


def _parse_block_scalar(lines: list[str], *, folded: bool) -> str:
    block = _dedent_block(lines)
    if folded:
        return " ".join(line.strip() for line in block if line.strip())
    return "\n".join(block).strip()


def _frontmatter(path: Path) -> dict[str, str]:
    lines = _frontmatter_lines(path)
    values: dict[str, str] = {}
    index = 0

    while index < len(lines):
        line = lines[index]
        if not line or line.startswith((" ", "\t", "#")):
            index += 1
            continue
        key, separator, value = line.partition(":")
        if not separator:
            index += 1
            continue

        value = value.strip()
        if value in {"|", "|-", "|+", ">", ">-", ">+"}:
            block_start = index + 1
            block_end = block_start
            while block_end < len(lines):
                next_line = lines[block_end]
                if next_line and not next_line.startswith((" ", "\t")):
                    break
                block_end += 1
            values[key.strip()] = _parse_block_scalar(
                lines[block_start:block_end], folded=value.startswith(">")
            )
            index = block_end
            continue

        values[key.strip()] = value.strip("'\"")
        index += 1

    return values


def test_repo_skill_filter_ignores_local_agent_dirs() -> None:
    for ignored in _IGNORED_SKILL_PARTS:
        assert not _is_repo_skill_file(_ROOT / ignored / "skills" / "x" / "SKILL.md")
    assert _is_repo_skill_file(_ROOT / "skills" / "x" / "SKILL.md")


def test_frontmatter_parser_handles_block_scalars(tmp_path: Path) -> None:
    skill = tmp_path / "block-skill" / "SKILL.md"
    skill.parent.mkdir()
    skill.write_text(
        "---\n"
        "name: block-skill\n"
        "description: >\n"
        "  Run a workflow\n"
        "  when block scalar frontmatter is used.\n"
        "literal: |\n"
        "  line one\n"
        "  line two\n"
        "---\n"
        "# Body\n",
        encoding="utf-8",
    )

    values = _frontmatter(skill)

    assert (
        values["description"] == "Run a workflow when block scalar frontmatter is used."
    )
    assert values["literal"] == "line one\nline two"


def test_skill_frontmatter_matches_agentskills_spec() -> None:
    skill_files = _repo_skill_files()
    assert skill_files, "expected at least one repo skill"

    for path in skill_files:
        frontmatter = _frontmatter(path)
        name = frontmatter.get("name", "")
        description = frontmatter.get("description", "")

        assert 1 <= len(name) <= 64, f"{path}: invalid name length"
        assert _NAME_RE.fullmatch(name), f"{path}: invalid skill name {name!r}"
        assert name == path.parent.name, (
            f"{path}: name must match parent directory {path.parent.name!r}"
        )
        assert 1 <= len(description) <= 1024, f"{path}: invalid description length"
