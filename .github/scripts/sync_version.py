#!/usr/bin/env python3
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

"""Sync the canonical version from flashdreams/_version.py to all pyproject.toml files.

The version in ``flashdreams/flashdreams/_version.py`` is the single source of
truth for the entire monorepo.  This script reads it and updates the
``version = "..."`` line in every workspace-member ``pyproject.toml``, except
for packages with independent versioning (ludus-renderer, self-forcing
parity-check).

Intended to run as a pre-commit hook and as a CI safety-net step.

Exit codes:
    0 -- all versions were already in sync (no files changed).
    1 -- one or more files were updated (pre-commit will re-stage them).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Packages that maintain their own independent version.
SKIP_PACKAGES = {
    "ludus-renderer",
    "self-forcing-parity-check",
}

# Regex to extract __version__ = "X.Y.Z" from _version.py.
_VERSION_RE = re.compile(r'^__version__\s*=\s*"([^"]+)"', re.MULTILINE)

# Regex to replace version = "..." in pyproject.toml (under [project]).
_TOML_VERSION_RE = re.compile(r'^(version\s*=\s*)"[^"]*"', re.MULTILINE)

# Regex to read the project name from pyproject.toml.
_TOML_NAME_RE = re.compile(r'^name\s*=\s*"([^"]+)"', re.MULTILINE)


def read_canonical_version() -> str:
    """Read the version string from flashdreams/_version.py."""
    version_file = REPO_ROOT / "flashdreams" / "flashdreams" / "_version.py"
    text = version_file.read_text()
    match = _VERSION_RE.search(text)
    if not match:
        print(f"ERROR: could not parse version from {version_file}", file=sys.stderr)
        sys.exit(2)
    return match.group(1)


def find_pyproject_files() -> list[Path]:
    """Return all pyproject.toml files under the repo root."""
    return sorted(REPO_ROOT.rglob("pyproject.toml"))


def should_skip(path: Path, text: str) -> bool:
    """Return True if this pyproject.toml should not be version-synced."""
    # Skip the root workspace config (has no [project] section).
    if path == REPO_ROOT / "pyproject.toml":
        return True
    # Skip the canonical source itself.
    if path == REPO_ROOT / "flashdreams" / "pyproject.toml":
        return True
    # Skip packages with independent versioning.
    name_match = _TOML_NAME_RE.search(text)
    if name_match and name_match.group(1) in SKIP_PACKAGES:
        return True
    # Skip files that don't have a version field (e.g. root config).
    if not _TOML_VERSION_RE.search(text):
        return True
    return False


def sync_version(version: str) -> list[Path]:
    """Update all pyproject.toml files and return the list of changed paths."""
    changed: list[Path] = []
    for path in find_pyproject_files():
        text = path.read_text()
        if should_skip(path, text):
            continue
        new_text = _TOML_VERSION_RE.sub(rf'\g<1>"{version}"', text)
        if new_text != text:
            path.write_text(new_text)
            changed.append(path)
    return changed


def main() -> None:
    version = read_canonical_version()
    changed = sync_version(version)
    if changed:
        for path in changed:
            rel = path.relative_to(REPO_ROOT)
            print(f"Updated {rel} -> {version}")
        # Non-zero exit tells pre-commit that files were modified.
        sys.exit(1)
    else:
        print(f"All versions already at {version}")


if __name__ == "__main__":
    main()
