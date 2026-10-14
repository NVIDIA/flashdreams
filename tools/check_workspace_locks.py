# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workspace lockfile discovery checks for recursive dependency scanners."""

import subprocess
from pathlib import Path


def check_workspace_locks(root: Path) -> None:
    """Check that each tracked manifest exposes its committed lockfile."""
    tracked = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "pyproject.toml", "**/pyproject.toml"],
        cwd=root,
    ).decode().split("\0")
    manifests = [root / path for path in tracked if path]
    aliases = 0
    for manifest in manifests:
        lock = manifest.with_name("uv.lock")
        assert lock.is_file(), f"Missing lockfile beside {manifest}"
        if not lock.is_symlink():
            continue
        assert not lock.readlink().is_absolute(), f"Non-portable lock alias: {lock}"
        owner = lock.parent.parent
        while owner != root and not (owner / "uv.lock").is_file():
            owner = owner.parent
        expected = (owner / "uv.lock").resolve()
        assert expected.is_relative_to(root), f"Lock escapes repository: {lock}"
        assert lock.resolve() == expected, f"Wrong workspace lock: {lock}"
        aliases += 1
    print(f"{len(manifests)} manifests have locks; {aliases} shared-lock aliases")


if __name__ == "__main__":
    check_workspace_locks(Path(__file__).resolve().parents[1])
