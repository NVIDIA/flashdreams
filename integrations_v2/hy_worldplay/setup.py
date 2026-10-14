# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bundle the canonical repository license texts for standalone distributions."""

from pathlib import Path

from setuptools import setup

package_root = Path(__file__).resolve().parent
canonical_licenses = package_root.parents[1] / "LICENSES"
bundle = package_root / "_licenses.txt"
license_names = (
    "Apache-2.0",
    "MIT",
    "LicenseRef-Tencent-HunyuanVideo-1.5-Community",
    "LicenseRef-Tencent-HY-WorldPlay-Community",
)

# A source archive already contains this bundle; a checkout uses root texts.
from_repository = (
    package_root.parent.name == "integrations_v2" and canonical_licenses.is_dir()
)
if from_repository:
    contents = b"".join(
        f"===== {name} =====\n\n".encode()
        + (canonical_licenses / f"{name}.txt").read_bytes()
        + b"\n\n"
        for name in license_names
    )
    bundle.write_bytes(contents)
elif not bundle.is_file():
    raise FileNotFoundError("Build from the repository or a complete source archive.")

try:
    setup(script_name="setup.py")
finally:
    if from_repository:
        bundle.unlink(missing_ok=True)
