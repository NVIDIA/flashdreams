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
