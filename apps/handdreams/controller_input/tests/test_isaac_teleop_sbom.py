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

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.ci_cpu

SIDECAR = Path(__file__).parents[1] / "isaac_teleop"
SPEC = importlib.util.spec_from_file_location(
    "isaac_teleop_sbom", SIDECAR / "build_sbom.py"
)
assert SPEC and SPEC.loader
SBOM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SBOM)


def test_inventory_includes_both_platforms_without_workspace_lock():
    for prefix, scope in (
        (Path("internal"), "promotion-customer-installed-runtime-review"),
        (Path(), "customer-installed-runtime"),
    ):
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            manifests = repository / prefix / SBOM.SIDECAR
            manifests.mkdir(parents=True)
            for path in SIDECAR.glob("requirements-linux-*-py312.txt"):
                shutil.copyfile(path, manifests / path.name)
            result = SBOM.build_sbom(repository)
            assert result["spdxVersion"] == "SPDX-2.3"
            assert len(result["files"]) == 2
            assert (
                len(result["packages"]) == 17
            )  # Descriptor plus 8 wheels per platform.
            ids = {item["SPDXID"] for item in result["packages"] + result["files"]}
            assert len(ids) == 19
            assert set(result["documentDescribes"]) <= ids
            assert len(result["relationships"]) == 16
            for relationship in result["relationships"]:
                assert relationship["relationshipType"] == "DEPENDS_ON"
                assert relationship["spdxElementId"] in ids
                assert relationship["relatedSpdxElement"] in ids
            for manifest in result["files"]:
                path = repository / manifest["fileName"]
                assert path.parent == manifests
                assert manifest["checksums"] == [
                    {
                        "algorithm": "SHA256",
                        "checksumValue": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                ]
                expected = SBOM.parse_requirements(path.read_text())
                assert {name for name, _, _ in expected} == {
                    "cffi",
                    "cryptography",
                    "isaacteleop",
                    "numpy",
                    "pycparser",
                    "pyyaml",
                    "scipy",
                    "websockets",
                }
                packages = [
                    item
                    for item in result["packages"]
                    if f"manifest={manifest['fileName']}." in item["comment"]
                ]
                assert len(packages) == 8
                for name, version, digest in expected:
                    package = next(item for item in packages if item["name"] == name)
                    assert package["versionInfo"] == version
                    assert package["checksums"] == [
                        {"algorithm": "SHA256", "checksumValue": digest}
                    ]
                    assert f"scope={scope};" in package["comment"]
                    assert package["licenseConcluded"] == "NOASSERTION"
                    assert package["licenseDeclared"] == (
                        "Apache-2.0" if name == "isaacteleop" else "NOASSERTION"
                    )
                    assert (
                        package["downloadLocation"]
                        == f"https://pypi.org/project/{name}/{version}/"
                    )
            (manifests / "requirements-linux-aarch64-py312.txt").unlink()
            with pytest.raises(FileNotFoundError):
                SBOM.build_sbom(repository)


def test_inventory_rejects_missing_or_unreviewed_inputs():
    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(ValueError, match="No Isaac Teleop"):
            SBOM.build_sbom(Path(directory))
    good = f"{SBOM.REQUIREMENT} --hash=sha256:{'a' * 64}\n"
    for bad in (
        good.replace("==1.4.142", ">=1.4.142"),
        good.replace(f" --hash=sha256:{'a' * 64}", ""),
        good.replace("1.4.142", "1.4.999"),
        good.replace("[cloudxr,retargeters-lite]", ""),
        good + good,
        good + "--index-url https://private.example/simple\n",
    ):
        with pytest.raises(ValueError):
            SBOM.parse_requirements(bad)
