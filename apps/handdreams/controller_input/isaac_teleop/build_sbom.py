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

"""Inventory the customer-installed Isaac Teleop wheels as draft SPDX 2.3."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

SIDECAR = Path("apps/handdreams/controller_input/isaac_teleop")
VERSION = "1.4.142"
REQUIREMENT = f"isaacteleop[cloudxr,retargeters-lite]=={VERSION}"


def parse_requirements(content: str) -> list[tuple[str, str, str]]:
    """Read exact package pins and their single reviewed wheel hashes."""
    packages = {}
    isaac_requirement = None
    for line in content.replace("\\\n", " ").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line in {
            "--index-url https://pypi.org/simple",
            "--only-binary :all:",
        }:
            continue
        match = re.fullmatch(
            r"([A-Za-z0-9][A-Za-z0-9._-]*)(\[[A-Za-z0-9,._ -]+\])?"
            r"==([A-Za-z0-9.+!_-]+)\s+--hash=sha256:([0-9a-f]{64})",
            line,
        )
        if not match:
            raise ValueError(
                f"Expected an exact package pin and one wheel SHA-256: {line}"
            )
        name, extras, version, digest = match.groups()
        name = re.sub(r"[-_.]+", "-", name).lower()
        if name in packages:
            raise ValueError(f"Duplicate package: {name}")
        packages[name] = (name, version, digest)
        if name == "isaacteleop":
            isaac_requirement = f"{name}{extras or ''}=={version}".replace(" ", "")
    if isaac_requirement != REQUIREMENT:
        raise ValueError(f"Manifest must select the reviewed {REQUIREMENT}")
    return sorted(packages.values())


def build_sbom(repository: Path) -> dict:
    """Inventory both platform manifests at their actual checkout paths."""
    directories = [
        repository / path
        for path in (SIDECAR, Path("internal") / SIDECAR)
        if (repository / path).is_dir()
    ]
    if not directories:
        raise ValueError("No Isaac Teleop sidecar manifests found in the repository.")
    files, packages, relationships = [], [], []
    for directory in directories:
        scope = (
            "customer-installed-runtime"
            if directory == repository / SIDECAR
            else "promotion-customer-installed-runtime-review"
        )
        for arch in ("aarch64", "x86_64"):
            manifest = directory / f"requirements-linux-{arch}-py312.txt"
            content = manifest.read_bytes()
            path = manifest.relative_to(repository).as_posix()
            digest = hashlib.sha256(content).hexdigest()
            context = hashlib.sha256(path.encode()).hexdigest()[:16]
            files.append(
                {
                    "SPDXID": f"SPDXRef-Manifest-{context}",
                    "fileName": path,
                    "checksums": [{"algorithm": "SHA256", "checksumValue": digest}],
                    "licenseConcluded": "Apache-2.0",
                    "licenseInfoInFiles": ["Apache-2.0"],
                    "copyrightText": "Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES",
                }
            )
            for name, version, wheel_hash in parse_requirements(content.decode()):
                package_id = f"SPDXRef-Package-{context}-{name}"
                comment = (
                    f"scope={scope}; platform=linux-{arch}-py312; manifest={path}. "
                    "Customer installs this wheel from public PyPI into an isolated "
                    "subprocess environment; FlashDreams does not redistribute it."
                )
                if name == "isaacteleop":
                    comment += (
                        " PyPI wheel metadata declares Apache-2.0, but the wheel "
                        "contains CloudXR native components requiring explicit "
                        "NVIDIA CloudXR EULA acceptance. The combined license "
                        "conclusion remains NOASSERTION pending applicable review."
                    )
                packages.append(
                    {
                        "SPDXID": package_id,
                        "name": name,
                        "versionInfo": version,
                        "downloadLocation": f"https://pypi.org/project/{name}/{version}/",
                        "filesAnalyzed": False,
                        "checksums": [
                            {"algorithm": "SHA256", "checksumValue": wheel_hash}
                        ],
                        "licenseDeclared": "Apache-2.0"
                        if name == "isaacteleop"
                        else "NOASSERTION",
                        "licenseConcluded": "NOASSERTION",
                        "copyrightText": "NOASSERTION",
                        "externalRefs": [
                            {
                                "referenceCategory": "PACKAGE-MANAGER",
                                "referenceType": "purl",
                                "referenceLocator": f"pkg:pypi/{name}@{version}",
                            }
                        ],
                        "comment": comment,
                    }
                )
                relationships.append(
                    {
                        "spdxElementId": "SPDXRef-IsaacTeleopSidecar",
                        "relationshipType": "DEPENDS_ON",
                        "relatedSpdxElement": package_id,
                        "comment": "Resolved sidecar closure; individual dependency edges are not encoded by the manifest.",
                    }
                )
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": "flashdreams-isaac-teleop-sidecar-draft",
        "documentNamespace": f"https://spdx.nvidia.com/flashdreams/isaac-teleop/{identity}",
        "creationInfo": {
            "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "creators": ["Tool: isaac_teleop/build_sbom.py"],
            "comment": "Draft checkout manifest inventory; not a license approval or a complete FlashDreams release SBOM.",
        },
        "documentDescribes": ["SPDXRef-IsaacTeleopSidecar"],
        "packages": [
            {
                "SPDXID": "SPDXRef-IsaacTeleopSidecar",
                "name": "flashdreams-isaac-teleop-sidecar",
                "versionInfo": VERSION,
                "downloadLocation": "NOASSERTION",
                "filesAnalyzed": False,
                "licenseDeclared": "NOASSERTION",
                "licenseConcluded": "NOASSERTION",
                "copyrightText": "NOASSERTION",
                "comment": "Logical environment descriptor; all external wheels are customer-installed dependencies.",
            },
            *packages,
        ],
        "files": files,
        "relationships": relationships,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        document = build_sbom(args.repository)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    args.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"Wrote draft Isaac Teleop inventory to {args.output}")


if __name__ == "__main__":
    main()
