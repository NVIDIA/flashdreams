# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the inline SPDX workflow against a temporary source tree."""

import subprocess
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.ci_cpu


def test_inline_spdx_workflow(tmp_path: Path) -> None:
    workflow = (
        Path(__file__).resolve().parents[1] / ".github/workflows/reuse-lint.yml"
    ).read_text()
    step = workflow.split(
        "- name: Inline SPDX headers on first-party source files\n", 1
    )[1]
    script = textwrap.dedent(
        step.split("run: |\n", 1)[1].split("\n      - name:", 1)[0]
    )
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    # REUSE-IgnoreStart
    header = (
        "# SPDX-FileCopyrightText: Copyright (c) 2020 Example contributors\n"
        "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
    )
    # REUSE-IgnoreEnd
    for name in (
        "sample.py",
        "run.sh",
        "Dockerfile",
        "integrations_v2/omnidreams/impl/grpc/protos/example_pb2.py",
        "integrations_v2/omnidreams/impl/ludus-renderer/ludus_renderer/_cpp/cudaraster/upstream.h",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "// Licensed under the Apache License\n"
            if name.startswith("integrations_v2/")
            else "#!/bin/sh\n" + header
        )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)

    def check(expected_success: bool) -> str:
        result = subprocess.run(
            ["bash", "-c", script], cwd=tmp_path, capture_output=True, text=True
        )
        assert (result.returncode == 0) == expected_success, (
            result.stdout + result.stderr
        )
        return result.stdout

    check(True)
    sample = tmp_path / "sample.py"
    for prefix in ("#", "//", " *", "/*"):
        sample.write_text(
            header
            + f'{prefix} Licensed under the Apache License, Version 2.0 (the "License");\n'
        )
        assert "file=sample.py,line=1,title=Long SPDX header" in check(False)
    sample.write_text("pass\n")
    assert "file=sample.py,line=1,title=Missing SPDX header" in check(False)
    sample.write_text(header + 'message = "Licensed under the Apache License"\n')
    check(True)
