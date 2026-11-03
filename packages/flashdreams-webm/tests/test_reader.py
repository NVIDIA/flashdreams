# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check native WebM reader cleanup."""

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.ci_cpu


def test_reader_close() -> None:
    # Isolate allocator aborts so a regression fails this test, not the runner.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""\
                from pathlib import Path
                from tempfile import TemporaryDirectory

                from flashdreams_webm import Reader, Writer

                with TemporaryDirectory() as directory:
                    path = str(Path(directory) / "frame.webm")
                    writer = Writer(path, 64, 64, 30, 1)
                    writer.write(b"\\x10" * (64 * 64) + b"\\x80" * (64 * 64 // 2))
                    writer.close()
                    reader = Reader(path)
                    assert reader.read() is not None
                    assert reader.read() is None
                    reader.close()
                    reader.close()
                    del reader
                """),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
