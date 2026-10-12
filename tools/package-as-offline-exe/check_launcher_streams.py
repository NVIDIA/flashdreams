# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import os
import runpy
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from loguru import logger

packager = runpy.run_path(str(Path(__file__).with_name("package_as_offline_exe.py")))
source = packager["_launcher_source"]("demo", [])
compile(source, "launcher.py", "exec")
startup = source.split("\nfrom flashdreams.runtime_v2.cli import", 1)[0]

for missing_stdout, missing_stderr in (
    (True, True),
    (True, False),
    (False, True),
    (False, False),
):
    stdout = None if missing_stdout else io.StringIO()
    stderr = None if missing_stderr else io.StringIO()
    with (
        TemporaryDirectory() as directory,
        patch.dict(os.environ),
        patch.object(sys, "argv", ["launcher.py"]),
        patch.object(sys, "stdout", stdout),
        patch.object(sys, "stderr", stderr),
    ):
        namespace: dict[str, Any] = {"__file__": str(Path(directory) / "launcher.py")}
        exec(startup, namespace)
        assert sys.stdout is not None and sys.stderr is not None
        assert missing_stdout or sys.stdout is stdout
        assert missing_stderr or sys.stderr is stderr
        handler = logger.add(sys.stderr, format="{message}")
        try:
            logger.info("launcher logging works")
            sys.stdout.write("launcher output works\n")
            if missing_stdout or missing_stderr:
                log = Path(directory) / "cache" / "runtime.log"
                contents = log.read_text(encoding="utf-8")
                assert not missing_stderr or "launcher logging works" in contents
                assert not missing_stdout or "launcher output works" in contents
            else:
                assert "runtime_log" not in namespace
        finally:
            logger.remove(handler)
            if "runtime_log" in namespace:
                namespace["runtime_log"].close()

print("Launcher stream checks passed.")
