# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from omnidreams.interactive_drive import cli


class CliManifestResolutionTest(unittest.TestCase):
    def test_resolves_bundled_manifest_by_filename(self) -> None:
        manifest = cli.resolve_manifest_path("example_world_model_perf.yaml")

        self.assertEqual(manifest.name, "example_world_model_perf.yaml")
        self.assertEqual(manifest.parent, cli._CONFIGS_ROOT)
        self.assertTrue(manifest.is_file())

    def test_cwd_relative_manifest_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = Path.cwd()
            root = Path(tmpdir)
            manifest = root / "example_world_model_perf.yaml"
            manifest.write_text("resolution_wh: [1280, 704]\n", encoding="utf-8")
            try:
                os.chdir(root)
                resolved = cli.resolve_manifest_path(manifest.name)
            finally:
                os.chdir(old_cwd)

        self.assertEqual(resolved, manifest.resolve())


if __name__ == "__main__":
    unittest.main()
