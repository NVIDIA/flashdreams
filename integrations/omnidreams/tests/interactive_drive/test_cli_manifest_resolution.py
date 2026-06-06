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

    def test_relative_recording_dir_resolves_from_flashdreams_root(self) -> None:
        resolved = cli._resolve_recording_output_dir(Path("captures"), enabled=True)

        self.assertEqual(resolved, (cli._FLASHDREAMS_ROOT / "captures").resolve())

    def test_absolute_recording_dir_is_preserved(self) -> None:
        absolute = Path(tempfile.gettempdir()) / "interactive-drive-captures"

        resolved = cli._resolve_recording_output_dir(absolute, enabled=True)

        self.assertEqual(resolved, absolute)

    def test_default_recording_dir_is_flashdreams_root_recordings(self) -> None:
        resolved = cli._resolve_recording_output_dir(None, enabled=True)

        self.assertEqual(
            resolved,
            (cli._FLASHDREAMS_ROOT / "recordings").resolve(),
        )

    def test_disabled_recording_has_no_output_dir(self) -> None:
        resolved = cli._resolve_recording_output_dir(Path("captures"), enabled=False)

        self.assertIsNone(resolved)

    def test_flashdreams_root_fallback_warns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            start = Path(tmpdir) / "installed" / "omnidreams"
            start.mkdir(parents=True)
            old_cwd = Path.cwd()
            try:
                os.chdir(tmpdir)
                with self.assertWarnsRegex(
                    RuntimeWarning,
                    "Could not locate the flashdreams repository root",
                ):
                    resolved = cli._find_flashdreams_root(start)
            finally:
                os.chdir(old_cwd)

        self.assertEqual(resolved, Path(tmpdir).resolve())


if __name__ == "__main__":
    unittest.main()
