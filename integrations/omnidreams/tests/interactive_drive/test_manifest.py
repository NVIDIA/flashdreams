# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import tempfile
import textwrap
import unittest
from pathlib import Path

from omnidreams.interactive_drive.world_model.manifest import load_world_model_manifest


class WorldModelManifestTest(unittest.TestCase):
    def test_loads_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.yaml"
            path.write_text(
                textwrap.dedent(
                    """
                    resolution_wh: [1280, 704]
                    """
                ).strip(),
                encoding="utf-8",
            )
            manifest = load_world_model_manifest(path)
            self.assertEqual(manifest.num_frames_per_block, 8)
            self.assertEqual(manifest.denoising_steps, [1000, 500])
            self.assertEqual(manifest.resolution_wh, (1280, 704))

    def test_resolves_relative_paths_from_manifest_location(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_dir = root / "configs"
            config_dir.mkdir()
            path = config_dir / "manifest.yaml"
            path.write_text(
                textwrap.dedent(
                    """
                    debug_condition_frame_dir: ../debug-trace/replay-from-recording-v4/live/camera_front_wide_120fov
                    """
                ).strip(),
                encoding="utf-8",
            )
            manifest = load_world_model_manifest(path)
            self.assertEqual(
                manifest.debug_condition_frame_dir,
                (
                    root
                    / "debug-trace/replay-from-recording-v4/live/camera_front_wide_120fov"
                ).resolve(),
            )


if __name__ == "__main__":
    unittest.main()
