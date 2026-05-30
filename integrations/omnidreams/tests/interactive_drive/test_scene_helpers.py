# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from omnidreams.interactive_drive.assets.scene_bundle import (
    _discover_first_frames,
    _discover_prompts,
)


class SceneHelperTest(unittest.TestCase):
    def test_discovers_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # ``prompt_<N>.txt`` is the canonical convention used by the
            # synthetic-scene fixture. The bare numeric ``promptN.txt``
            # form is also accepted for legacy demo assets.
            (root / "prompt.txt").write_text("default text", encoding="utf-8")
            (root / "prompt_1.txt").write_text("hello", encoding="utf-8")
            (root / "prompt_2.txt").write_text("snow", encoding="utf-8")
            (root / "prompt3.txt").write_text("rain", encoding="utf-8")
            (root / "promptnight.txt").write_text("ignored", encoding="utf-8")
            (root / "first_image.png").write_bytes(b"")
            (root / "first_image_2.png").write_bytes(b"")
            (root / "first_image3.png").write_bytes(b"")
            (root / "first_imagenight.png").write_bytes(b"")
            prompts = _discover_prompts(root)
            first_frames = _discover_first_frames(root)
            self.assertEqual(prompts["1"], "hello")
            self.assertEqual(prompts["2"], "snow")
            self.assertEqual(prompts["3"], "rain")
            self.assertEqual(prompts["default"], "default text")
            self.assertNotIn("night", prompts)
            self.assertEqual(first_frames["default"].name, "first_image.png")
            self.assertEqual(first_frames["2"].name, "first_image_2.png")
            self.assertEqual(first_frames["3"].name, "first_image3.png")
            self.assertNotIn("night", first_frames)


if __name__ == "__main__":
    unittest.main()
