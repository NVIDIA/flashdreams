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
            self.assertEqual(manifest.native_dit_acceleration, "disabled")
            self.assertEqual(manifest.native_vae_encoder, "disabled")
            self.assertIsNone(manifest.native_vae_fp8_state_path)
            self.assertFalse(manifest.upsampling_enabled)
            self.assertEqual(manifest.upsampling_server, "127.0.0.1:8090")
            self.assertEqual(manifest.upsampling_scale, 4)

    def test_loads_upsampling_knobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.yaml"
            path.write_text(
                textwrap.dedent(
                    """
                    upsampling_enabled: true
                    upsampling_server: uplift.example:8090
                    upsampling_scale: 2
                    upsampling_sparse_ratio: 1.5
                    upsampling_input_format: raw
                    upsampling_input_jpeg_quality: 85
                    upsampling_return_frames: true
                    upsampling_max_queue_chunks: 2
                    upsampling_max_message_mb: 256
                    """
                ).strip(),
                encoding="utf-8",
            )
            manifest = load_world_model_manifest(path)
            self.assertTrue(manifest.upsampling_enabled)
            self.assertEqual(manifest.upsampling_server, "uplift.example:8090")
            self.assertEqual(manifest.upsampling_scale, 2)
            self.assertEqual(manifest.upsampling_sparse_ratio, 1.5)
            self.assertEqual(manifest.upsampling_input_format, "raw")
            self.assertEqual(manifest.upsampling_input_jpeg_quality, 85)
            self.assertTrue(manifest.upsampling_return_frames)
            self.assertEqual(manifest.upsampling_max_queue_chunks, 2)
            self.assertEqual(manifest.upsampling_max_message_mb, 256)

    def test_loads_native_dit_knobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.yaml"
            path.write_text(
                textwrap.dedent(
                    """
                    native_dit_acceleration: required
                    native_dit_build_root: /tmp/omnidreams-native
                    native_dit_max_jobs: 8
                    native_dit_verbose_build: true
                    native_dit_backend: bf16
                    native_dit_attention_backend: sparge
                    native_dit_sparge_topk: 0.4
                    native_dit_sparge_hybrid_period: 4
                    native_dit_sparge_hybrid_phase: 1
                    """
                ).strip(),
                encoding="utf-8",
            )
            manifest = load_world_model_manifest(path)
            self.assertEqual(manifest.native_dit_acceleration, "required")
            self.assertEqual(manifest.native_dit_build_root, "/tmp/omnidreams-native")
            self.assertEqual(manifest.native_dit_max_jobs, 8)
            self.assertTrue(manifest.native_dit_verbose_build)
            self.assertEqual(manifest.native_dit_backend, "bf16")
            self.assertEqual(manifest.native_dit_attention_backend, "sparge")
            self.assertEqual(manifest.native_dit_sparge_topk, 0.4)
            self.assertEqual(manifest.native_dit_sparge_hybrid_period, 4)
            self.assertEqual(manifest.native_dit_sparge_hybrid_phase, 1)

    def test_loads_native_vae_knobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_dir = root / "configs"
            config_dir.mkdir()
            path = config_dir / "manifest.yaml"
            path.write_text(
                textwrap.dedent(
                    """
                    native_vae_encoder: fp8
                    native_vae_fp8_state_path: ../native/lightvae-fp8-state.pt
                    """
                ).strip(),
                encoding="utf-8",
            )
            manifest = load_world_model_manifest(path)
            self.assertEqual(manifest.native_vae_encoder, "fp8")
            self.assertEqual(
                manifest.native_vae_fp8_state_path,
                (root / "native/lightvae-fp8-state.pt").resolve(),
            )

    def test_rejects_unaligned_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.yaml"
            path.write_text(
                textwrap.dedent(
                    """
                    resolution_wh: [1164, 640]
                    """
                ).strip(),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "divisible by 16"):
                load_world_model_manifest(path)

    def test_rejects_invalid_native_dit_acceleration(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.yaml"
            path.write_text("native_dit_acceleration: fast\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "native_dit_acceleration"):
                load_world_model_manifest(path)

    def test_rejects_invalid_native_vae_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.yaml"
            path.write_text("native_vae_encoder: fast\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "native_vae_encoder"):
                load_world_model_manifest(path)

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
