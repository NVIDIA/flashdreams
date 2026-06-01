# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Per-variant thumbnail discovery feeding the HUD variant dropdown."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from omnidreams.interactive_drive.demo import (
    SCENE_THUMB_SIZE,
    _discover_variants,
    _load_variant_thumbnails,
)
from omnidreams.interactive_drive.scene_fixture import build_synthetic_scene_usdz
from PIL import Image


def test_load_variant_thumbnails_one_per_variant(tmp_path: Path) -> None:
    # The synthetic bundle ships first_image.png + first_image_1/2.png, so
    # each variant must get its own distinctly-rendered preview.
    scene_path = build_synthetic_scene_usdz(tmp_path / "scene.usdz", length_frames=60)
    # Numbered variants exist, so the bare "default" (a duplicate of "1") is
    # dropped and "1" becomes the default selection.
    variants = _discover_variants(scene_path)
    assert variants == ("1", "2")

    thumbs = _load_variant_thumbnails(scene_path, variants)
    assert set(thumbs) == {"1", "2"}
    for thumb in thumbs.values():
        assert isinstance(thumb, Image.Image)
        assert thumb.size == SCENE_THUMB_SIZE
    # The per-variant first images are distinct (variant_1/2 are shifted),
    # so the rendered thumbnails must not collapse to one shared image.
    rendered = {variant: thumb.tobytes() for variant, thumb in thumbs.items()}
    assert len(set(rendered.values())) == 2


def test_discover_variants_default_only_when_unnumbered(tmp_path: Path) -> None:
    # A scene with only the bare prompt / first_image (no numbered variants)
    # exposes a single "default".
    scene_path = tmp_path / "plain.usdz"
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color=(5, 5, 5)).save(buf, format="PNG")
    with zipfile.ZipFile(scene_path, "w") as zf:
        zf.writestr("first_image.png", buf.getvalue())
        zf.writestr("prompt.txt", "a plain scene")
    assert _discover_variants(scene_path) == ("default",)


def test_load_variant_thumbnails_falls_back_to_default(tmp_path: Path) -> None:
    # A bundle with only first_image.png (no per-variant images): every
    # requested variant should reuse the single default thumbnail.
    scene_path = tmp_path / "default_only.usdz"
    buf = io.BytesIO()
    Image.new("RGB", (64, 32), color=(10, 120, 200)).save(buf, format="PNG")
    with zipfile.ZipFile(scene_path, "w") as zf:
        zf.writestr("first_image.png", buf.getvalue())

    thumbs = _load_variant_thumbnails(scene_path, ("default", "1"))
    assert set(thumbs) == {"default", "1"}
    assert thumbs["1"] is thumbs["default"]


def test_load_variant_thumbnails_missing_images_returns_empty(tmp_path: Path) -> None:
    # No first_image*.png at all -> empty mapping (HUD draws text-only rows).
    scene_path = tmp_path / "empty.usdz"
    with zipfile.ZipFile(scene_path, "w") as zf:
        zf.writestr("metadata.yaml", "scene_id: x\n")
    assert _load_variant_thumbnails(scene_path, ("default",)) == {}
