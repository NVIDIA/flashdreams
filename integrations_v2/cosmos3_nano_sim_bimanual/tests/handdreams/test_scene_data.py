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

"""Local scene validation and conditioning contracts without model weights."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from cosmos3_nano_sim_bimanual.apps.handdreams import scenes
from PIL import Image

pytestmark = pytest.mark.ci_cpu


def _metadata() -> dict:
    return {
        "schema_version": 1,
        "title": "Synthetic scene",
        "prompt": "Move the synthetic object",
        "fps": 30.0,
        "image_size": [720, 1280, 360, 640],
        "wrist_poses": {side: np.eye(4).tolist() for side in ("left", "right")},
        "gripper_open": {"left": 0.25, "right": 0.75},
    }


def _scene_directory(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "scene.json").write_text(json.dumps(_metadata()), encoding="utf-8")
    Image.new("RGB", (1280, 720), (64, 64, 64)).save(path / "first_frame.jpg")
    return path


def test_external_scene_preserves_initial_conditioning(tmp_path, monkeypatch) -> None:
    path = _scene_directory(tmp_path / "synthetic")
    expected = _metadata()
    encoded = []

    def encode(frame: torch.Tensor) -> torch.Tensor:
        encoded.append(frame.clone())
        return torch.zeros(1, 128)

    def patchify(latent: torch.Tensor, **geometry: int) -> torch.Tensor:
        assert geometry == {"content_height": 360, "content_width": 640}
        return latent

    def tokenize(prompt: str) -> torch.Tensor:
        assert prompt == expected["prompt"]
        return torch.tensor([11, 12])

    monkeypatch.setattr(scenes, "tokenize_prompt", tokenize)
    scene = scenes.load_scene(
        "synthetic", path, encode=encode, patchify=patchify, device="cpu"
    )
    assert len(encoded) == 1
    canvas = encoded[0].numpy()
    assert canvas.shape == (480, 832, 3)
    assert canvas.dtype == np.uint8
    np.testing.assert_array_equal(canvas, 64)
    assert scene.init_rgb.shape == (352, 640, 3)
    np.testing.assert_array_equal(scene.init_rgb, canvas[:352, :640])
    assert (scene.content_height, scene.content_width) == (360, 640)
    assert scene.init_latent.shape == (1, 128)
    assert scene.fps == 30.0
    assert scene.gripper_open == expected["gripper_open"]
    assert not hasattr(scene, "replay_actions")
    np.testing.assert_array_equal(scene.text_ids.numpy(), [11, 12])
    for side in ("left", "right"):
        pose = scene.wrist_poses[side]
        assert pose.dtype == np.float32
        np.testing.assert_array_equal(pose, np.eye(4))
        np.testing.assert_array_equal(scene.wrist_rotations[side], pose[:3, :3])
        assert not np.shares_memory(scene.wrist_rotations[side], pose)
    reloaded = scenes.load_scene(
        "synthetic", path, encode=encode, patchify=patchify, device="cpu"
    )
    assert reloaded is not scene
    assert len(encoded) == 2
    np.testing.assert_array_equal(reloaded.init_rgb, scene.init_rgb)


@pytest.mark.parametrize("pattern", ["gradient", "checkerboard", "impulse", "noise"])
def test_resize_matches_training_uint8_tensor_contract(pattern: str) -> None:
    rows, columns = np.indices((480, 640))
    if pattern == "gradient":
        source = np.stack(
            (rows % 256, columns % 256, (rows + columns) % 256), axis=-1
        ).astype(np.uint8)
    elif pattern == "checkerboard":
        source = np.repeat((((rows + columns) % 2) * 255)[..., None], 3, axis=-1)
        source = source.astype(np.uint8)
    elif pattern == "impulse":
        source = np.zeros((480, 640, 3), dtype=np.uint8)
        source[::11, ::17] = [255, 127, 1]
    else:
        source = np.random.default_rng(4).integers(
            0, 256, (480, 640, 3), dtype=np.uint8
        )
    before = source.copy()
    # torchvision's tensor Resize casts uint8 to float, interpolates, then
    # clamps/rounds back to uint8. Native uint8 interpolation is a different path.
    expected = torch.nn.functional.interpolate(
        torch.tensor(source, dtype=torch.float32).permute(2, 0, 1)[None],
        size=(360, 640),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    expected = expected.clamp(0, 255).round().to(torch.uint8)[0].permute(1, 2, 0)

    actual = scenes._resize_rgb_content(source, height=360, width=640)

    assert actual.shape == (360, 640, 3)
    assert actual.dtype == np.uint8
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected.numpy())
    np.testing.assert_array_equal(source, before)


def test_runtime_canvas_matches_training_reflection_padding() -> None:
    rows, columns = np.indices((360, 640))
    content = np.stack((rows % 256, columns % 256, (rows + columns) % 256), axis=-1)
    content = content.astype(np.uint8)
    source = np.full((720, 1280, 3), 17, dtype=np.uint8)
    source[:360, :640] = content

    actual = scenes._prepare_runtime_canvas(
        source, source_content_height=360, source_content_width=640
    )
    expected = torch.nn.functional.pad(
        torch.from_numpy(content).permute(2, 0, 1),
        (0, 192, 0, 120),
        mode="reflect",
    ).permute(1, 2, 0)

    assert actual.shape == (480, 832, 3)
    assert actual.dtype == np.uint8
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, expected.numpy())
    np.testing.assert_array_equal(actual[:360, :640], content)
    np.testing.assert_array_equal(actual[360, 640], content[358, 638])
    np.testing.assert_array_equal(actual[-1, -1], content[239, 447])


def test_scene_previews_use_external_titles_and_first_frames(tmp_path) -> None:
    _scene_directory(tmp_path / "second")
    _scene_directory(tmp_path / "first")
    previews = scenes.scene_previews(scenes.discover_scenes(sample_dir=tmp_path))
    assert tuple(preview.name for preview in previews) == ("first", "second")
    for preview in previews:
        assert preview.title == "Synthetic scene"
        assert preview.thumbnail is not None
        assert preview.thumbnail.shape == (352, 640, 3)
        assert not preview.thumbnail.flags.writeable
        np.testing.assert_array_equal(preview.thumbnail, 64)


def test_scene_display_crops_bottom_without_rescaling_source(monkeypatch) -> None:
    rows, columns = np.indices((360, 640))
    content = np.stack((rows % 256, columns % 256, (rows + columns) % 256), axis=-1)
    content = content.astype(np.uint8)
    metadata = _metadata()
    metadata["image_size"] = [360, 640, 360, 640]
    sample = scenes.validate_scene_sample(metadata, init_rgb=content)
    monkeypatch.setattr(scenes, "load_scene_data", lambda _: sample)
    monkeypatch.setattr(scenes, "tokenize_prompt", lambda _: torch.tensor([11, 12]))
    encoded = []

    def encode(frame: torch.Tensor) -> torch.Tensor:
        encoded.append(frame.clone())
        return torch.zeros(1, 128)

    def patchify(latent: torch.Tensor, **geometry: int) -> torch.Tensor:
        assert geometry == {"content_height": 360, "content_width": 640}
        return latent

    scene = scenes.load_scene(
        "synthetic", "unused", encode=encode, patchify=patchify, device="cpu"
    )
    preview = scenes.scene_previews([("synthetic", "unused")])[0]

    assert (scene.content_height, scene.content_width) == (360, 640)
    np.testing.assert_array_equal(encoded[0].numpy()[:360, :640], content)
    np.testing.assert_array_equal(scene.init_rgb, content[:352])
    np.testing.assert_array_equal(preview.thumbnail, content[:352])
    for side in ("left", "right"):
        np.testing.assert_array_equal(
            scene.wrist_poses[side], sample["wrist_poses"][side]
        )
    assert scene.gripper_open == sample["gripper_open"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("schema_version", 1.0),
        ("title", ""),
        ("title", None),
        ("prompt", "  "),
        ("prompt", ["text"]),
        ("fps", 0),
        ("fps", -1),
        ("fps", float("nan")),
        ("fps", float("inf")),
        ("fps", True),
        ("fps", "30"),
        ("image_size", [720, 1280, 360]),
        ("image_size", [720, 1280, 0, 640]),
        ("image_size", [720, 1280, 721, 640]),
        ("image_size", [720, 1280, 360, 1281]),
        ("image_size", [721, 1280, 360, 640]),
        ("image_size", [720, 1280, 360.5, 640]),
        ("image_size", [720, 1280, True, 640]),
        ("image_size", [720, 1280, float("nan"), 640]),
        ("wrist_poses", {"left": np.eye(4).tolist()}),
        ("gripper_open", {"left": 0.5}),
        ("gripper_open", []),
    ],
)
def test_scene_validation_rejects_malformed_metadata(field: str, value) -> None:
    data = _metadata()
    data[field] = value
    with pytest.raises(ValueError, match=field):
        scenes.validate_scene_sample(
            data, init_rgb=np.zeros((720, 1280, 3), dtype=np.uint8)
        )


@pytest.mark.parametrize("field", ["left", "right"])
@pytest.mark.parametrize(
    "value", [None, True, "0.5", -0.01, 1.01, float("nan"), float("inf")]
)
def test_scene_validation_requires_valid_grippers(field: str, value) -> None:
    data = _metadata()
    data["gripper_open"][field] = value
    with pytest.raises(ValueError, match="gripper_open"):
        scenes.validate_scene_sample(
            data, init_rgb=np.zeros((720, 1280, 3), dtype=np.uint8)
        )


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize(
    "fault", ["missing", "shape", "nan", "bottom", "scale", "reflection"]
)
def test_scene_validation_requires_rigid_wrist_poses(side: str, fault: str) -> None:
    data = _metadata()
    pose = np.eye(4)
    if fault == "missing":
        del data["wrist_poses"][side]
    else:
        if fault == "shape":
            pose = pose[:3]
        elif fault == "nan":
            pose[0, 3] = float("nan")
        elif fault == "bottom":
            pose[3, 0] = 1.0
        elif fault == "scale":
            pose[0, 0] = 2.0
        else:
            pose[0, 0] = -1.0
        data["wrist_poses"][side] = pose.tolist()
    with pytest.raises(ValueError, match="wrist_poses"):
        scenes.validate_scene_sample(
            data, init_rgb=np.zeros((720, 1280, 3), dtype=np.uint8)
        )


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "title",
        "prompt",
        "fps",
        "image_size",
        "wrist_poses",
        "gripper_open",
    ],
)
def test_scene_validation_requires_all_metadata(field: str) -> None:
    data = _metadata()
    del data[field]
    with pytest.raises(ValueError, match=field):
        scenes.validate_scene_sample(
            data, init_rgb=np.zeros((720, 1280, 3), dtype=np.uint8)
        )


@pytest.mark.parametrize("field", ["action", "video", "text_token_ids", "initial_pose"])
def test_scene_validation_rejects_recording_fields(field: str) -> None:
    data = _metadata()
    data[field] = []
    with pytest.raises(ValueError, match="unexpected"):
        scenes.validate_scene_sample(
            data, init_rgb=np.zeros((720, 1280, 3), dtype=np.uint8)
        )


def test_scene_validation_preserves_prompt_and_calibration() -> None:
    data = _metadata()
    data["prompt"] = "  Keep original spacing.\n"
    before = copy.deepcopy(data)
    valid = scenes.validate_scene_sample(
        data, init_rgb=np.zeros((720, 1280, 3), dtype=np.uint8)
    )
    assert data == before
    assert valid["prompt"] == data["prompt"]
    for side in ("left", "right"):
        np.testing.assert_array_equal(
            valid["wrist_poses"][side],
            np.asarray(data["wrist_poses"][side], dtype=np.float32),
        )


def test_scene_load_validates_before_encoding(tmp_path: Path) -> None:
    local = _scene_directory(tmp_path / "local")
    data = _metadata()
    data["fps"] = -1
    (local / "scene.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="fps"):
        scenes.load_scene(
            "local",
            local,
            encode=lambda _: pytest.fail("invalid scene reached the model"),
            patchify=lambda *_args, **_kwargs: pytest.fail("invalid latent"),
            device="cpu",
        )


@pytest.mark.parametrize("fault", ["json", "jpeg", "object", "mismatch"])
def test_scene_load_rejects_invalid_files(tmp_path: Path, fault: str) -> None:
    local = _scene_directory(tmp_path / "local")
    if fault == "json":
        (local / "scene.json").write_text("{", encoding="utf-8")
    elif fault == "jpeg":
        (local / "first_frame.jpg").write_bytes(b"invalid")
    elif fault == "object":
        (local / "scene.json").write_text("[]", encoding="utf-8")
    else:
        data = _metadata()
        data["image_size"][0] += 1
        (local / "scene.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        scenes.load_scene_data(local)


def test_scene_directory_rejects_ambiguous_and_absent_names(tmp_path: Path) -> None:
    _scene_directory(tmp_path / "first" / "duplicate")
    _scene_directory(tmp_path / "second" / "duplicate")
    with pytest.raises(ValueError, match="duplicate"):
        scenes.discover_scenes(sample_dir=tmp_path)
    with pytest.raises(ValueError, match="absent"):
        scenes.discover_scenes(sample_dir=tmp_path / "first", samples=("absent",))


def test_scene_sources_reject_conflicting_overrides(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        scenes.discover_scenes(scene_path=tmp_path, sample_dir=tmp_path)


def test_missing_source_fails_without_downloading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda **kwargs: pytest.fail("scene lookup attempted a download"),
    )
    with pytest.raises(ValueError, match="No scenarios are packaged.*--sample-dir"):
        scenes.discover_scenes()


@pytest.mark.parametrize("missing", [False, True])
def test_empty_or_missing_catalog_fails(tmp_path: Path, missing: bool) -> None:
    with pytest.raises(ValueError, match="does not exist|no scene.json"):
        scenes.discover_scenes(sample_dir=tmp_path / "missing" if missing else tmp_path)


def test_explicit_local_scene_overrides(tmp_path: Path) -> None:
    local = _scene_directory(tmp_path / "custom")
    assert scenes.discover_scenes(scene_path=local) == [("custom", str(local))]
    assert scenes.discover_scenes(scene_path=local, sample="alias") == [
        ("alias", str(local))
    ]
    assert scenes.discover_scenes(sample_dir=tmp_path) == [("custom", str(local))]
    assert scenes.discover_scenes(sample_dir=tmp_path, samples=("custom",)) == [
        ("custom", str(local))
    ]


def test_scene_selection_preserves_exact_names_and_order(tmp_path: Path) -> None:
    names = ("shirt", "fold,shirt", "fold", " fold shirt ")
    for name in names:
        _scene_directory(tmp_path / name)
    assert scenes.discover_scenes(sample_dir=tmp_path, samples=names) == [
        (name, str(tmp_path / name)) for name in names
    ]
    assert scenes.discover_scenes(sample_dir=tmp_path, samples=()) == []


@pytest.mark.parametrize("filename", ["scene.json", "first_frame.jpg"])
def test_incomplete_scene_directory_is_rejected(tmp_path: Path, filename: str) -> None:
    (tmp_path / filename).touch()
    with pytest.raises(FileNotFoundError, match="scene is missing"):
        scenes.resolve_scene_path(tmp_path)
