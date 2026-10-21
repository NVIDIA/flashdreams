# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for prepared scene precision, alignment, and runtime independence."""

from __future__ import annotations

import subprocess
import sys
import sysconfig
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest
import torch
from cosmos3_nano_sim_auto.impl.conditioning.world_scenario import (
    prepared_scene as prepared,
)
from ludus_renderer import CUBE_FLAG_FLAT_FACES, PRIM_OBSTACLE, PRIM_TRAFFIC_LIGHT
from PIL import Image

pytestmark = pytest.mark.ci_cpu


@pytest.fixture
def bundle(tmp_path):
    cameras = list(prepared.CAMERAS[:7])
    maps = {
        "lines": [{"prim": 100, "vertices": [[[1.0000001192092896, 2, 3], [4, 5, 6]]]}],
        "triangles": [{"prim": 101, "vertices": [[[0, 0, 0], [1, 0, 0], [0, 1, 0]]]}],
    }
    static = {
        "timestamps_us": ["0", str(2**62 + 3)],
        "translations": [[0, 0, 1], [0, 0, 1]],
        "quaternions_xyzw": [[0, 0, 0, 1]] * 2,
        "scale": [4, 2, 1],
        "colors": [1, 0, 0, 0.5, 0, 0],
    }
    red = {**static, "timestamps_us": ["1000000", "1049999"]}
    green = {
        **static,
        "timestamps_us": ["1050000", "1100000"],
        "colors": [0, 1, 0, 0, 0.5, 0],
    }
    tracks = [
        {"prim": PRIM_OBSTACLE, "flags": 0, "tracks": [static]},
        {
            "prim": PRIM_TRAFFIC_LIGHT,
            "flags": CUBE_FLAG_FLAT_FACES,
            "tracks": [red, green],
        },
    ]
    poses = {}
    for i, name in enumerate(cameras):
        track = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
        track[:, 0, 3] = np.arange(4)
        track[:, 1, 3] = i * 0.25
        poses[name] = track.tolist()
    sample = {
        "prepared_scene": "scene.json",
        "scene_start": {
            "clip_id": "scene_1000000_1133333",
            "frame_index": 1,
            "timestamp_us": "1033333",
        },
        "name": "scene",
        "model_mode": "image2video",
        "resolution": "480",
        "prompt": "A sunny street.",
        "multiview": {
            "views": [
                {
                    "camera_key": name,
                    "control_path": f"controls/{name}.png",
                    "image_path": f"initial_rgb/{name}.png",
                }
                for name in cameras
            ]
        },
    }
    for filename, data in (
        ("map.json", maps),
        ("tracks.json", tracks),
        ("poses.json", poses),
        ("sample.json", sample),
    ):
        prepared.write_json(tmp_path / filename, data)
    for directory in ("controls", "initial_rgb"):
        (tmp_path / directory).mkdir()
        for i, name in enumerate(cameras):
            Image.new("RGB", (1280, 720), (i + 1, 20, 30)).save(
                tmp_path / directory / f"{name}.png"
            )
    manifest = {
        "format": prepared.FORMAT,
        "version": 1,
        "render_settings": prepared.RENDER_SETTINGS,
        "camera_order": cameras,
        "preparation_anchor": cameras[0],
        "coordinates": "metres; OpenCV RDF camera-to-world",
        "clip_id": "scene_1000000_1133333",
        "source_clip_id": "scene",
        "frames": 4,
        "opening_frame": 1,
        "start_us": "1000000",
        "end_us": "1133333",
        "palette": {"100": [255, 255, 255, 255], "101": [0, 255, 0, 255]},
        "widths": {"100": 12.0},
        "cameras": {
            name: {
                "intrinsics": [960, 540, 1924, 1084, 0, 0.001, 0, 0, 0, 0, 1, 1, 0, 0]
            }
            for name in cameras
        },
        "files_sha256": {
            str(p.relative_to(tmp_path)): prepared.sha256(p)
            for p in sorted(tmp_path.rglob("*"))
            if p.is_file()
        },
    }
    prepared.write_json(tmp_path / "scene.json", manifest)
    return tmp_path


def rewrite(root, name, data):
    prepared.write_json(root / name, data)
    if name != "scene.json":
        manifest = prepared.read_json(root / "scene.json")
        manifest["files_sha256"][name] = prepared.sha256(root / name)
        prepared.write_json(root / "scene.json", manifest)


def test_round_trip_preserves_float_bits_timestamps_and_light_lifetimes(bundle):
    scene = prepared.load(bundle)
    assert prepared.map_data(scene.scene) == prepared.read_json(bundle / "map.json")
    assert prepared.track_data(scene.scene) == prepared.read_json(
        bundle / "tracks.json"
    )
    assert scene.scene.polyline_pools[0].vertices.dtype == torch.float32
    assert scene.scene.polyline_pools[0].vertices[0, 0].item() == 1.0000001192092896
    assert scene.scene.cube_pools is not None
    assert scene.scene.cube_pools[0].track_timestamps_us[-1].item() == 2**62 + 3
    assert scene.scene.cube_pools[1].track_timestamps_us.tolist() == [
        1000000,
        1049999,
        1050000,
        1100000,
    ]
    assert scene.scene.cube_pools[1].cube_ts_prefix_sum.tolist() == [2, 4]
    assert scene.frame_mapping(0, 2).tolist() == [1, 2]
    with pytest.raises(ValueError, match="sample frame 0"):
        scene.frame_mapping(1, None)
    with pytest.raises(ValueError, match="3 frames"):
        scene.frame_mapping(0, 4)


@pytest.mark.parametrize(
    "problem",
    [
        "hash",
        "version",
        "style",
        "pose",
        "start",
        "camera",
        "nan",
        "timestamp",
        "flags",
        "image",
    ],
)
def test_corrupt_bundle_is_refused(bundle, problem):
    name = "scene.json"
    data = prepared.read_json(bundle / name)
    if problem == "hash":
        (bundle / "map.json").write_text("{}")
    elif problem == "image":
        image_name = f"initial_rgb/{prepared.CAMERAS[0]}.png"
        Image.new("RGB", (32, 32)).save(bundle / image_name)
        data["files_sha256"][image_name] = prepared.sha256(bundle / image_name)
    elif problem == "version":
        data["version"] = 2
    elif problem == "style":
        del data["widths"]["100"]
    elif problem == "pose":
        name = "poses.json"
        data = prepared.read_json(bundle / name)
        data[prepared.CAMERAS[0]][1][0][0] = 2
    elif problem in ("start", "camera"):
        name = "sample.json"
        data = prepared.read_json(bundle / name)
        if problem == "start":
            data["scene_start"]["frame_index"] = 0
        else:
            data["multiview"]["views"].reverse()
    elif problem == "nan":
        (bundle / "scene.json").write_text('{"format":NaN}')
        with pytest.raises(ValueError, match="Nonfinite"):
            prepared.load(bundle)
        return
    elif problem in ("timestamp", "flags"):
        name = "tracks.json"
        data = prepared.read_json(bundle / name)
        if problem == "timestamp":
            data[0]["tracks"][0]["timestamps_us"][1] = 2**62
        else:
            data[0]["flags"] = 4096
    rewrite(bundle, name, data)
    with pytest.raises(ValueError):
        prepared.load(bundle)


def test_sample_paths_remain_relative_on_disk_and_resolve_from_another_directory(
    bundle, monkeypatch, tmp_path
):
    from cosmos3_nano_sim_auto.inputs import MultiviewSample

    monkeypatch.chdir(tmp_path.parent)
    sample = MultiviewSample.load(bundle / "sample.json")
    assert sample.cameras == prepared.CAMERAS[:7]
    assert sample.image_paths[0] == str(
        bundle / "initial_rgb" / f"{sample.cameras[0]}.png"
    )
    assert prepared.read_json(bundle / "sample.json")["multiview"]["views"][0][
        "image_path"
    ].startswith("initial_rgb/")
    bad = prepared.read_json(bundle / "sample.json")
    bad["multiview"]["views"][0]["image_path"] = "../outside.png"
    rewrite(bundle, "sample.json", bad)
    with pytest.raises(ValueError, match="inside"):
        MultiviewSample.load(bundle / "sample.json")


@pytest.mark.parametrize("version", [1, 2])
def test_runtime_import_and_load_with_public_packages(bundle, version):
    if version == 2:
        add_lidar(bundle)
    script = """
import sys
from pathlib import Path
from cosmos3_nano_sim_auto.impl.conditioning.world_scenario.prepared_scene import load
from cosmos3_nano_sim_auto.impl.conditioning.ludus_controls import open_clip
from cosmos3_nano_sim_auto.apps.interactive_drive.scene import load_scene_bundle
from cosmos3_nano_sim_auto.apps.interactive_drive.backend import MultiViewRenderBackend
scene = load(Path(sys.argv[1]))
assert len(scene.tracks) == 7
if scene.lidar is not None:
    from types import SimpleNamespace
    import numpy as np
    from cosmos_interactive.lidar import LidarTokenizer
    from cosmos3_nano_sim_auto.apps.interactive_drive.lidar import make_live_lidar
    LidarTokenizer.from_checkpoint = lambda *a, **kw: LidarTokenizer(None)  # no weights
    app_scene = SimpleNamespace(metadata=dict(prepared_scene=scene, pose_frame=1), initial_rig_to_world=np.eye(4))
    assert make_live_lidar(app_scene, checkpoint='unused', patch_dim=128, device='cpu') is not None
"""
    # -I -S exposes only these public workspace sources and installed libraries.
    paths = [sysconfig.get_path("purelib"), sysconfig.get_path("platlib")]
    for name in (
        "flashdreams.core",
        "cosmos_interactive",
        "cosmos3_nano_sim_auto",
        "interactive_drive",
        "ludus_renderer",
    ):
        spec = find_spec(name)
        assert spec is not None and spec.origin is not None
        paths.append(str(Path(spec.origin).resolve().parents[len(name.split("."))]))
    script = f"import sys\nsys.path[:0] = {paths!r}\n" + script
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(bundle)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def add_lidar(root):
    from .test_prepared_lidar import lidar_data

    data = lidar_data(prepared.CAMERAS[0])
    data["sensor_to_ego"][2][3] = 2.0
    manifest = prepared.read_json(root / "scene.json")
    manifest["version"] = 2
    rewrite(root, "scene.json", manifest)
    rewrite(root, "lidar.json", data)


@pytest.mark.parametrize("optimized", [False, True])
def test_v2_calibration_metadata_is_camera_independent_and_hash_checked(
    bundle, monkeypatch, optimized
):
    from types import SimpleNamespace

    from cosmos3_nano_sim_auto.apps.interactive_drive import lidar
    from cosmos_interactive.lidar import LidarTokenizer

    add_lidar(bundle)
    scene = prepared.load(bundle)
    monkeypatch.setattr(
        LidarTokenizer, "from_checkpoint", lambda *a, **kw: LidarTokenizer(None)
    )
    worlds = []
    for camera in prepared.CAMERAS[:2]:
        anchor = scene.tracks[camera][1]
        app_scene = SimpleNamespace(
            metadata=dict(prepared_scene=scene, pose_frame=1),
            initial_rig_to_world=anchor,
        )
        stream = lidar.make_live_lidar(
            app_scene,
            checkpoint="unused",
            patch_dim=128,
            device="cpu",
            optimized=optimized,
        )
        worlds.append(anchor @ stream._sensor_to_rig.astype(np.float64))
        assert scene.lidar is not None
        np.testing.assert_allclose(scene.lidar.sensor_to_ego[:3, 3], [0, 0, 2])
    np.testing.assert_allclose(worlds[0], worlds[1])
    np.testing.assert_allclose(worlds[0][:3, 3], [1, 0, 2])
    (bundle / "lidar.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepared.load(bundle)


def test_v1_lidar_request_fails_before_checkpoint_load(bundle, monkeypatch):
    from types import SimpleNamespace

    from cosmos3_nano_sim_auto.apps.interactive_drive import lidar

    scene = SimpleNamespace(metadata=dict(prepared_scene=prepared.load(bundle)))
    monkeypatch.setattr(
        lidar.LidarTokenizer,
        "from_checkpoint",
        lambda *a, **kw: pytest.fail("Loaded a checkpoint before validating scene"),
    )
    with pytest.raises(ValueError, match="v2 prepared scene"):
        lidar.make_live_lidar(scene, checkpoint="unused", patch_dim=128, device="cpu")


def test_interactive_scene_uses_recorded_opening_and_shared_cpu_geometry(
    bundle, monkeypatch
):
    from cosmos3_nano_sim_auto.apps.interactive_drive import scene as app
    from interactive_drive.config import AppConfig, RasterConfig

    read = []

    def opening(sample, names, *, start_frame, raster):
        read.append(start_frame)
        return tuple(np.zeros((8, 8, 3), np.uint8) for _ in names)

    monkeypatch.setattr(app, "_opening_frames", opening)
    names = prepared.CAMERAS[:2]
    result = app.load_scene_bundle(
        bundle,
        names,
        "default",
        None,
        RasterConfig(),
        config=AppConfig(scene_path=bundle),
    )
    assert result.initial_timestamp_us == 1033333
    assert result.metadata["pose_frame"] == 1
    assert result.metadata["start_frame"] == 0
    assert result.prompt == "A sunny street."
    assert len(result.initial_rgbs) == 2 and read == [0]
    assert isinstance(result.metadata["prepared_scene"], prepared.PreparedScene)
    np.testing.assert_array_equal(result.initial_rig_to_world[:3, 3], [1, 0, 0])
    np.testing.assert_allclose(
        result.cameras[1].sensor_to_rig_flu[:3, 3], [0, 0, -0.25]
    )


def test_open_clip_uses_saved_mapping_settings_and_does_not_reload(bundle, monkeypatch):
    import ludus_renderer
    from cosmos3_nano_sim_auto.impl.conditioning.ludus_controls import open_clip

    saved = prepared.load(bundle)
    events = {}

    class Context:
        def __init__(self, **kwargs):
            pass

        def __getattr__(self, name):
            return lambda value: events.update({name: value})

        def upload_scene(self, scene):
            events["scene"] = scene
            return 42

    monkeypatch.setattr(ludus_renderer, "LudusCudaTimestampedContext", Context)
    monkeypatch.setattr(prepared, "load", lambda _: pytest.fail("Parsed twice"))
    drawn = open_clip(
        bundle,
        prepared.CAMERAS[:2],
        device="cpu",
        against=bundle,
        wanted=2,
        prepared=saved,
        requested_fps=30,
    )
    assert drawn.frames == 2
    assert drawn._map.tolist() == [1, 2]
    assert events["set_max_extrapolation_us"] == 0
    assert events["set_msaa_samples"] == 4
    assert events["scene"].cube_pools[0].track_timestamps_us[-1].item() == 2**62 + 3
    with pytest.raises(ValueError, match="requires .* FPS"):
        open_clip(bundle, prepared.CAMERAS[:2], prepared=saved, requested_fps=24)
    with pytest.raises(ValueError, match="camera is absent"):
        open_clip(bundle, prepared.CAMERAS[7:8], prepared=saved)
    with pytest.raises(ValueError, match="own matching"):
        open_clip(
            bundle,
            prepared.CAMERAS[:2],
            device="cpu",
            against=bundle.parent,
            prepared=saved,
        )


def test_still_frame_loading_does_not_decode_video(bundle, monkeypatch):
    from cosmos3_nano_sim_auto.impl.conditioning.controls import loaded_control_clips

    from flashdreams.infra import runner_io

    called = []

    def load(path, **kwargs):
        called.append(path)
        return torch.zeros(1, 3, 8, 8)

    monkeypatch.setattr(runner_io, "load_first_frame_tensor", load)
    monkeypatch.setattr(
        runner_io,
        "load_video_tensor",
        lambda *_a, **_k: pytest.fail("Tried to decode PNG as video"),
    )
    path = bundle / "initial_rgb" / f"{prepared.CAMERAS[0]}.png"
    [clip] = loaded_control_clips([path], pixel_frames=1, pixel_height=8, pixel_width=8)
    assert clip.shape == (1, 3, 8, 8) and called == [path]
    with pytest.raises(ValueError, match="exactly one"):
        loaded_control_clips([path], pixel_frames=2, pixel_height=8, pixel_width=8)


def test_recorded_cli_uses_prepared_default_views_and_pose_length(bundle):
    from cosmos3_nano_sim_auto.impl import app

    args = app._parse(["--scene", str(bundle), "--frames", "0"])
    sample = app._sample_for(args)
    assert args.views == 7
    assert sample.num_views == 7
    assert app._frames_the_controls_allow(sample, args) == 1
    explicit = app._parse(["--scene", str(bundle), "--views", "11"])
    app._sample_for(explicit)
    assert explicit.views == 11


def test_prepared_recorded_captions_keep_their_window(bundle):
    record = prepared.read_json(bundle / "sample.json")
    record["frame_range"] = [0, 3]
    for view in record["multiview"]["views"]:
        view["caption"] = "A test street."
    rewrite(bundle, "sample.json", record)
    prepared.load(bundle)
    from cosmos3_nano_sim_auto.inputs import MultiviewSample

    assert MultiviewSample.load(bundle / "sample.json").frame_range == (0, 3)
    record["frame_range"] = [1, 3]
    rewrite(bundle, "sample.json", record)
    with pytest.raises(ValueError, match="recorded opening"):
        prepared.load(bundle)


def test_prepared_recorded_scene_preserves_per_camera_captions(bundle, monkeypatch):
    from cosmos3_nano_sim_auto.apps.interactive_drive import scene
    from interactive_drive.config import RasterConfig

    sample = prepared.read_json(bundle / "sample.json")
    for view in sample["multiview"]["views"]:
        view["caption"] = "This camera's recorded caption."
    rewrite(bundle, "sample.json", sample)
    monkeypatch.setattr(
        scene,
        "_opening_frames",
        lambda *a, **kw: (np.zeros((1, 1, 3), dtype=np.uint8),) * 7,
    )
    loaded = scene._prepared_bundle(
        bundle, prepared.CAMERAS[:7], None, RasterConfig(), None
    )
    assert loaded.prompt == ""  # The backend keeps the sample's per-camera captions.
    overridden = scene._prepared_bundle(
        bundle, prepared.CAMERAS[:7], "A new prompt", RasterConfig(), None
    )
    assert overridden.prompt == "A new prompt"


@pytest.mark.parametrize(
    "name",
    [None, "clipgt-residential-rain-iter1500", "clipgt-residential-snow-iter1500"],
)
def test_recorded_cli_resolves_hosted_scene_and_matching_sample(
    bundle, monkeypatch, name
):
    from cosmos3_nano_sim_auto.impl import app

    requested = []

    def prepare(scene):
        requested.append(scene)
        return bundle

    monkeypatch.setattr(app, "prepare_scene", prepare)
    args = app._parse([] if name is None else ["--scene", name])
    sample = app._sample_for(args)
    assert requested == [Path(name or "clipgt-residential-iter1500")]
    assert args.draw_map == bundle
    assert args.sample == str(bundle)
    assert sample.num_views == 7


def test_explicit_sample_does_not_download_a_scene(bundle, monkeypatch):
    from cosmos3_nano_sim_auto.impl import app

    monkeypatch.setattr(
        app, "prepare_scene", lambda *_a: pytest.fail("Unexpected download")
    )
    args = app._parse(["--sample", str(bundle)])
    assert app._sample_for(args).num_views == 7
    assert args.draw_map is None
