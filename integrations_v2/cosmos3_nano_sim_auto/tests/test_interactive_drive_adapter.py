# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Our integration plugging into the shared Interactive Drive app."""

from __future__ import annotations

from pathlib import Path

import pytest
from cosmos3_nano_sim_auto.apps.interactive_drive.adapter import (
    DEFAULT_SAMPLE,
    DEFAULT_SCENE,
    MODEL_HEIGHT,
    MODEL_WIDTH,
    create_app,
    defaults,
)
from cosmos3_nano_sim_auto.apps.interactive_drive.backend import MultiViewRenderBackend
from cosmos3_nano_sim_auto.apps.interactive_drive.scene import load_scene_bundle
from cosmos3_nano_sim_auto.impl.conditioning.camera_rig import CAMERAS, view_ids
from interactive_drive import InteractiveDriveApplication

from flashdreams.core.visualization.lidar import LidarView
from flashdreams.runtime_v2.session_desc import BackpressureMode, PresentationMode

pytestmark = pytest.mark.ci_cpu


def test_the_app_is_sized_for_what_the_checkpoint_generates() -> None:
    """Anything else would present an upscale of the model's own output."""
    built = defaults()

    assert (built.width, built.height) == (MODEL_WIDTH, MODEL_HEIGHT)
    assert built.fps == 30


def test_the_defaults_name_our_backend_and_not_a_pipeline() -> None:
    built = defaults()

    assert built.backend_factory is not None
    assert built.pipeline_config is None


def test_defaults_select_the_hosted_residential_bundle() -> None:
    built = defaults()

    assert (built.scene, built.sample) == (DEFAULT_SCENE, DEFAULT_SAMPLE)
    assert built.scene == Path("clipgt-residential-iter1500")


def test_colliders_are_off_for_a_scene_the_host_cannot_read() -> None:
    """Our geometry stays on the GPU, so a collider world would be empty."""
    assert defaults().physics is False


def test_reverse_gear_is_enabled() -> None:
    assert defaults().reverse is True


def test_no_overhead_view_is_claimed_for_a_rig_we_render_ourselves() -> None:
    """The app's BEV is a camera its own rasterizer appends, and we are the rig.

    Left on, the HUD's map panel would read our second camera as a top-down
    view, or stand empty at one view.
    """
    assert defaults().bev is False


def test_a_drive_with_no_end_declares_no_length() -> None:
    """The app stops asking for chunks at ``total_blocks``, and we never stop."""
    assert defaults().total_blocks == 0


def test_creating_the_app_wires_our_scene_loader_in() -> None:
    """A USDZ loader would have nothing our map builder can read."""
    app = create_app()

    assert isinstance(app, InteractiveDriveApplication)
    assert app._scene_loader is load_scene_bundle


@pytest.mark.parametrize(
    "name",
    [None, "clipgt-residential-rain-iter1500", "clipgt-residential-snow-iter1500"],
)
def test_init_prepares_all_scenes_and_selects_matching_sample(
    monkeypatch, tmp_path, name
):
    from cosmos3_nano_sim_auto.apps.interactive_drive import adapter
    from cosmos3_nano_sim_auto.scenes import SCENE_NAMES

    prepared = []

    def prepare(scene):
        prepared.append(str(scene))
        path = tmp_path / scene
        path.mkdir(exist_ok=True)
        return path

    monkeypatch.setattr(adapter, "prepare_scene", prepare)
    monkeypatch.setattr(adapter, "build_backend", lambda *_a, **_k: object())
    app = create_app()
    assert isinstance(app, adapter.CosmosInteractiveDriveApplication)
    app.init(["--no-ui", *([] if name is None else ["--scene", name])])
    assert prepared == list(SCENE_NAMES)
    selected = tmp_path / (name or SCENE_NAMES[0])
    assert app._config is not None
    assert app._config.app.scene_path == selected
    assert app._config.app.sample_path == selected
    assert [option.path for option in app._interactive_scene_options] == [
        tmp_path / item for item in SCENE_NAMES
    ]


def test_explicit_local_scene_and_help_do_not_download(monkeypatch, tmp_path):
    from cosmos3_nano_sim_auto.apps.interactive_drive import adapter

    monkeypatch.setattr(
        adapter, "prepare_scene", lambda *_a: pytest.fail("Unexpected download")
    )
    monkeypatch.setattr(adapter, "build_backend", lambda *_a, **_k: object())
    app = create_app()
    assert isinstance(app, adapter.CosmosInteractiveDriveApplication)
    app.init(["--scene", str(tmp_path), "--no-ui"])
    assert app._config is not None
    assert app._config.app.scene_path == tmp_path
    with pytest.raises(SystemExit) as raised:
        create_app().init(["--help"])
    assert raised.value.code == 0


@pytest.mark.parametrize(
    "views,ids", [(1, (0,)), (4, (0, 1, 3, 5)), (7, tuple(range(7)))]
)
@pytest.mark.parametrize("lidar", [False, True])
def test_view_presets_preserve_camera_identity_with_optional_lidar(
    monkeypatch, tmp_path, views, ids, lidar
):
    from cosmos3_nano_sim_auto import checkpoint

    codec = tmp_path / "codec.safetensors"
    monkeypatch.setattr(checkpoint, "resolve_lidar_checkpoint", lambda *a, **k: codec)
    app = create_app()
    app.init(
        [
            "--scene",
            str(tmp_path),
            "--views",
            str(views),
            *(["--lidar"] if lidar else []),
        ]
    )
    assert app._config is not None
    config = app._config.app
    assert view_ids(config.camera_names) == ids
    assert config.presentation.cells >= views + int(lidar)
    assert isinstance(app._backend, MultiViewRenderBackend)
    assert app._backend._settings.lidar_checkpoint == (codec if lidar else None)
    assert app._backend._settings.lidar_view == LidarView.POINT_CLOUD


def test_short_launch_matches_explicit_launch_defaults(tmp_path):
    short, explicit = create_app(), create_app()
    short.init(["--scene", str(tmp_path)])
    explicit.init(
        [
            "--scene",
            str(tmp_path),
            "--camera",
            "camera_front_wide_120fov",
            "--start-frame",
            "0",
            "--world-model-seed",
            "0",
            "--compile-denoise",
            "off",
        ]
    )
    assert short._config is not None and explicit._config is not None
    assert (
        short._config.app.camera_names
        == explicit._config.app.camera_names
        == (CAMERAS[0],)
    )
    assert short._config.app.start_frame == explicit._config.app.start_frame == 0
    assert isinstance(short._backend, MultiViewRenderBackend)
    assert isinstance(explicit._backend, MultiViewRenderBackend)
    assert short._backend._settings == explicit._backend._settings
    assert short._backend._settings.seed == 0
    assert short._backend._settings.compile_denoise is None
    assert not short._config.no_ui
    assert short._config.view_mode == "rgb"
    assert short.session_desc() == explicit.session_desc()
    assert short.session_desc().backpressure_mode == BackpressureMode.BLOCK
    assert short.session_desc().presentation_mode == PresentationMode.CONTINUOUS


@pytest.mark.parametrize(
    "selection", ["--views=4", f"--camera={CAMERAS[0]},{CAMERAS[3]}"]
)
def test_camera_selection_preserves_other_explicit_options(tmp_path, selection):
    app = create_app()
    app.init(
        [
            "--scene",
            str(tmp_path),
            selection,
            "--view",
            "hdmap",
            "--no-ui",
            "--world-model-seed",
            "37",
            "--start-frame",
            "4",
            "--compile-denoise",
            "default",
        ]
    )
    assert app._config is not None
    assert app._config.app.camera_names == (
        tuple(CAMERAS[index] for index in (0, 1, 3, 5))
        if selection.startswith("--views")
        else (CAMERAS[0], CAMERAS[3])
    )
    assert app._config.view_mode == "hdmap"
    assert app._config.no_ui
    assert app._config.app.start_frame == 4
    assert isinstance(app._backend, MultiViewRenderBackend)
    assert app._backend._settings.seed == 37
    assert app._backend._settings.compile_denoise == "default"


@pytest.mark.parametrize(
    "args,message",
    [
        (["--views", "0"], "invalid choice"),
        (["--views=2"], "invalid choice"),
        (["--views", "8"], "invalid choice"),
        (["--views", "7", "--camera", CAMERAS[0]], "not allowed"),
        ([f"--camera={CAMERAS[0]}", "--views=1"], "not allowed"),
    ],
)
def test_invalid_camera_presets_fail_before_preparing_assets(
    monkeypatch, capsys, args, message
):
    from cosmos3_nano_sim_auto.apps.interactive_drive import adapter

    monkeypatch.setattr(
        adapter, "prepare_scene", lambda *_: pytest.fail("Unexpected download")
    )
    with pytest.raises(SystemExit) as error:
        create_app().init(args)
    assert error.value.code == 2
    assert message in capsys.readouterr().err
