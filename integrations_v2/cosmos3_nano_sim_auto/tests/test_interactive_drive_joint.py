# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live joint backend timing, sensor identity, and CLI regression checks."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from cosmos3_nano_sim_auto.apps.interactive_drive import backend as module
from cosmos3_nano_sim_auto.apps.interactive_drive.adapter import (
    CosmosInteractiveDriveApplication,
)
from cosmos3_nano_sim_auto.apps.interactive_drive.backend import (
    MultiViewRenderBackend,
    MultiViewSettings,
)
from interactive_drive.config import (
    AppConfig,
    ChunkConfig,
    PresentationLayout,
    RasterConfig,
)

pytestmark = pytest.mark.ci_cpu


@pytest.mark.parametrize("source_kind", ["default", "environment", "cli"])
def test_lidar_cli_downloads_and_selects_matching_hf_codec(
    monkeypatch, tmp_path, source_kind
):
    from cosmos3_nano_sim_auto import checkpoint

    monkeypatch.delenv("COSMOS_INTERACTIVE_CHECKPOINT", raising=False)
    source = "https://huggingface.co/nvidia/Cosmos3-Nano-Sim-Auto/resolve/pinned-qa/model.safetensors.index.json"
    if source_kind == "environment":
        monkeypatch.setenv("COSMOS_INTERACTIVE_CHECKPOINT", source)
    elif source_kind == "cli":
        monkeypatch.setenv(
            "COSMOS_INTERACTIVE_CHECKPOINT", "/unused/environment/export"
        )
    cached = tmp_path / "lidar_vae/diffusion_pytorch_model.safetensors"
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(cached)

    monkeypatch.setattr(checkpoint, "hf_hub_download", download)
    app = CosmosInteractiveDriveApplication(settings=MultiViewSettings())
    app.init(
        [
            "--scene",
            str(tmp_path),
            "--lidar",
            "--no-ui",
            *(["--checkpoint", source] if source_kind == "cli" else []),
        ]
    )
    assert isinstance(app._backend, MultiViewRenderBackend)
    assert app._backend._settings.lidar_checkpoint == cached
    assert app._backend._settings.culled_lidar_renderer
    assert app._backend._settings.checkpoint == (
        module.DEFAULT_CHECKPOINT if source_kind == "default" else source
    )
    assert app._config is not None
    assert app._config.app.presentation.cells == 2
    assert [call["revision"] for call in calls] == [
        "main" if source_kind == "default" else "pinned-qa"
    ] * 2


def test_lidar_cli_help_does_not_download(monkeypatch):
    from cosmos3_nano_sim_auto import checkpoint

    monkeypatch.setattr(
        checkpoint,
        "hf_hub_download",
        lambda **_: pytest.fail("Help downloaded a checkpoint"),
    )
    app = CosmosInteractiveDriveApplication(settings=MultiViewSettings())
    with pytest.raises(SystemExit) as error:
        app.init(["--lidar", "--help"])
    assert error.value.code == 0


@pytest.mark.parametrize(
    "flags,expected",
    [
        ([], False),
        (["--lidar"], True),
        (["--lidar", "--culled-lidar-renderer"], True),
        (["--lidar", "--no-culled-lidar-renderer"], False),
    ],
)
def test_range_culling_default_and_explicit_override(
    monkeypatch, tmp_path, flags, expected
):
    from cosmos3_nano_sim_auto import checkpoint

    monkeypatch.setattr(
        checkpoint,
        "resolve_lidar_checkpoint",
        lambda *args, **kwargs: tmp_path / "codec.safetensors",
    )
    app = CosmosInteractiveDriveApplication(settings=MultiViewSettings())
    app.init(["--scene", str(tmp_path), "--no-ui", *flags])
    assert isinstance(app._backend, MultiViewRenderBackend)
    assert app._backend._settings.culled_lidar_renderer is expected
    assert not app._backend._settings.lidar_codec_tf32
    assert not app._backend._settings.lidar_contiguous_linears


def test_explicit_range_culling_requires_lidar(tmp_path):
    app = CosmosInteractiveDriveApplication(settings=MultiViewSettings())
    with pytest.raises(SystemExit) as error:
        app.init(["--scene", str(tmp_path), "--culled-lidar-renderer", "--no-ui"])
    assert error.value.code == 2


def test_reset_releases_old_rollout_before_allocating_scene_assets(monkeypatch):
    import weakref

    class Cache:
        pass

    backend: Any = MultiViewRenderBackend(ChunkConfig(), RasterConfig())
    old = Cache()
    previous = weakref.ref(old)
    backend._rollout = backend._drawn = backend._decode = backend._lidar = old
    del old
    backend._scene = object()
    backend._network = SimpleNamespace(
        config=SimpleNamespace(
            spec=SimpleNamespace(vfm=SimpleNamespace(latent_patch_size=2))
        )
    )

    def before_allocation(*args, **kwargs):
        assert previous() is None
        raise RuntimeError("old rollout released")

    monkeypatch.setattr(module, "_geometry", before_allocation)
    with pytest.raises(RuntimeError, match="old rollout released"):
        backend._start_rollout()


def test_live_backend_presents_opening_then_current_steered_chunk(monkeypatch):
    class Profiler:
        def __init__(self, **kwargs):
            self.stages = {}

        def record(self, name):
            self.stages[name] = 1.0

        def sync_and_summarize(self):
            return self.stages

    monkeypatch.setattr(module, "EventProfiler", Profiler)
    backend: Any = MultiViewRenderBackend(
        ChunkConfig(), RasterConfig(), settings=MultiViewSettings(device="cpu")
    )
    supplied_rgb, supplied_lidar, pixel_starts = [], [], []
    backend._poses = SimpleNamespace(supply=supplied_rgb.append)
    backend._drawn = SimpleNamespace(
        take_drawn_ms=lambda: 0.0,
        take=lambda count: [
            (np.zeros((2, 2, 3), dtype=np.uint8),) for _ in range(count)
        ],
    )

    def frames(ranges, *, pixel_start, pixel_count, **kwargs):
        pixel_starts.append(pixel_start)
        return torch.full((pixel_count, 3, 2, 2), 0.25)

    backend._lidar = SimpleNamespace(
        last_metrics={"lidar_control_render_s": 0.02, "lidar_codec_encode_s": 0.003},
        supply=supplied_lidar.append,
        start_decode=lambda latent: lambda: latent,
        frames=frames,
        control_frames=lambda *, pixel_count, **kwargs: torch.full(
            (pixel_count, 3, 2, 2), 0.75
        ),
    )

    class Rollout:
        last_lidar = torch.zeros(1)
        last_lidar_range = (0, 1)
        last_metrics = {"lidar_sweeps": 1}
        bounds = iter(((0, 1), (1, 4)))

        def step(self):
            assert len(supplied_rgb) == len(supplied_lidar)
            np.testing.assert_array_equal(supplied_rgb[-1], supplied_lidar[-1])
            self.start, self.end = next(self.bounds)
            return SimpleNamespace(
                start=self.start, end=self.end, memory_tokens=8
            ), None

        def latent_for(self, start, end):
            assert (start, end) == (self.start, self.end)
            return 1 if start == 0 else 12

    backend._rollout = Rollout()
    backend._decode = lambda count: torch.zeros(1, count, 3, 2, 2)
    for index, count in enumerate((1, 12)):
        path = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
        path[:, 1, 3] = np.arange(count)
        trajectory = SimpleNamespace(
            rig_poses_world=path,
            timestamps_us=np.arange(count),
            boundary_state_after_chunk=object(),
        )
        chunk = backend._render(trajectory, opening=index == 0)
        assert len(chunk.frames) == count
        assert all(frame.views == 2 for frame in chunk.frames)
        for frame in chunk.frames:
            assert np.asarray(frame.view_model_rgb_host_uint8[1])[0, 0, 0] == 64
            assert np.asarray(frame.view_rgb_host_uint8[1])[0, 0, 0] == 191
        assert backend.finalize()["lidar_enabled"] == 1
        assert backend.finalize()["lidar_control_render_s"] == 0.02
        assert backend.finalize()["lidar_codec_encode_s"] == 0.003
    assert pixel_starts == [0, 1]
    assert backend._frames_emitted == 13
    backend.close()
    assert backend._lidar is None


@pytest.mark.parametrize("contiguous_linears", [False, True])
def test_live_cli_reserves_sensor_cell_and_passes_model_options(
    tmp_path, contiguous_linears
):
    app = CosmosInteractiveDriveApplication(settings=MultiViewSettings())
    codec = tmp_path / "codec.pt"
    app.init(
        [
            "--scene",
            str(tmp_path),
            "--lidar-checkpoint",
            str(codec),
            "--checkpoint",
            "joint",
            "--no-ui",
            "--world-model-device",
            "cpu",
            "--world-model-seed",
            "7",
            "--lidar-view",
            "range",
            "--optimized-lidar-renderer",
            *(["--lidar-contiguous-linears"] if contiguous_linears else []),
        ]
    )
    assert app._config is not None
    assert isinstance(app._backend, MultiViewRenderBackend)
    assert app._config.app.presentation.cells == 2
    assert (app.session_desc().video_width, app.session_desc().video_height) == (
        1664,
        480,
    )
    assert app._backend._settings.lidar_checkpoint == codec
    assert app._backend._settings.checkpoint == "joint"
    assert app._backend._settings.seed == 7
    assert app._backend._settings.device == "cpu"
    assert app._backend._settings.lidar_view == "range"
    assert app._backend._settings.optimized_lidar_renderer
    assert app._backend._settings.lidar_contiguous_linears == contiguous_linears
    assert app._backend._settings.culled_lidar_renderer
    # Lower-precision codec execution remains opt-in.
    assert not app._backend._settings.lidar_codec_tf32


@pytest.mark.parametrize(
    "flag",
    [
        "--optimized-lidar-renderer",
        "--culled-lidar-renderer",
        "--lidar-codec-tf32",
        "--lidar-contiguous-linears",
    ],
)
def test_lidar_optimizations_cannot_be_silently_enabled_without_lidar(flag):
    app = CosmosInteractiveDriveApplication(settings=MultiViewSettings())
    with pytest.raises(SystemExit):
        app.init([flag])


@pytest.mark.parametrize(
    "cameras, flags, layout, size",
    [
        ("front", [], (2, 1, None), (3328, 960)),
        ("front", ["--no-ui"], (2, 1, None), (1664, 480)),
        ("front", ["--no-u"], (2, 1, None), (1664, 480)),
        ("front,left", [], (2, 2, None), (1664, 960)),
        ("front,left", ["--no-ui"], (2, 2, None), (1664, 960)),
        ("front,left", ["--present-camera", "left"], (1, 1, 1), (1664, 960)),
        ("front,left", ["--no-ui", "--present-camera", "left"], (1, 1, 1), (832, 480)),
    ],
)
def test_sensor_layout_sizes_hud_and_preserves_camera_selection(
    tmp_path, cameras, flags, layout, size
):
    app = CosmosInteractiveDriveApplication(settings=MultiViewSettings())
    app.init(
        [
            "--scene",
            str(tmp_path),
            "--camera",
            cameras,
            "--lidar-checkpoint",
            str(tmp_path / "codec.pt"),
            *flags,
        ]
    )
    assert app._config is not None
    config = app._config.app
    assert config.camera_names == tuple(cameras.split(","))
    assert config.presentation == PresentationLayout(*layout)
    restarted = replace(config, scene_path=tmp_path / "another-scene")
    assert type(restarted) is type(config)
    assert restarted.presentation == config.presentation
    assert isinstance(restarted.raster, RasterConfig)
    assert (app.session_desc().video_width, app.session_desc().video_height) == size


@pytest.mark.parametrize("view_mode, values", [("rgb", [100, 101]), ("hdmap", [0, 1])])
def test_sensor_layout_tiles_camera_and_lidar_with_shared_presenter(
    tmp_path, view_mode, values
):
    from typing import cast

    from interactive_drive.core import _frame_chunk_tensor
    from interactive_drive.types import PresentedFrame

    app = CosmosInteractiveDriveApplication(
        settings=MultiViewSettings(lidar_checkpoint=Path("codec.pt"))
    )
    app.init(["--scene", str(tmp_path), "--no-ui"])
    assert app._config is not None
    frame = PresentedFrame(
        timestamp_us=0,
        view_rgb_host_uint8=tuple(
            np.full((1, 1, 3), value, dtype=np.uint8) for value in (0, 1)
        ),
        depth_host_f32=None,
        view_model_rgb_host_uint8=tuple(
            np.full((1, 1, 3), value, dtype=np.uint8) for value in (100, 101)
        ),
    )
    result = _frame_chunk_tensor(
        cast(Any, SimpleNamespace(frames=[frame])),
        view_mode,
        layout=app._config.app.presentation,
    )
    assert result.shape == (1, 3, 1, 2)
    assert result[0, 0, 0].tolist() == values


@pytest.mark.parametrize("flags, size", [([], (3328, 960)), (["--no-ui"], (1664, 480))])
def test_worker_ui_suppression_keeps_the_driver_session_size(tmp_path, flags, size):
    from flashdreams.core.distributed.parallel import ParallelContext

    for rank in (0, 1):
        mesh = replace(ParallelContext.single("cpu"), cp_rank=rank, cp_size=2)
        app = CosmosInteractiveDriveApplication(
            settings=MultiViewSettings(parallel=mesh, lidar_checkpoint=Path("codec.pt"))
        )
        app.init(["--scene", str(tmp_path), *flags])
        assert app._config is not None
        assert app._config.no_ui == bool(rank or flags)
        assert (app.session_desc().video_width, app.session_desc().video_height) == size


def test_rgb_only_application_retains_the_shared_config_and_layout(tmp_path):
    app = CosmosInteractiveDriveApplication(settings=MultiViewSettings())
    app.init(["--scene", str(tmp_path), "--no-ui"])
    assert app._config is not None
    assert type(app._config.app) is AppConfig
    assert app._config.app.presentation == PresentationLayout(1, 1, None)
    assert (app.session_desc().video_width, app.session_desc().video_height) == (
        832,
        480,
    )


def test_live_lidar_refuses_incompatible_clock_before_loading_weights():
    with pytest.raises(ValueError, match="30 FPS"):
        MultiViewRenderBackend(
            ChunkConfig(fps=24),
            RasterConfig(),
            settings=MultiViewSettings(lidar_checkpoint=Path("codec.pt")),
        )


def test_cli_preserves_programmatic_codec_and_view_defaults(tmp_path):
    from flashdreams.core.visualization.lidar import LidarView

    app = CosmosInteractiveDriveApplication(
        settings=MultiViewSettings(
            lidar_tokenizer_version="v1p1", lidar_view=LidarView.RANGE
        )
    )
    app.init(["--scene", str(tmp_path), "--no-ui"])
    assert isinstance(app._backend, MultiViewRenderBackend)
    assert app._backend._settings.lidar_tokenizer_version == "v1p1"
    assert app._backend._settings.lidar_view is LidarView.RANGE
