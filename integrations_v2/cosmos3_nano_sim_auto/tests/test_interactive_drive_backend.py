# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What our rollout reports to Interactive Drive, and how a chunk is assembled.

The chunk sizes are the contract between the two: the app samples a trajectory
as long as this backend says its chunk is, and presents a frame for every frame
of it. Control is drawn ahead of all that, so the spans do not line up and the
arithmetic is worth asserting rather than reasoning about.
"""

from __future__ import annotations

import ast
import inspect
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch
from cosmos3_nano_sim_auto import inputs as inputs_module
from cosmos3_nano_sim_auto.apps.interactive_drive import backend as backend_module
from cosmos3_nano_sim_auto.apps.interactive_drive.backend import (
    MultiViewRenderBackend,
    MultiViewSettings,
    _DrawnRig,
    _opening_span,
    _per_frame,
    _presented,
    _span_frames,
)
from cosmos3_nano_sim_auto.impl.conditioning.camera_rig import (
    CAMERAS as RIG_CAMERAS,
)
from cosmos3_nano_sim_auto.impl.conditioning.captions import (
    AV_JOINT_CAMERA_LIDAR_TRANSFER_SYSTEM_PROMPT,
    AV_MULTIVIEW_TRANSFER_SYSTEM_PROMPT,
    caption_json,
)
from cosmos3_nano_sim_auto.impl.packing import ClipGeometry
from cosmos3_nano_sim_auto.impl.transformer.mask import MASKLESS_SCOPES
from interactive_drive.config import ChunkConfig, RasterConfig
from loguru import logger

from flashdreams.core.distributed.parallel import ParallelContext

pytestmark = pytest.mark.ci_cpu


def _imports_of(path: Path) -> set[str]:
    """Every module a file imports, each with its dotted prefixes.

    The prefixes are what make a match exact: ``cosmos3_nano_sim_auto.apps`` must not
    read as the ``cosmos3_nano_sim_auto.impl.app`` it is spelled a letter away from.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(node.module)
    return {
        ".".join(name.split(".")[: depth + 1])
        for name in found
        for depth in range(name.count(".") + 1)
    }


def _geometry(views: int = 1, frames_per_chunk: int = 4) -> ClipGeometry:
    return ClipGeometry(
        num_views=views,
        frames_per_view=None,
        patch_h=2,
        patch_w=2,
        frames_per_chunk=frames_per_chunk,
        condition_frames=1,
        seconds_per_frame=4.0 / 30.0,
    )


class _Ludus:
    """A map renderer that draws numbered frames, a view's together."""

    def __init__(self, views: int, *, device: torch.device | None = None) -> None:
        self.cameras = tuple(range(views))
        self.asked: list[tuple[int, int]] = []
        self._device = device

    def render(self, first: int, last: int, views: Any = None) -> torch.Tensor:
        self.asked.append((first, last))
        drawn = list(range(len(self.cameras)) if views is None else views)
        span = last - first
        frames = torch.zeros((len(drawn) * span, 2, 2, 3), device=self._device)
        for index, view in enumerate(drawn):
            for offset in range(span):
                frames[index * span + offset] = view * 1000 + first + offset
        return frames


def test_a_chunk_is_as_long_as_the_latent_frames_it_covers() -> None:
    assert _span_frames(4) == 16
    assert _span_frames(1) == 4


def test_the_first_chunk_is_the_opening_frame_and_one_chunk() -> None:
    """Not the control window, which is several chunks longer.

    The window is rendered while the rollout is built, before the app has a
    trajectory to sample, so reporting it would have the app supply a path for
    frames it never gets a picture back for.
    """
    assert _opening_span(_geometry()) == 17
    assert _opening_span(_geometry(frames_per_chunk=8)) == 33


def _scene(views: int = 1) -> Any:
    """Only what ``_geometry`` reads: one frame to size the grid, and a rig."""
    frame = np.zeros((480, 832, 3), dtype=np.uint8)
    return SimpleNamespace(
        initial_rgbs=tuple(frame for _ in range(views)),
        cameras=tuple(
            SimpleNamespace(clipgt_name=name) for name in RIG_CAMERAS[:views]
        ),
    )


@pytest.mark.parametrize("fps", [30.0, 24.0])
def test_the_token_grid_is_spaced_at_the_rate_the_app_is_running(fps: float) -> None:
    """The poses and the rollout are both given the app's rate, so this must be.

    Stating 30 here while the app ran at another rate does not merely mislabel
    a frame: the decomposed scope compares these timestamps against its window,
    so a chunk would attend over the wrong stretch of time.
    """
    geometry = backend_module._geometry(
        _scene(), MultiViewSettings(), latent_patch_size=2, fps=fps
    )

    assert geometry.seconds_per_frame == pytest.approx(4.0 / fps)


def test_no_attention_setting_is_left_at_the_rollouts_own_default() -> None:
    """The rollout's defaults are the older stage 3 checkpoint's.

    So an argument left unset is wrong for these weights rather than merely
    conservative, and it costs 2.8x the frame rate without saying anything.
    """
    from cosmos3_nano_sim_auto.impl.rollout import ChunkRollout

    older = inspect.signature(ChunkRollout.__init__).parameters
    settings = MultiViewSettings()

    assert settings.attention_scope != older["attention_scope"].default
    assert settings.maskless_attention != older["maskless_attention"].default
    assert settings.fused_attention_merge != older["fused_attention_merge"].default
    assert (
        settings.temporal_window_seconds
        != older["decomposed_temporal_window_seconds"].default
    )
    assert settings.attention_scope in MASKLESS_SCOPES


def _mesh_of_two(device: str) -> ParallelContext:
    """A context axis of two, which is the smallest mesh worth the name.

    The groups are left unset because nothing here collects anything: what is
    under test is which card a rank settles on, decided before any process
    group is touched.
    """
    return replace(ParallelContext.single(device), cp_rank=0, cp_size=2)


def _device_of(settings: MultiViewSettings) -> torch.device:
    """The card a backend built from these settings would run on.

    The default chunk and raster configs will do: resolving a device happens
    before anything is loaded, and neither of them has a say in it.
    """
    backend = MultiViewRenderBackend(ChunkConfig(), RasterConfig(), settings=settings)
    return backend._device


def test_a_drive_on_one_card_declares_no_mesh_at_all() -> None:
    """``None`` rather than a mesh of one, which are not the same thing.

    Building a default context would call into the distributed machinery and
    set a CUDA device to answer a question nobody asked, on every run that
    constructs settings, tests included.
    """
    assert MultiViewSettings().parallel is None


def test_a_launched_rank_sits_on_the_card_the_mesh_names() -> None:
    """Under a launcher the setting was decided once for every rank.

    Every rank of a mesh must sit on its own GPU, and only the mesh knows which
    that is. Two ranks sharing a card is a run that works and has half the
    memory it thinks, so it is worth making impossible rather than documenting.
    """
    settings = MultiViewSettings(device="cuda:0", parallel=_mesh_of_two("cuda:3"))

    assert _device_of(settings) == torch.device("cuda:3")


def test_on_one_rank_the_setting_still_picks_the_card() -> None:
    """A mesh of one names a card by accident rather than by decision.

    It is whatever was current when no launcher set `LOCAL_RANK`, so letting it
    win would make one card out of eight unaskable for. This is the rule the
    batch command follows: the flag loses to a real mesh and to nothing else.
    """
    settings = MultiViewSettings(
        device="cuda:3", parallel=ParallelContext.single("cuda:0")
    )

    assert _device_of(settings) == torch.device("cuda:3")


def test_the_app_is_told_which_rank_owns_the_window_and_nothing_else() -> None:
    """The mesh reaches the app, because only it can know about the window.

    Everything else about a mesh is the checkpoint's: a context axis over
    tokens and a tensor axis over heads, neither of which the app has any
    business naming. What it cannot work out for itself is that four ranks are
    one drive, so that is the single fact that crosses over.
    """
    from cosmos3_nano_sim_auto.apps.interactive_drive.adapter import defaults

    mesh = ParallelContext.single("cuda:2")

    assert defaults(MultiViewSettings(parallel=mesh)).parallel_context is mesh
    assert defaults().parallel_context is None


def test_nothing_in_this_binding_reaches_for_the_batch_command() -> None:
    """The batch command owns a parser, and this binding must not read it.

    Every setting the two share lives in :mod:`cosmos3_nano_sim_auto.config`, and an
    import of the other front end is easy to add back without noticing. It
    would make the two undeletable as a pair and let a parser's tri-state
    defaults leak into a path that has no parser. Read off the source rather
    than the loaded module, since an import that only runs under
    ``TYPE_CHECKING`` counts just as much.
    """
    package = Path(backend_module.__file__).parent
    leaking = {
        path.name: sorted(imports)
        for path in sorted(package.glob("*.py"))
        if (imports := _imports_of(path) & {"cosmos3_nano_sim_auto.impl.app"})
    }

    assert not leaking


def test_clearing_for_a_new_scene_does_not_bind_the_one_being_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the app calls before binding a restart's scene, and what it cost.

    The base class clears by resetting, and resetting here means binding, so a
    restart bound the outgoing scene in full before binding the incoming one
    and charged the driver for both.
    """
    backend = MultiViewRenderBackend(ChunkConfig(), RasterConfig())
    bound: list[str] = []
    monkeypatch.setattr(backend, "_start_rollout", lambda: bound.append("bound"))
    bound_scene: Any = object()
    backend._scene = bound_scene
    backend._rollout = object()

    backend.reset_scene_conditioning()

    assert bound == []
    assert backend._rollout is None

    # Resetting on its own still starts the drive again, which is the gamepad's
    # restart and the one case where rebinding is the whole point.
    backend.reset()

    assert bound == ["bound"]


def test_drawn_frames_are_taken_in_order_across_the_spans_they_came_in() -> None:
    """The prefill draws the window in slots, and a chunk shows part of it."""
    rig = _DrawnRig(_Ludus(1))
    rig.render(0, 1)
    rig.render(1, 17)

    first = rig.take(17)

    assert [float(one[0][0, 0, 0]) for one in first] == list(range(17))


def test_a_view_is_kept_with_its_own_frame() -> None:
    rig = _DrawnRig(_Ludus(2))
    rig.render(0, 2)

    frames = rig.take(2)

    assert [float(view[0, 0, 0]) for view in frames[0]] == [0.0, 1000.0]
    assert [float(view[0, 0, 0]) for view in frames[1]] == [1.0, 1001.0]


def test_taking_more_than_was_drawn_is_refused() -> None:
    rig = _DrawnRig(_Ludus(1))
    rig.render(0, 2)

    with pytest.raises(RuntimeError, match="read no control for"):
        rig.take(3)


def test_drawing_reports_nothing_rather_than_failing_without_a_gpu() -> None:
    """The queue is worth unit-testing on the CPU; only its timing needs a device."""
    rig = _DrawnRig(_Ludus(1))
    rig.render(0, 2)

    assert rig.take_drawn_ms() == 0.0
    assert len(rig.take(2)) == 2


def test_a_chunks_drawing_is_timed_on_the_gpu_and_not_billed_to_the_next() -> None:
    """The host clock knows only how long it took to ask.

    Drawing is queued like everything else, so a host-timed span reports the
    launch and leaves the work to whichever later block happens to wait for it.
    Events measure it where it happens, and clearing them as a chunk reads them
    is what stops the next chunk being charged for this one's map.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA events need a GPU.")
    device = torch.device("cuda")
    rig = _DrawnRig(_Ludus(2, device=device), device=device)

    rig.render(0, 4)
    torch.cuda.synchronize(device)

    assert rig.take_drawn_ms() > 0.0
    assert rig.take_drawn_ms() == 0.0


class _OtherRanks:
    """The ranks a one-process test does not have, standing in for ``dist``.

    The gather receives the cameras rank 0 does not own in rig order, so a
    call's position says which camera it is for; each is filled in as its owner
    would have drawn it, which is what makes a misordered gather visible, and
    asked of the rank that owns it, which is what makes a misrouted one fail.
    """

    def __init__(
        self, *, views: int, world: int, owned: Sequence[int], first: int, span: int
    ) -> None:
        self._missing = iter(view for view in range(views) if view not in owned)
        self._world = world
        self._first = first
        self._span = span

    def get_backend(self) -> str:
        return "gloo"

    def recv(self, tensor: torch.Tensor, src: int) -> None:
        view = next(self._missing)
        assert src == view % self._world, f"camera {view} asked of rank {src}"
        for offset in range(self._span):
            tensor[offset] = view * 1000 + self._first + offset


def _other_ranks(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> None:
    from cosmos3_nano_sim_auto.impl.distributed import parallel as parallel_module

    fake = _OtherRanks(**kwargs)
    monkeypatch.setattr(backend_module, "dist", fake)
    monkeypatch.setattr(parallel_module, "dist", fake)


def _sharded(views: int, world: int, owned: Sequence[int]) -> _DrawnRig:
    """A rig on a mesh of ``world``, holding ``owned`` of a ``views`` camera rig."""
    mesh = replace(ParallelContext.single("cpu"), cp_rank=owned[0], cp_size=world)
    return _DrawnRig(_Ludus(views), parallel=mesh)


def test_a_sharded_draw_still_queues_every_camera_in_rig_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sharding the draw is for the encoder; presenting needs the whole rig.

    A presented frame carries one picture per camera and ``_presented`` refuses
    anything else, so the cameras a rank did not draw have to arrive from their
    owners and land back in the rig's order. Out of order is worse than missing:
    it would quietly show the rear camera's map beneath the front camera.
    """
    rig = _sharded(views=4, world=2, owned=[0, 2])
    _other_ranks(monkeypatch, views=4, world=2, owned=[0, 2], first=0, span=2)

    rig.render(0, 2, views=[0, 2])
    frames = rig.take(2)

    assert [float(v[0, 0, 0]) for v in frames[0]] == [0.0, 1000.0, 2000.0, 3000.0]
    assert [float(v[0, 0, 0]) for v in frames[1]] == [1.0, 1001.0, 2001.0, 3001.0]


def test_a_sharded_draw_hands_the_encoder_this_ranks_cameras_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The provider sizes its clips by what this rank owns and checks the count.

    So the gather must not reach the value handed back, only the queue.
    """
    rig = _sharded(views=4, world=2, owned=[0, 2])
    _other_ranks(monkeypatch, views=4, world=2, owned=[0, 2], first=0, span=2)

    frames = rig.render(0, 2, views=[0, 2])

    assert [float(one[0, 0, 0]) for one in frames] == [0.0, 1.0, 2000.0, 2001.0]


def test_a_drive_on_one_card_gathers_nothing() -> None:
    """No mesh means no collective, which is what lets the CPU tests above run."""
    rig = _DrawnRig(_Ludus(2))

    assert rig.render(0, 2).shape[0] == 4
    assert len(rig.take(2)) == 2


def test_a_cameras_owner_is_the_same_rank_in_the_gather_and_the_encode() -> None:
    """Two spellings of one rule, and a drift between them would swap cameras.

    The gather receives camera ``v`` from rank ``v % world``; the provider
    hands a rank ``range(rank, views, world)``. Neither reads the other, so the
    agreement is worth asserting rather than noticing later in a picture.
    """
    from cosmos3_nano_sim_auto.impl.distributed.parallel import owned_views

    for world in (2, 3, 4):
        for views in (1, 4, 11):
            for rank in range(world):
                mesh = replace(
                    ParallelContext.single("cpu"), cp_rank=rank, cp_size=world
                )
                assert all(view % world == rank for view in owned_views(mesh, views))


def test_a_span_splits_into_one_entry_a_frame() -> None:
    split = _per_frame(_Ludus(2).render(0, 3), views=2)

    assert len(split) == 3
    assert all(len(one) == 2 for one in split)


def _model(views: int, frames: int) -> torch.Tensor:
    return torch.zeros((views, frames, 3, 2, 2))


def _drawn(views: int, frames: int) -> list[tuple[Any, ...]]:
    one = np.zeros((2, 2, 3), dtype=np.uint8)
    return [tuple(one for _ in range(views)) for _ in range(frames)]


def test_a_presented_frame_carries_every_camera_of_the_rig() -> None:
    frames = _presented(
        model=_model(2, 3), drawn=_drawn(2, 3), timestamps_us=[0, 100, 200]
    )

    assert [frame.timestamp_us for frame in frames] == [0, 100, 200]
    assert all(len(frame.view_model_rgb_host_uint8) == 2 for frame in frames)
    assert all(len(frame.view_rgb_host_uint8) == 2 for frame in frames)


def test_generating_a_different_number_of_frames_than_the_app_asked_is_refused() -> (
    None
):
    with pytest.raises(ValueError, match="chunk size and this backend"):
        _presented(model=_model(1, 3), drawn=_drawn(1, 3), timestamps_us=[0, 100])


def test_presenting_frames_with_no_map_drawn_for_them_is_refused() -> None:
    with pytest.raises(ValueError, match="come apart"):
        _presented(model=_model(1, 3), drawn=_drawn(1, 2), timestamps_us=[0, 100, 200])


def test_a_drawn_frame_missing_a_camera_is_refused() -> None:
    with pytest.raises(ValueError, match="disagree about the rig"):
        _presented(model=_model(2, 2), drawn=_drawn(1, 2), timestamps_us=[0, 100])


class _Reporter(MultiViewRenderBackend):
    """Just the reporting, with the timings a chunk would have left behind."""

    def __init__(self, total_ms: float) -> None:
        self._chunks = 0
        self._timings = {
            "draw_ms": 10.0,
            "model_ms": total_ms - 20.0,
            "decode_ms": 10.0,
            "total_ms": total_ms,
        }


def _reported(backend: _Reporter, chunks: int, *, opening: bool = False) -> list[str]:
    lines: list[str] = []
    handle = logger.add(lambda message: lines.append(message), level="INFO")
    try:
        for _ in range(chunks):
            backend._report(
                opening=opening, frames=16, trace=SimpleNamespace(start=1, end=5)
            )
    finally:
        logger.remove(handle)
    return lines


def test_a_chunk_reports_its_frame_rate_as_well_as_its_cost() -> None:
    line = _reported(_Reporter(500.0), 1)[0]

    assert "total_ms=500.0" in line
    # 16 frames in half a second.
    assert "fps=32.00" in line
    assert "draw_ms=10.0" in line


def test_joint_report_separates_controls_codec_model_and_viewer() -> None:
    reporter = _Reporter(500.0)
    reporter._settings = MultiViewSettings(
        optimized_lidar_renderer=True, culled_lidar_renderer=False
    )
    reporter._timings.update(
        lidar_enabled=1,
        lidar_control_render_s=0.12,
        lidar_codec_encode_s=0.003,
        joint_denoise_s=0.3,
        lidar_render_s=0.004,
    )
    line = _reported(reporter, 1)[0]
    assert "lidar_renderer=optimized" in line
    assert "lidar_contiguous_linears=False" in line
    assert "lidar_render_ms=120.0" in line and "lidar_encode_ms=3.0" in line
    assert "joint_denoise_ms=300.0" in line and "lidar_present_ms=4.0" in line
    assert "total_ms=500.0" in line  # The breakdown must not be counted twice.


def test_a_steady_run_reports_every_chunk_at_first_and_then_one_in_ten() -> None:
    """A steady run says the same thing over and over, as OmniDreams decided."""
    lines = _reported(_Reporter(100.0), 25)

    assert len(lines) == 5  # 1, 2, 3, 10, 20


def test_a_chunk_slow_enough_to_drop_frames_is_always_reported() -> None:
    lines = _reported(_Reporter(600.0), 25)

    assert len(lines) == 25


def _tokenizer_recording(
    monkeypatch: pytest.MonkeyPatch, *, lidar: bool = False
) -> list[str]:
    """Record what is tokenized. The real tokenizer is a Qwen checkpoint on disk.

    What is worth pinning without it is which words reach the model and how
    many runs they are split into, which is the whole of what this fixes.
    """
    seen: list[str] = []

    def tokenize(text: Any, **kwargs: Any) -> torch.Tensor:
        assert kwargs["system_prompt"] == (
            AV_JOINT_CAMERA_LIDAR_TRANSFER_SYSTEM_PROMPT
            if lidar
            else AV_MULTIVIEW_TRANSFER_SYSTEM_PROMPT
        )
        seen.append(text if isinstance(text, str) else caption_json(text))
        return torch.arange(len(seen) + 2)

    monkeypatch.setattr(inputs_module, "tokenize_prompt", tokenize)
    monkeypatch.setattr(
        inputs_module,
        "caption_token_ids",
        lambda payload, **kwargs: tokenize(payload, **kwargs),
    )
    return seen


def _drive_scene(tmp_path: Path, rig: Sequence[str], *, prompt: str = "") -> Any:
    """A sample on disk and the bundle naming it, as the scene loader leaves them."""
    (tmp_path / "sample.json").write_text(
        json.dumps(
            {
                "name": "sample_3",
                "model_mode": "image2video",
                "resolution": "480",
                "frame_range": [0, 200],
                "multiview": {
                    "views": [
                        {
                            "camera_key": camera,
                            "control_path": f"{camera}.mp4",
                            "image_path": f"{camera}.rgb.mp4",
                            "caption": {"summary": f"{camera} sees a road"},
                        }
                        for camera in RIG_CAMERAS[:4]
                    ]
                },
            }
        )
    )
    return SimpleNamespace(
        metadata={"sample": str(tmp_path)},
        prompt=prompt,
        cameras=tuple(SimpleNamespace(clipgt_name=name) for name in rig),
    )


@pytest.mark.parametrize("lidar", [False, True])
def test_a_drive_conditions_on_the_samples_own_caption_for_each_camera(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lidar: bool
) -> None:
    """A recorded-scene record keeps its words in the views and carries no single prompt.

    Reading the bundle's one sentence instead left every camera with nothing
    but the prefix saying which way it points, which is a silent loss: the run
    still generates, against conditioning the batch command would not have used.
    """
    seen = _tokenizer_recording(monkeypatch, lidar=lidar)
    scene = _drive_scene(tmp_path, RIG_CAMERAS[:4])

    text_ids, view_text_tokens = backend_module._prompt(scene, fps=30.0, lidar=lidar)

    assert len(seen) == 4
    assert all("sees a road" in caption for caption in seen)
    assert view_text_tokens == (3, 4, 5, 6)
    assert sum(view_text_tokens) == text_ids.shape[0]


def test_a_drive_takes_its_captions_in_the_order_its_rig_names_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Order is what the model conditions on, so the rig decides it, not the file."""
    seen = _tokenizer_recording(monkeypatch)
    rig = (RIG_CAMERAS[2], RIG_CAMERAS[0])
    scene = _drive_scene(tmp_path, rig)

    backend_module._prompt(scene, fps=30.0)

    assert len(seen) == 2
    assert RIG_CAMERAS[2] in seen[0]
    assert RIG_CAMERAS[0] in seen[1]


@pytest.mark.parametrize("lidar", [False, True])
def test_a_prompt_the_driver_supplied_replaces_the_samples_captions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lidar: bool
) -> None:
    """``--prompt``, or one typed into a restart, describes the rig and not a camera."""
    seen = _tokenizer_recording(monkeypatch, lidar=lidar)
    scene = _drive_scene(tmp_path, RIG_CAMERAS[:2], prompt="Drive through snow.")

    _, view_text_tokens = backend_module._prompt(scene, fps=30.0, lidar=lidar)

    assert view_text_tokens == (3, 4)
    assert all(text.endswith("Drive through snow.") for text in seen)
    assert not any("sees a road" in text for text in seen)


def test_the_batch_commands_pipelining_flags_reach_the_drives_settings(
    tmp_path: Path,
) -> None:
    from cosmos3_nano_sim_auto.apps.interactive_drive.adapter import (
        CosmosInteractiveDriveApplication,
    )

    app = CosmosInteractiveDriveApplication(settings=MultiViewSettings())
    app.init(
        ["--scene", str(tmp_path), "--no-ui", "--parallel-decode", "--decoder-compile"]
        + ["--compile-denoise", "max-autotune", "--compile-control-encode"]
        + ["--prefetch-controls", "--prefetch-controls-gpu-offset", "2"]
    )
    assert isinstance(app._backend, MultiViewRenderBackend)
    settings = app._backend._settings

    assert settings.parallel_decode and settings.decoder_compile
    assert settings.compile_denoise == "max-autotune"
    assert settings.compile_control_encode and settings.prefetch_controls
    assert settings.prefetch_controls_gpu_offset == 2


@pytest.mark.parametrize(
    "settings",
    [
        MultiViewSettings(
            compile_denoise="default",
            parallel=replace(ParallelContext.single("cpu"), tp_size=2),
        ),
        MultiViewSettings(
            compile_denoise="default",
            lidar_checkpoint=Path("c.pt"),
            parallel=replace(ParallelContext.single("cpu"), tp_size=2),
        ),
    ],
)
def test_compiled_denoising_is_refused_before_any_weights_load(
    settings: MultiViewSettings,
) -> None:
    """A tensor axis all-reduces in place inside the compiled region."""
    with pytest.raises(ValueError, match="--compile-denoise"):
        MultiViewRenderBackend(ChunkConfig(), RasterConfig(), settings=settings)


class _DrivenPipe:
    """A driven worker's end of the pipe: draws one camera, stamping each frame with its pose.

    Channel 0 is the frame's forward position and channel 1 its sideways one,
    in metres, so the queue the app presents from reads back the path drawn.
    """

    def __init__(self) -> None:
        self.sent: list[tuple[int, int]] = []

    def send(self, message: tuple[Any, ...]) -> None:
        start, end, self._poses = message
        self.sent.append((start, end))

    def recv(self) -> tuple[Any, ...]:
        from cosmos3_nano_sim_auto.impl.conditioning.control_provider import (
            pixel_span_for_latents,
        )

        start, end = self.sent[-1]
        first, last = pixel_span_for_latents(start, end)
        drawn = torch.zeros((last - first, 2, 2, 3), dtype=torch.uint8)
        drawn[..., 0] = torch.from_numpy(self._poses[:, 0, 3]).round()[:, None, None]
        drawn[..., 1] = torch.from_numpy(self._poses[:, 1, 3]).round()[:, None, None]
        return [torch.zeros(4, end - start, 2, 2)], last - first, drawn


def _host(frames: range) -> np.ndarray:
    """The host's path at a metre a frame, its frame 0 one frame past the opening.

    Straight for the first 13 frames, then drifting a metre left a frame, so a
    drawn frame's sideways position says which host frame it was drawn from.
    """
    path = np.repeat(np.eye(4, dtype=np.float32)[None], len(frames), axis=0)
    path[:, 0, 3] = [frame + 1 for frame in frames]
    path[:, 1, 3] = [max(frame - 12, 0) for frame in frames]
    return path


def _driven(*, delay: int) -> tuple[Any, Any, _DrawnRig, _DrivenPipe]:
    from cosmos3_nano_sim_auto.apps.interactive_drive.poses import SuppliedPoses
    from cosmos3_nano_sim_auto.impl.conditioning.control_provider import (
        RenderedControls,
    )
    from cosmos3_nano_sim_auto.impl.conditioning.control_worker import (
        ControlWorker,
    )

    poses = SuppliedPoses(
        np.eye(4, dtype=np.float32),
        opening_rig_to_world=np.eye(4, dtype=np.float32),
        speed_mps=30.0,
        seconds_per_frame=1.0 / 30.0,
        delay_frames=delay,
    )
    drawn = _DrawnRig(None, cameras=1)
    pipe = _DrivenPipe()
    worker = backend_module._DrivenWorker(
        ControlWorker(cast(Any, pipe), device=torch.device("cpu")),
        poses=poses,
        drawn=drawn,
        owned=[0],
    )
    controls = RenderedControls(
        None,
        [lambda clip: clip],
        geometry=_geometry(1, frames_per_chunk=3),
        latent_patch_size=2,
        worker=worker,
    )
    # Without the anchor's axis change, so a drawn pose reads as the rig's.
    poses._anchor = np.eye(4, dtype=np.float32)
    return poses, controls, drawn, pipe


def test_prefetch_draws_the_next_chunk_a_chunk_late_along_the_hosts_own_path() -> None:
    """The RGB drive's prefetch, in the order the rollout asks for control.

    Drawing ahead of the host extrapolates, and the host's poses for that ground
    are then dropped as already drawn: the car would never steer again. So the
    map runs one chunk behind the host, and every frame of it is the host's own
    path, moved on by the opening's carried-on frames so nothing jumps.
    """
    from cosmos3_nano_sim_auto.impl.packing import ar_chunk_range, control_window

    poses, controls, drawn, pipe = _driven(delay=12)
    geometry = _geometry(1, frames_per_chunk=3)
    window = control_window(geometry, 1)
    for start, end in window:  # the prefill
        controls.tokens(start, end)
    ready = window[-1][1]
    supplied, generated = 0, 1
    in_flight = []
    for _ in range(5):  # what ChunkRollout.step does, a chunk at a time
        current = ar_chunk_range(geometry, generated)
        assert current is not None
        start, end = current
        pixels = 13 if start == 1 else 12
        poses.supply(_host(range(supplied, supplied + pixels)))
        supplied += pixels
        while ready < end:
            following = ar_chunk_range(geometry, ready)
            assert following is not None
            controls.tokens(*following)
            ready = following[1]
        following = ar_chunk_range(geometry, ready)
        assert following is not None
        controls.prefetch(*following)
        in_flight.append(pipe.sent[-1])
        generated = end

    # Nothing is drawn ahead of the host: the first chunk's request waits a step.
    assert in_flight == [(4, 7), (7, 10), (10, 13), (13, 16), (16, 19)]
    frames = drawn.take(4 * 15 + 1)  # latents [0, 16); [16, 19) is still in flight
    forward = [int(frame[0][0, 0, 0]) for frame in frames]
    sideways = [int(frame[0][0, 0, 1]) for frame in frames]
    # One path, every frame once, in order and without a jump...
    assert forward == list(range(1, 62))
    # ...on which the host's steering arrives exactly one chunk late.
    assert sideways == [max(frame - 24, 0) for frame in range(61)]


def test_with_no_delay_a_request_waits_for_the_hosts_poses_and_then_uses_them() -> None:
    """The joint drive's case: the RGB map must be drawn along the LiDAR's path."""
    poses, controls, drawn, pipe = _driven(delay=0)

    controls.prefetch(0, 1)
    assert pipe.sent == []
    poses.supply(_host(range(1)))
    controls.prefetch(0, 1)
    controls.tokens(0, 1)

    assert pipe.sent == [(0, 1)]
    assert int(drawn.take(1)[0][0][0, 0, 0]) == 1


def test_a_worker_rank_decodes_its_own_cameras_and_presents_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The app reads no frame a worker returns, so a worker builds none."""
    monkeypatch.setattr(
        backend_module,
        "EventProfiler",
        lambda **_: SimpleNamespace(record=lambda _: None),
    )
    mesh = replace(ParallelContext.single("cpu"), cp_rank=1, cp_size=2)
    backend: Any = MultiViewRenderBackend(
        ChunkConfig(),
        RasterConfig(),
        settings=MultiViewSettings(device="cpu", parallel=mesh, parallel_decode=True),
    )
    decoded: list[Any] = []
    monkeypatch.setattr(
        backend_module,
        "decoded_views",
        lambda latent, decode, *, parallel: decoded.append(parallel),
    )
    backend._poses = backend._drawn = SimpleNamespace(supply=lambda poses: None)
    backend._rollout = SimpleNamespace(
        step=lambda: (SimpleNamespace(start=1, end=4), None),
        latent_for=lambda start, end: torch.zeros(1),
    )
    backend._decode = lambda latent: latent
    trajectory = SimpleNamespace(
        rig_poses_world=np.repeat(np.eye(4, dtype=np.float32)[None], 12, axis=0),
        timestamps_us=np.arange(12),
        boundary_state_after_chunk=object(),
    )

    chunk = backend._render(trajectory, opening=False)

    assert chunk.frames == ()
    assert decoded == [mesh]


def test_a_rank_with_no_camera_moves_the_car_without_a_worker() -> None:
    """It draws nothing, so it uploads no scene, but its poses must keep step."""
    from cosmos3_nano_sim_auto.apps.interactive_drive.poses import SuppliedPoses

    poses = SuppliedPoses(
        np.eye(4, dtype=np.float32),
        opening_rig_to_world=np.eye(4, dtype=np.float32),
        speed_mps=30.0,
        seconds_per_frame=1.0 / 30.0,
    )
    driven = backend_module._DrivenWorker(
        None, poses=poses, drawn=_DrawnRig(None, cameras=1), owned=[]
    )

    driven.request(0, 1)
    assert driven.take(0, 1) == ([], 1)
    assert driven.take(1, 4) == ([], 12)
    assert poses.next_frame == 13


def test_a_driven_request_for_another_range_than_the_one_in_flight_is_refused() -> None:
    poses, controls, _drawn, _pipe = _driven(delay=0)
    poses.supply(_host(range(13)))
    worker = controls._worker
    worker.request(0, 1)

    with pytest.raises(ValueError, match="already in flight"):
        worker.request(1, 4)
