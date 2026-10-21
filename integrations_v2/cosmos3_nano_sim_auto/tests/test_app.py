# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the v2 application surface.

No GPU and no checkpoint. Two halves: the argument handling, resolved session
description and mosaic, which are plumbing worth pinning because tiling eleven
cameras into a grid produces a picture whatever you get wrong -- so the tests
check *which* tile each camera landed in rather than that the size is right.

The other half is the model loop, which is the new logic rather than the
plumbing. ``step`` decodes ``[decoded_to, chunk_end)`` rather than the chunk it
just generated, so that an image2video clip's opening frames reach the output;
off by one there and the clip drops or repeats its opening while still looking
fine. A tiny network runs on CPU and the decoder is swappable, so a real step is
affordable here -- only the checkpoint and the pictures need the GB300.
"""

from __future__ import annotations

import argparse
import math
import queue
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from cosmos3_nano_sim_auto.config import (
    DEFAULT_ATTENTION_SCOPE,
    DEFAULT_DENOISE_STEPS,
    DEFAULT_HISTORY_FRAMES,
)
from cosmos3_nano_sim_auto.impl import app as app_module
from cosmos3_nano_sim_auto.impl.app import (
    DEFAULT_MASKLESS_ATTENTION,
    Cosmos3NanoSimAutoApplication,
    Cosmos3NanoSimAutoModelLoop,
    Cosmos3NanoSimAutoModelState,
    Cosmos3NanoSimAutoSession,
    _build,
    _frames_the_controls_allow,
    _mosaic_grid,
    _peak_reserved,
    create_app,
    mosaic,
    torn,
)
from cosmos3_nano_sim_auto.impl.packing import DEFAULT_CHUNK_FRAMES, ClipGeometry
from cosmos3_nano_sim_auto.impl.rollout import ChunkRollout
from cosmos3_nano_sim_auto.inputs import MultiviewSample, Prompt
from cosmos_interactive._network import (
    CosmosMoTNetwork,
    CosmosMoTNetworkConfig,
)
from cosmos_interactive._spec import Cosmos3InteractiveSpec, LLMSpec, VFMSpec

from flashdreams.accelerated.multi_head_attention.optimized import SDPABackend
from flashdreams.core.distributed.parallel import ParallelContext
from flashdreams.runtime_v2.session_desc import PresentationMode, SessionDesc
from flashdreams.runtime_v2.user_input_events import UserInputEvents
from flashdreams.runtime_v2.video_tensor import VideoTensorLayout

TEST_SAMPLE = Path(__file__).with_name("sample.jsonl")


pytestmark = pytest.mark.ci_cpu


def _sample(views: int = 11) -> MultiviewSample:
    return MultiviewSample.load(TEST_SAMPLE).trimmed_to(views)


# --- a network small enough to step on CPU ----------------------------------

SPEC = Cosmos3InteractiveSpec(
    llm=LLMSpec(
        hidden_size=32,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=16,
        vocab_size=100,
        mrope_section=(2, 1, 1),
    ),
    vfm=VFMSpec(vae_latent_channels=4, latent_patch_size=2, time_embed_freq_dim=8),
)
GEOMETRY_FRAMES = 5
"""Latent frames per view. Five, so a text2video clip is the singleton frame 0
and then one full chunk, and an i2v clip is one chunk and a short one."""


def build_network(seed: int = 0) -> CosmosMoTNetwork:
    torch.manual_seed(seed)
    return (
        CosmosMoTNetworkConfig(
            spec=SPEC, dtype=torch.float32, include_vision_tower=False
        )
        .setup()
        .eval()
    )


def _geometry(condition_frames: int = 0, frames: int = GEOMETRY_FRAMES) -> ClipGeometry:
    return ClipGeometry(
        num_views=2,
        frames_per_view=frames,
        patch_h=1,
        patch_w=2,
        frames_per_chunk=4,
        condition_frames=condition_frames,
    )


def _cpu_rollout(
    network: CosmosMoTNetwork,
    *,
    condition_frames: int = 0,
    frames: int = GEOMETRY_FRAMES,
    **kwargs,
) -> ChunkRollout:
    geometry = _geometry(condition_frames, frames)
    generator = torch.Generator().manual_seed(1)
    width = SPEC.vfm.patch_latent_dim
    tokens = geometry.num_views * frames * geometry.spatial_tokens
    conditions = None
    if condition_frames:
        conditions = torch.randn(
            geometry.num_views * condition_frames * geometry.spatial_tokens,
            width,
            generator=generator,
        )
    return ChunkRollout(
        network,
        geometry=geometry,
        control_tokens=torch.randn(tokens, width, generator=generator),
        condition_tokens=conditions,
        text_ids=torch.randint(0, SPEC.llm.vocab_size, (3,), generator=generator),
        **kwargs,
    )


_NO_EVENTS = UserInputEvents([])
"""This app's ``step`` deletes its events; the rig is driven by the sample."""


# --- mosaic -----------------------------------------------------------------


def _views(count: int, *, frames: int = 2, size: int = 4) -> torch.Tensor:
    """One camera per tile, each filled with its own view index."""
    views = torch.zeros(count, frames, 3, size, size)
    for view in range(count):
        views[view] = float(view)
    return views


def test_cameras_tile_in_rig_order_left_to_right_then_down() -> None:
    """A transposed grid still decodes; only the camera positions give it away."""
    tiles = mosaic(_views(4, size=4), columns=2, rows=2)

    assert tiles.shape == (2, 3, 8, 8)
    # View 1 belongs top-right, not bottom-left.
    assert tiles[0, 0, 0, 4] == pytest.approx(1.0)
    assert tiles[0, 0, 4, 0] == pytest.approx(2.0)
    assert tiles[0, 0, 4, 4] == pytest.approx(3.0)


def test_an_unused_cell_is_black_rather_than_an_error() -> None:
    """Eleven views do not make a rectangle, and the empty twelfth is honest."""
    tiles = mosaic(_views(11, size=4), columns=4, rows=3)

    assert tiles.shape == (2, 3, 12, 16)
    # Cell 11 is the bottom-right one, and nothing filled it.
    assert tiles[0, 0, 8, 12] == pytest.approx(-1.0)


def test_the_frame_axis_survives_the_tiling() -> None:
    views = _views(2, frames=5, size=4)
    views[:, 3] = 9.0

    tiles = mosaic(views, columns=2, rows=1)

    assert tiles.shape == (5, 3, 4, 8)
    assert torch.all(tiles[3] == 9.0)


def test_more_views_than_cells_is_refused() -> None:
    with pytest.raises(ValueError, match="do not fit"):
        mosaic(_views(5), columns=2, rows=2)


def test_a_clip_without_a_view_axis_is_refused() -> None:
    with pytest.raises(ValueError, match=r"\[V, T, 3, H, W\]"):
        mosaic(torch.zeros(2, 3, 4, 4), columns=1, rows=1)


def test_tearing_a_mosaic_up_gives_back_the_views_it_was_made_of() -> None:
    """Anything reading a run's output has to undo this exactly, tile for tile."""
    views = _views(11, size=4)

    assert torch.equal(
        torn(mosaic(views, columns=4, rows=3), views=11, columns=4, rows=3), views
    )


def test_tearing_up_drops_the_cells_the_tiling_left_black() -> None:
    frame = mosaic(_views(11, size=4), columns=4, rows=3)

    assert torn(frame, views=11, columns=4, rows=3).shape[0] == 11


def test_a_frame_that_does_not_divide_into_the_grid_is_refused() -> None:
    with pytest.raises(ValueError, match="does not divide"):
        torn(torch.zeros(2, 3, 10, 16), views=11, columns=4, rows=3)


def test_tearing_out_more_views_than_the_grid_holds_is_refused() -> None:
    with pytest.raises(ValueError, match="do not fit"):
        torn(torch.zeros(2, 3, 8, 8), views=5, columns=2, rows=2)


def test_a_mosaic_without_a_frame_axis_is_refused() -> None:
    with pytest.raises(ValueError, match=r"\[T, 3, H, W\]"):
        torn(torch.zeros(2, 2, 3, 4, 4), views=1, columns=1, rows=1)


@pytest.mark.parametrize(
    ("views", "grid"),
    [(1, (1, 1)), (2, (2, 1)), (4, (2, 2)), (5, (3, 2)), (11, (4, 3))],
)
def test_the_grid_stays_as_square_as_the_view_count_allows(
    views: int, grid: tuple[int, int]
) -> None:
    assert _mosaic_grid(views) == grid


# --- session description ----------------------------------------------------


@pytest.fixture(autouse=True)
def local_sample_for_config_tests(monkeypatch):
    """Use synthetic metadata for model/session tests without a scene installation."""
    original = app_module._sample_for

    def sample_for(args):
        return (
            _sample()
            if args.sample is None and args.draw_map is None
            else original(args)
        )

    monkeypatch.setattr(app_module, "_sample_for", sample_for)


def _config(**overrides):
    app = Cosmos3NanoSimAutoApplication()
    args = ["--views", "11", "--frames", "33"]
    for key, value in overrides.items():
        flag = f"--{key.replace('_', '-')}"
        # BooleanOptionalAction switches use --flag/--no-flag; giving them a
        # value is an argparse error, and `--flag False` would set the flag.
        if value is True:
            args.append(flag)
        elif value is False:
            args.append(f"--no-{key.replace('_', '-')}")
        else:
            args += [flag, str(value)]
    app.init(args)
    return app._config, app._sample


@pytest.mark.parametrize("device", ["cpu", "cuda:3", "cuda"])
def test_single_device_selection_reaches_the_runtime_context(monkeypatch, device):
    original = ParallelContext.single("cuda:0")
    monkeypatch.setattr(
        "cosmos3_nano_sim_auto.impl.app.init_parallel", lambda: original
    )
    config, sample = _config(device=device)
    session = Cosmos3NanoSimAutoSession(config, SessionDesc(), sample)
    expected = original.device if device == "cuda" else torch.device(device)
    assert session.parallel_context.device == expected
    assert config.device == device
    if device == "cuda":
        assert session.parallel_context is original


@pytest.mark.parametrize("device", ["cpu", "cuda:3"])
def test_a_mesh_rejects_an_explicit_device_override(monkeypatch, device):
    monkeypatch.setattr("cosmos3_nano_sim_auto.impl.app.init_parallel", _worker_context)
    with pytest.raises(ValueError, match="Drop --device"):
        _config(device=device)


def test_the_session_reports_the_mosaic_size_not_the_one_it_was_asked_for() -> None:
    """The frame size is the model's, so the sink has to be told the real one."""
    config, sample = _config()
    session = Cosmos3NanoSimAutoSession(config, SessionDesc(), sample)

    width, height = sample.view_size
    assert session.session_desc.video_width == width * 4
    assert session.session_desc.video_height == height * 3
    assert session.session_desc.frames_per_second_for_step == 30


def test_one_camera_reports_that_camera_s_size() -> None:
    config, sample = _config(camera="camera_front_wide_120fov")
    session = Cosmos3NanoSimAutoSession(config, SessionDesc(), sample)

    width, height = sample.view_size
    assert (session.session_desc.video_width, session.session_desc.video_height) == (
        width,
        height,
    )


def test_a_layout_this_app_does_not_produce_is_refused() -> None:
    config, sample = _config()

    with pytest.raises(ValueError, match="tchw"):
        Cosmos3NanoSimAutoSession(
            config,
            SessionDesc(output_layout=VideoTensorLayout.bcthw),
            sample,
        )


# --- application ------------------------------------------------------------


def test_the_factory_returns_an_uninitialized_app() -> None:
    """The registry calls this with no arguments and loads nothing yet."""
    app = create_app()

    assert isinstance(app, Cosmos3NanoSimAutoApplication)
    with pytest.raises(RuntimeError, match="init\\(\\) must run"):
        app.create_session(SessionDesc())


def test_the_app_asks_to_present_on_demand_before_it_is_initialized() -> None:
    """Presenting every UI tick padded a 1.2 second clip out to 107 of freezes.

    The runtime asks for this before ``init``, so it has to answer without any
    parsed arguments.
    """
    assert (
        Cosmos3NanoSimAutoApplication().session_desc().presentation_mode
        is PresentationMode.ON_DEMAND
    )


def test_the_session_keeps_the_pacing_it_was_asked_for() -> None:
    """Size and rate are the session's to replace; how it is presented is not."""
    config, sample = _config()

    session = Cosmos3NanoSimAutoSession(
        config,
        SessionDesc(presentation_mode=PresentationMode.CONTINUOUS),
        sample,
    )

    assert session.session_desc.presentation_mode is PresentationMode.CONTINUOUS


def test_more_views_than_the_sample_has_is_refused() -> None:
    app = Cosmos3NanoSimAutoApplication()

    with pytest.raises(ValueError, match="the sample has"):
        app.init(["--views", "99"])


def test_a_camera_outside_the_chosen_views_is_refused() -> None:
    # --views 2 --camera <the eleventh> would otherwise index past the rig and
    # emit whichever camera happened to sit at that position.
    app = Cosmos3NanoSimAutoApplication()

    with pytest.raises(ValueError, match="not among the first 2"):
        app.init(["--views", "2", "--camera", "camera_rear_fisheye_200fov"])


def test_the_sample_is_trimmed_before_the_session_sees_it() -> None:
    _config_, sample = _config(views=4)

    assert sample.num_views == 4
    assert len(sample.control_paths) == 4


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--views", "0"),
        ("--frames", "-1"),
        ("--fps", "0"),
        ("--history-frames", "-1"),
        ("--denoise-steps", "-1"),
    ],
)
def test_arguments_out_of_range_are_refused(flag: str, value: str) -> None:
    app = Cosmos3NanoSimAutoApplication()

    with pytest.raises(ValueError):
        app.init([flag, value])


def _asked(**overrides) -> argparse.Namespace:
    """The arguments ``--frames 0`` reads, on a recorded sample by default."""
    asked = {
        "sample": "sample_3",
        "draw_map": None,
        "views": 4,
        "fps": 30.0,
        "outlive_caption": False,
    }
    return argparse.Namespace(**(asked | overrides))


def test_generated_controls_have_no_length_to_take() -> None:
    """They are one track drawn to whatever is asked for, so 0 cannot resolve."""
    with pytest.raises(ValueError, match="generated controls have none"):
        _frames_the_controls_allow(_sample(views=4), _asked(sample=None))


def test_the_captions_end_a_run_before_the_controls_do(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "available_pixel_frames", lambda *_, **__: 649)
    sample = replace(_sample(views=4), frame_range=(0, 200))

    # 200 described frames is 197 on the encoder's grid, not the 649 on disk.
    assert _frames_the_controls_allow(sample, _asked()) == 197


def test_outliving_the_caption_leaves_the_controls_in_charge(monkeypatch) -> None:
    monkeypatch.setattr(app_module, "available_pixel_frames", lambda *_, **__: 649)
    sample = replace(_sample(views=4), frame_range=(0, 200))

    assert _frames_the_controls_allow(sample, _asked(outlive_caption=True)) == 649


def test_a_sample_that_describes_no_span_leaves_the_controls_in_charge(
    monkeypatch,
) -> None:
    monkeypatch.setattr(app_module, "available_pixel_frames", lambda *_, **__: 401)
    sample = replace(_sample(views=4), frame_range=None)

    assert _frames_the_controls_allow(sample, _asked()) == 401


def test_compiling_without_a_bounded_cache_is_refused_at_the_command_line() -> None:
    """``--history-frames 0`` is how a run asks for no bound, and compiling wants one.

    Refused here as well as in the rollout, so the answer arrives before a
    checkpoint is loaded rather than after.
    """
    app = Cosmos3NanoSimAutoApplication()

    with pytest.raises(ValueError, match="bound the cache"):
        app.init(["--history-frames", "0", "--compile-denoise", "default"])


def test_a_sample_directory_without_a_record_says_what_to_run() -> None:
    app = Cosmos3NanoSimAutoApplication()

    with pytest.raises(FileNotFoundError, match="prepared scene"):
        app.init(["--sample", "/nonexistent/sample/dir"])


@pytest.mark.parametrize("value", ["nan", "inf"])
def test_a_frame_rate_that_is_not_a_number_is_refused(value: str) -> None:
    """``nan`` fails every comparison, so a bound on its own lets it through.

    It would then reach the prompt's stated duration and the mp4's frame rate as
    a number nothing can order. Finiteness has to be checked first.
    """
    app = Cosmos3NanoSimAutoApplication()

    with pytest.raises(ValueError, match="finite"):
        app.init(["--fps", value])


def test_the_history_window_defaults_to_bounding_the_cache() -> None:
    """A default that keeps the whole clip would make streaming the exception."""
    config, _sample = _config()

    assert config.history_frames == DEFAULT_HISTORY_FRAMES


def test_asking_for_no_history_bound_reads_as_no_bound() -> None:
    """0 is how a command line says "unbounded"; None is how the rollout hears it."""
    config, _sample = _config(history_frames=0)

    assert config.history_frames is None


def test_the_chunk_follows_the_rig_and_not_the_shared_spec() -> None:
    """Three is this checkpoint's; the spec's four describes the single-view one.

    Nothing in a run says which was used, so taking the spec's word for it
    would generate a whole clip at the wrong chunk without a symptom.
    """
    config, _sample = _config()

    assert DEFAULT_CHUNK_FRAMES == 3
    assert config.chunk_frames == 3


def test_asking_for_the_specs_chunk_reads_as_deferring_to_it() -> None:
    """0 is how a command line says "whatever the checkpoint ships"."""
    config, _sample = _config(chunk_frames=0)

    assert config.chunk_frames is None


def test_a_chosen_chunk_reaches_the_config_as_asked() -> None:
    config, _sample = _config(chunk_frames=4)

    assert config.chunk_frames == 4


def test_a_negative_chunk_is_refused() -> None:
    app = Cosmos3NanoSimAutoApplication()

    with pytest.raises(ValueError, match="--chunk-frames must be >= 0"):
        app.init(["--chunk-frames", "-1"])


def test_a_run_is_held_to_its_caption_unless_told_otherwise() -> None:
    """The refusal is the default because training always matched the two."""
    config, _sample = _config()

    assert config.outlive_caption is False


def test_asking_to_outlive_the_caption_reaches_the_config() -> None:
    config, _sample = _config(outlive_caption=True)

    assert config.outlive_caption is True


def test_attention_is_gated_by_the_block_mask_by_default() -> None:
    """The dense mask costs several times the memory for the same visibility."""
    config, _sample = _config()

    assert config.flex_attention is True


def test_the_block_mask_can_be_turned_off_for_a_machine_that_cannot_compile() -> None:
    """FlexAttention only pays off compiled, and not every backend can."""
    app = Cosmos3NanoSimAutoApplication()
    app.init(["--views", "11", "--frames", "33", "--no-flex-attention"])

    assert app._config is not None
    assert app._config.flex_attention is False
    assert app._config.attention_impl.sdpa_backend is SDPABackend.CUDNN


def test_four_views_use_generic_attention_and_the_exact_kv_arena() -> None:
    config, _sample = _config(views=4)

    assert config.attention_impl.flex_attention.mask_block_size == 128
    assert config.attention_impl.flex_attention.kernel_options == {}
    assert config.attention_kv_arena is True
    assert config.compile_denoise is None


def test_rtx_pro_6000_blackwell_profile_selects_the_eleven_view_winner() -> None:
    config, _sample = _config(views=11, optimization_profile="rtx-pro-6000-blackwell")
    options = config.attention_impl.flex_attention

    assert config.attention_impl.sdpa_backend is SDPABackend.FLEX
    assert options.block_size == 128
    assert options.mask_block_size == (256, 64)
    assert options.compile_dynamic is False
    assert options.kernel_options == {
        "BLOCK_N": 64,
        "PRESCALE_QK": True,
        "ROWS_GUARANTEED_SAFE": True,
    }
    assert config.accelerated_rms_norm is False
    assert config.attention_kv_arena is False
    assert config.compile_denoise == "default"


def test_rtx_pro_6000_blackwell_perf_profile_is_explicit_and_atomic() -> None:
    safe, _sample = _config(views=11, optimization_profile="rtx-pro-6000-blackwell")
    perf, _sample = _config(
        views=11, optimization_profile="rtx-pro-6000-blackwell-perf"
    )

    assert perf.optimization_profile == "rtx-pro-6000-blackwell-perf"
    assert perf.accelerated_rms_norm is True
    assert safe.accelerated_rms_norm is False
    assert perf.attention_impl == safe.attention_impl
    assert perf.compile_denoise == "default"
    assert (perf.denoise_steps, perf.history_frames) == (4, 12)
    # A "complete measured configuration" has to pin the chunk it was measured
    # at, or the claim quietly stops being true when the default moves.
    assert perf.chunk_frames == 4
    assert perf.attention_kv_arena is False


@pytest.mark.parametrize("views", [4, 7])
@pytest.mark.parametrize(
    "profile", ["rtx-pro-6000-blackwell", "rtx-pro-6000-blackwell-perf"]
)
def test_rtx_pro_6000_profile_refuses_unmeasured_view_counts(
    views: int, profile: str
) -> None:
    with pytest.raises(ValueError, match="validated only for 11 views"):
        _config(views=views, optimization_profile=profile)


@pytest.mark.parametrize("views", [7, 11])
def test_other_view_counts_keep_generic_attention_without_the_arena(views: int) -> None:
    config, _sample = _config(views=views)

    assert config.attention_impl.flex_attention.mask_block_size == 128
    assert config.attention_impl.flex_attention.kernel_options == {}
    assert config.accelerated_rms_norm is True
    assert config.attention_kv_arena is False
    assert config.compile_denoise is None


def test_accelerated_rms_norm_is_on_unless_a_profile_or_the_flag_says_otherwise():
    assert app_module._parse([]).accelerated_rms_norm is True
    assert (
        app_module._parse(["--no-accelerated-rms-norm"]).accelerated_rms_norm is False
    )
    safe = app_module._parse(
        ["--views", "11", "--optimization-profile", "rtx-pro-6000-blackwell"]
    )
    assert safe.accelerated_rms_norm is False
    forced = app_module._parse(
        [
            "--views",
            "11",
            "--optimization-profile",
            "rtx-pro-6000-blackwell",
            "--accelerated-rms-norm",
        ]
    )
    assert forced.accelerated_rms_norm is True


@pytest.mark.parametrize("views", [7, 11])
def test_attention_kv_arena_refuses_unqualified_view_counts(views: int) -> None:
    with pytest.raises(ValueError, match="quality-qualified only for 4 views"):
        _config(views=views, attention_kv_arena=True)


@pytest.mark.parametrize(
    "override",
    [
        "--no-flex-attention",
        "--compile-denoise",
    ],
)
def test_optimization_profile_refuses_low_level_overrides(override: str) -> None:
    app = Cosmos3NanoSimAutoApplication()
    args = ["--optimization-profile", "rtx-pro-6000-blackwell", override]
    if override == "--compile-denoise":
        args.append("off")

    with pytest.raises(ValueError, match="owns its kernel and compile policy"):
        app.init(args)


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--denoise-steps", "4"), ("--history-frames", "12"), ("--chunk-frames", "3")],
)
def test_perf_profile_refuses_schedule_overrides(flag: str, value: str) -> None:
    app = Cosmos3NanoSimAutoApplication()
    with pytest.raises(ValueError, match="complete measured configuration"):
        app.init(["--optimization-profile", "rtx-pro-6000-blackwell-perf", flag, value])


def test_the_denoise_pass_is_left_uncompiled_unless_asked_for() -> None:
    """Compiling costs a warmup up front and repays it only over a long run."""
    config, _sample = _config()

    assert config.compile_denoise is None


def test_switching_compilation_off_reads_as_no_mode() -> None:
    config, _sample = _config(compile_denoise="off")

    assert config.compile_denoise is None


def test_a_compile_mode_reaches_the_config_as_asked() -> None:
    config, _sample = _config(compile_denoise="max-autotune")

    assert config.compile_denoise == "max-autotune"


@pytest.mark.parametrize("mode", ["fastest", "reduce-overhead"])
def test_an_unsupported_or_unsafe_compile_mode_is_refused(mode: str) -> None:
    app = Cosmos3NanoSimAutoApplication()

    with pytest.raises(SystemExit):
        app.init(["--compile-denoise", mode])


def test_the_scope_defaults_to_the_one_the_checkpoint_was_trained_with() -> None:
    """The one default where the conservative choice is the dangerous one.

    Every other setting here fails loudly or measurably when it is wrong. A
    mismatched scope generates a plausible clip and reports nothing, so a
    default naming any checkpoint but the one being run is a silent fault, and
    pinning it is the only way this stays noticed.
    """
    config, _sample = _config()

    assert DEFAULT_ATTENTION_SCOPE == "decomposed"
    assert config.attention_scope == "decomposed"


def test_the_older_checkpoints_scope_is_still_reachable() -> None:
    config, _sample = _config(attention_scope="all_views")

    assert config.attention_scope == "all_views"


def test_the_fold_is_on_wherever_the_scope_can_be_folded() -> None:
    """The fold won on both counts, so nothing is left arguing for the mask.

    It is the arithmetic the joint checkpoints were trained under, and once the
    passes stopped building masks it also became the faster and the cheaper of
    the two. Left as ``None`` rather than ``True`` so that an unfoldable scope
    reached by default can be told from one asked for outright.
    """
    config, _sample = _config()

    assert DEFAULT_MASKLESS_ATTENTION is None
    assert config.maskless_attention is True


def test_a_scope_with_no_fold_is_not_made_to_refuse_its_own_default() -> None:
    """`--attention-scope all_views` on its own is a diagnostic, not a mistake.

    It cannot be folded, and asking for both is still refused. But a default
    that turned every such run into an error would make the diagnostic need a
    second flag to say what it already said.
    """
    config, _sample = _config(attention_scope="all_views")

    assert config.maskless_attention is False


def test_the_fold_reaches_the_config_when_it_is_asked_for() -> None:
    config, _sample = _config(maskless_attention=True)

    assert config.maskless_attention is True
    assert config.attention_scope == "decomposed"


def test_the_fold_refuses_the_scope_it_has_no_partition_for() -> None:
    """Caught at the flag rather than deep in the prefill.

    `all_views` folds to a single pass over everything, so accepting the pair
    would silently run the mask and report the fold -- which is the one outcome
    that makes a later measurement meaningless.
    """
    with pytest.raises(ValueError, match="all_views"):
        _config(maskless_attention=True, attention_scope="all_views")


def test_the_fold_needs_no_flex_attention_to_gate_it() -> None:
    """The point of the gathered folds: there is no mask to pick a backend for.

    This pair was refused while the passes were block masks. Now they are dense
    runs, so `--no-flex-attention` is not a contradiction -- it describes a
    mask this run does not build.
    """
    config, _sample = _config(maskless_attention=True, flex_attention=False)

    assert config.maskless_attention is True


def test_the_schedule_defaults_to_the_steps_the_model_team_baselines_on() -> None:
    """Four is the reference schedule their quality claims are measured against.

    Two is the delivery's speed-first setting and was the default here, but a
    clip generated at the wrong step count looks plausible rather than wrong,
    so the default follows the reference and speed stays a flag away.
    """
    config, _sample = _config()

    assert DEFAULT_DENOISE_STEPS == 4
    assert config.denoise_steps == 4


def test_asking_for_every_step_reads_as_no_trim() -> None:
    config, _sample = _config(denoise_steps=0)

    assert config.denoise_steps is None


def test_a_trimmed_schedule_reaches_the_config_as_asked() -> None:
    config, _sample = _config(denoise_steps=2)

    assert config.denoise_steps == 2


def test_the_spec_still_calls_the_schedule_what_the_build_reads() -> None:
    """``_build`` needs a loaded checkpoint, so nothing here exercises it.

    That makes a wrong attribute name in it a mistake the station reports after
    a seventy-second prefill rather than one CI reports in a second -- which is
    how ``spec.vfm.n`` got written in the first place. This pins the one name
    the trim depends on.
    """
    assert len(VFMSpec().fixed_step_t_list) >= 2


# --- model loop -------------------------------------------------------------


def _loop(
    rollout: ChunkRollout,
    sample: MultiviewSample,
    *,
    camera: str | None = None,
    parallel: ParallelContext | None = None,
):
    """A loop wired to a CPU rollout, so ``step`` runs without a checkpoint.

    ``step`` builds on its first call only when ``rollout`` is None, so seating
    one here is what lets the bookkeeping be tested at all -- the alternative is
    a GB300 and a 55-second checkpoint load per assertion.

    ``parallel`` stands a rank somewhere on a mesh. Nothing initialized a process
    group, so the agreement built from it is inert and the loop decides for
    itself -- which leaves exactly the part a single process can check: who
    publishes.
    """
    config, _ = _config(views=sample.num_views)
    if camera is not None:
        config = replace(config, camera=camera)
    if parallel is not None:
        config = replace(config, parallel=parallel)
    state = Cosmos3NanoSimAutoModelState(
        config=replace(config, device="cpu"),
        session_desc=SessionDesc(),
        sample=sample,
        network=rollout._network,
        rollout=rollout,
        decode=_decode,
    )
    loop = Cosmos3NanoSimAutoModelLoop()
    loop.register_session_loop_objects(
        state=state,
        frequency=0,
        shutdown_event=threading.Event(),
        failure_queue=queue.Queue(),
    )
    return loop, state


def _decode(latent: torch.Tensor) -> torch.Tensor:
    """``[V, C, T, H, W]`` -> ``[V, T, 3, H, W]``, four pixel frames per latent.

    Stands in for the real decoder. Only the frame count matters to these tests,
    so each pixel frame is filled with the latent frame index it came from and
    the picture itself is not the point.
    """
    views, _channels, frames, height, width = latent.shape
    pixels = torch.zeros(views, frames * 4, 3, height, width)
    for frame in range(frames * 4):
        pixels[:, frame] = float(frame)
    return pixels


def test_a_step_emits_the_frames_it_decoded_and_says_how_many() -> None:
    """``frame_count`` is what the sink paces by, so it has to be the real one."""
    network = build_network()
    rollout = _cpu_rollout(network)
    loop, state = _loop(rollout, _sample(2))

    results = loop.step(0, _NO_EVENTS)

    assert len(results) == 1
    result = results[0]
    frames = result.read_output()
    assert result.frame_count == frames.shape[0]
    assert frames.ndim == 4
    assert state.frames_emitted == result.frame_count


def test_a_step_reports_a_total_the_parts_add_up_inside() -> None:
    """The whole point of a ``step_s``: a cost no part covers has somewhere to go.

    The long run's later chunks each spent about three seconds outside both
    ``generate`` and ``decode``, which was invisible except as a gap between two
    log timestamps. Timing the step as well as its parts is what makes that
    number reportable, so the parts must sit inside the total and not beside it.
    """
    loop, _state = _loop(_cpu_rollout(build_network()), _sample(2))

    metrics = loop.step(0, _NO_EVENTS)[0].metrics

    parts = ("generate_s", "decode_s", "emit_s")
    assert all(metrics[part] >= 0.0 for part in parts)
    assert metrics["step_s"] >= sum(float(metrics[part]) for part in parts)


def test_generating_is_timed_apart_from_decoding_and_tiling() -> None:
    """Three numbers, not one: a stream repeats generate and little else."""
    loop, _state = _loop(_cpu_rollout(build_network()), _sample(2))

    metrics = loop.step(0, _NO_EVENTS)[0].metrics

    assert {"generate_s", "decode_s", "emit_s", "step_s"} <= set(metrics)


def test_step_metrics_identify_the_final_chunk_that_skips_clean_replay() -> None:
    loop, _state = _loop(_cpu_rollout(build_network()), _sample(2))

    first = loop.step(0, _NO_EVENTS)[0]
    final = loop.step(1, _NO_EVENTS)[0]

    assert first.metrics["clean_pass"] == 1
    assert final.metrics["clean_pass"] == 0
    assert loop.is_finished()


def test_a_step_says_how_many_frames_a_second_it_managed() -> None:
    """The first number anyone asks a demo for, and we had every part but it."""
    loop, state = _loop(_cpu_rollout(build_network()), _sample(2))

    result = loop.step(0, _NO_EVENTS)[0]
    metrics = result.metrics

    assert metrics["chunk_fps"] == pytest.approx(
        result.frame_count / float(metrics["step_s"])
    )
    # Every camera counted, which is the figure that describes a rig.
    assert metrics["chunk_rig_fps"] == pytest.approx(
        state.config.views * float(metrics["chunk_fps"])
    )


def test_the_steady_rate_accumulates_over_the_chunks_it_covers() -> None:
    """The rate a viewer gets: frames over wall clock, gaps between steps included.

    One slot of history means one chunk of warmup, so the window opens on the
    second chunk and every chunk after adds to it. ``warmup_chunks`` is what
    holds it back at all, since a chunk generated while the cache is still
    filling is cheaper and would read as the run's speed.
    """
    rollout = _cpu_rollout(build_network(), frames=13, history_frames=4)
    loop, state = _loop(rollout, _sample(2))
    assert rollout.warmup_chunks == 1

    filling = loop.step(0, _NO_EVENTS)[0]
    first = loop.step(1, _NO_EVENTS)[0]
    second = loop.step(2, _NO_EVENTS)[0]

    assert "steady_state_fps" not in filling.metrics
    assert state.steady_frames == first.frame_count + second.frame_count
    assert second.metrics["steady_state_fps"] > 0.0
    assert second.metrics["steady_state_rig_fps"] == pytest.approx(
        state.config.views * float(second.metrics["steady_state_fps"])
    )


def test_a_new_clip_forgets_the_last_one_s_steady_window() -> None:
    """Its cache refills, so the old window would time frames it never generated."""
    loop, state = _loop(_cpu_rollout(build_network()), _sample(2))
    loop.step(0, _NO_EVENTS)
    loop.step(1, _NO_EVENTS)
    assert state.steady_frames > 0

    loop.reset()

    assert state.steady_started_at is None
    assert state.steady_frames == 0


def test_peak_memory_is_reported_as_a_number_a_benchmark_can_read() -> None:
    """The log has had the peak for a while; a stats file could not read a string.

    Off CUDA there is no peak to report, and NaN is what the metrics sink drops
    rather than recording a zero that would look like a run using no memory.
    """
    loop, _state = _loop(_cpu_rollout(build_network()), _sample(2))

    metrics = loop.step(0, _NO_EVENTS)[0].metrics

    assert math.isnan(float(metrics["peak_reserved_gib"]))


def test_the_first_step_of_a_text2video_clip_decodes_frame_zero_alone() -> None:
    """One latent frame, not four: frame 0 is generated on its own.

    The docstring claimed 13 pixel frames here for a while, which is a
    four-latent-frame first decode -- true of neither mode.
    """
    loop, _state = _loop(_cpu_rollout(build_network()), _sample(2))

    frames = loop.step(0, _NO_EVENTS)[0].frame_count

    assert frames == 4  # one latent frame, and this decoder adds no warm-up


def test_the_conditioning_frames_reach_the_output_before_any_chunk_does() -> None:
    """An i2v clip's opening frame is real footage that no step generated.

    So the first decode has to start at latent 0 rather than at the chunk's own
    start, or the clip drops the frame it was given. ``decoded_to`` starting at
    ``condition_frames`` instead is a whole chunk missing from the front.
    """
    rollout = _cpu_rollout(build_network(), condition_frames=1)
    loop, state = _loop(rollout, _sample(2))

    first = loop.step(0, _NO_EVENTS)[0]

    # Chunk one covers latents 1..5, and latent 0 was given -- so five, not four.
    assert first.metrics["chunk_start"] == 1
    assert state.decoded_to == 5
    assert first.frame_count == 5 * 4


def test_every_latent_frame_is_decoded_exactly_once_across_the_clip() -> None:
    """The bookkeeping's real invariant: no frame dropped, none sent twice."""
    rollout = _cpu_rollout(build_network(), condition_frames=1)
    loop, state = _loop(rollout, _sample(2))

    total = 0
    while not loop.is_finished():
        total += loop.step(0, _NO_EVENTS)[0].frame_count

    assert state.decoded_to == GEOMETRY_FRAMES
    assert total == GEOMETRY_FRAMES * 4


def test_a_finished_loop_says_so_and_a_fresh_one_does_not() -> None:
    """``is_finished`` before the first step must be false, or nothing ever runs."""
    rollout = _cpu_rollout(build_network())
    loop, state = _loop(rollout, _sample(2))
    assert not loop.is_finished()

    while not loop.is_finished():
        loop.step(0, _NO_EVENTS)

    assert loop.is_finished()
    assert state.decoded_to == GEOMETRY_FRAMES


def test_reset_drops_the_decoder_rather_than_reusing_its_cache() -> None:
    """TAEHV-super carries temporal state, so a new clip needs a new decoder."""
    loop, state = _loop(_cpu_rollout(build_network()), _sample(2))
    loop.step(0, _NO_EVENTS)

    loop.reset()

    assert state.rollout is None
    assert state.decode is None
    assert state.decoded_to == 0
    assert state.frames_emitted == 0
    # And with no rollout it is not finished, so the next generation runs.
    assert not loop.is_finished()


def test_a_cpu_run_reports_no_peak_rather_than_a_wrong_one() -> None:
    """The log line runs on CPU too, where there is no reservation to report."""
    assert _peak_reserved("cpu") == "n/a"


# --- a rank nobody is watching ----------------------------------------------


def _worker_context() -> ParallelContext:
    """Rank 1 of a two-rank context axis, with no process group behind it."""
    return ParallelContext(
        tp_group=None,
        tp_rank=0,
        tp_size=1,
        cp_group=None,
        cp_rank=1,
        cp_size=2,
        device=torch.device("cpu"),
    )


def test_a_worker_rank_generates_the_chunk_and_publishes_no_channel() -> None:
    """The redundant decode and the redundant mp4, both gone.

    Every rank of a mesh computes the same chunk, so the one with the window
    decodes it and the rest return nothing. An empty channel list is what the v2
    runtime reads as a step that presented nothing; it used to raise, which is
    why every rank needed an ``--output-path`` of its own.
    """
    rollout = _cpu_rollout(build_network())
    loop, state = _loop(rollout, _sample(2), parallel=_worker_context())

    results = loop.step(0, _NO_EVENTS)

    assert results == []
    # Generated all the same: the chunk is what the other ranks are waiting on.
    assert rollout.chunks_done == 1
    # And nothing was decoded, which is the work being saved.
    assert state.decoded_to == 0
    assert state.frames_emitted == 0


def test_a_worker_rank_needs_no_decoder_at_all() -> None:
    """Proof by absence: a state with no decoder still steps on a worker rank.

    ``_chunk`` asserts on the decoder before using it, so a worker that reached
    that line would fail here rather than quietly decode.
    """
    loop, state = _loop(
        _cpu_rollout(build_network()), _sample(2), parallel=_worker_context()
    )
    state.decode = None

    assert loop.step(0, _NO_EVENTS) == []


@pytest.mark.parametrize("views", [1, 7])
@pytest.mark.parametrize("lidar", [False, True])
def test_a_worker_rank_does_not_build_an_unused_decoder(
    monkeypatch: pytest.MonkeyPatch, views: int, lidar: bool
) -> None:
    """Workers should not load a GPU decoder they never call."""
    config, sample = _config(views=views)
    if lidar:
        config = replace(
            config,
            lidar=SimpleNamespace(
                tokenizer=object(),
                stream=lambda: SimpleNamespace(condition_tokens=lambda: None),
            ),
        )
    network = build_network()
    setattr(network, "rig_view_embed", torch.nn.Identity())
    state = Cosmos3NanoSimAutoModelState(
        config=replace(config, device="cpu", parallel=_worker_context()),
        session_desc=SessionDesc(),
        sample=sample,
        network=network,
    )
    rollout = object()
    monkeypatch.setattr(
        "cosmos3_nano_sim_auto.impl.app.make_control_encode_fn",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        "cosmos3_nano_sim_auto.impl.app.generated_control_clips",
        lambda sample, **_kwargs: [object()] * sample.num_views,
    )
    monkeypatch.setattr(
        "cosmos3_nano_sim_auto.impl.app.control_latents",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "cosmos3_nano_sim_auto.impl.app.control_tokens",
        lambda *_args, **_kwargs: object(),
    )

    def tokenize(*args: object, **kwargs: object) -> Prompt:
        assert kwargs["lidar"] is lidar
        return Prompt(torch.zeros(1, dtype=torch.long), None)

    monkeypatch.setattr(app_module, "prompt_token_ids", tokenize)
    monkeypatch.setattr(
        "cosmos3_nano_sim_auto.impl.app.ChunkRollout",
        lambda *_args, **_kwargs: rollout,
    )
    monkeypatch.setattr(
        app_module, "JointChunkRollout", lambda *_args, **_kwargs: rollout
    )
    monkeypatch.setattr(
        "cosmos3_nano_sim_auto.impl.app.make_multiview_chunk_decoder",
        lambda **_kwargs: pytest.fail("worker constructed a decoder"),
    )

    built, decoder = _build(state)

    assert built is rollout
    assert decoder is None
    assert "decoder_load_s" not in state.metrics


def test_rank_zero_of_a_mesh_still_publishes() -> None:
    """The other half of the rule, so "publishes nothing" cannot mean "ever"."""
    main = replace(_worker_context(), cp_rank=0)
    loop, _state = _loop(_cpu_rollout(build_network()), _sample(2), parallel=main)

    assert main.is_main
    assert len(loop.step(0, _NO_EVENTS)) == 1


def test_a_runtime_reset_discards_the_previous_rollout() -> None:
    """The runtime distributes resets before calling the model hook."""
    loop, state = _loop(_cpu_rollout(build_network()), _sample(2))

    loop.reset()

    assert state.rollout is None
    assert state.decode is None
    assert not loop.is_finished()


def test_a_finished_rollout_is_what_stops_a_single_rank() -> None:
    """With nobody to agree with, the decision is the local one it always was."""
    loop, _state = _loop(_cpu_rollout(build_network()), _sample(2))

    while not loop.is_finished():
        loop.step(0, _NO_EVENTS)

    assert loop.is_finished()


def test_closing_releases_the_rollout_without_a_collective() -> None:
    """``close`` runs on every path out of the loop, including the silent ones."""
    loop, state = _loop(_cpu_rollout(build_network()), _sample(2))
    loop.step(0, _NO_EVENTS)

    loop.close()

    assert state.rollout is None
    assert state.decode is None


def test_one_camera_is_emitted_at_its_own_size_rather_than_tiled() -> None:
    """Compared against the mosaic rather than a written-out size.

    Two views tile 2x1, so the mosaic is exactly twice as wide as one camera and
    the same height -- which holds whatever the patch geometry works out to.
    """
    sample = _sample(2)

    tiled = _loop(_cpu_rollout(build_network()), sample)[0]
    single = _loop(_cpu_rollout(build_network()), sample, camera=sample.cameras[1])[0]

    whole = tiled.step(0, _NO_EVENTS)[0].read_output()
    one = single.step(0, _NO_EVENTS)[0].read_output()

    assert one.shape[-1] * 2 == whole.shape[-1]
    assert one.shape[-2] == whole.shape[-2]


@pytest.mark.parametrize("enabled", [False, True])
def test_parallel_camera_encoding_can_be_set_either_way(enabled: bool) -> None:
    config, _ = _config(parallel_encode=enabled)
    assert config.parallel_encode is enabled


def test_parallel_camera_encoding_is_on_by_default() -> None:
    config, _ = _config()
    assert config.parallel_encode is True


def test_fused_attention_merge_follows_maskless_attention_and_requires_it():
    assert app_module._parse([]).fused_attention_merge is True
    assert app_module._parse(["--no-maskless-attention"]).fused_attention_merge is False
    assert (
        app_module._parse(["--attention-scope", "all_views"]).fused_attention_merge
        is False
    )
    assert app_module._parse(["--fused-attention-merge"]).fused_attention_merge is True
    assert (
        app_module._parse(["--no-fused-attention-merge"]).fused_attention_merge is False
    )
    config, _sample = _config(fused_attention_merge=True)
    assert config.fused_attention_merge is True
    with pytest.raises(ValueError, match="needs --maskless-attention"):
        _config(fused_attention_merge=True, maskless_attention=False)
    with pytest.raises(ValueError, match="needs --maskless-attention"):
        _config(fused_attention_merge=True, attention_scope="all_views")


# --- the 7-camera 480p optimization switches (#48) --------------------------


def test_the_optimization_switches_are_off_unless_asked() -> None:
    config, _ = _config()
    assert (
        config.parallel_decode,
        config.compile_control_encode,
        config.decoder_compile,
    ) == (False, False, False)


def test_the_optimization_switches_reach_the_config() -> None:
    config, _ = _config(
        parallel_decode=True, compile_control_encode=True, decoder_compile=True
    )
    assert (
        config.parallel_decode,
        config.compile_control_encode,
        config.decoder_compile,
    ) == (True, True, True)


def test_prefetching_controls_is_off_unless_asked() -> None:
    assert _config()[0].prefetch_controls is False
    assert _config(prefetch_controls=True)[0].prefetch_controls is True
