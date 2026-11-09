# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The parity probe's bookkeeping: reference runs, history, PSNR per second."""

from __future__ import annotations

import json
import math
from argparse import Namespace
from pathlib import Path

import pytest
import torch
from cosmos3_nano_sim_transfer.config import HistoryProfile
from cosmos3_nano_sim_transfer.impl.decode import LatentDecoder, to_uint8
from cosmos3_nano_sim_transfer.probes.parity import (
    PSNR_CAP,
    Probe,
    ReferenceRun,
    _reference_history_limits,
    against_floor,
    frame_psnr,
    history_from_setup,
    latent_psnr,
    parse_history,
    per_second,
)

pytestmark = pytest.mark.ci_cpu


def _sample(
    tmp_path: Path, *, window: str | None = "30", sinks: str = "3", **extra
) -> Path:
    """A cosmos-framework run folder with one sample, as the inference CLI writes it."""
    run = tmp_path / "run"
    sample = run / "clip"
    sample.mkdir(parents=True)
    control = tmp_path / "seg.mp4"
    control.write_bytes(b"")
    args = {
        "name": "clip",
        "prompt": "A street.",
        "seed": 7,
        "num_frames": 201,
        "fps": 30,
        "num_steps": 4,
        "guidance": 1.0,
        "control_guidance": 1.0,
        "autoregressive": True,
        "num_conditional_frames": 0,
        "share_vision_temporal_positions": True,
        "emphasize_control_in_prompt": False,
        "transfer_resize_mode": "direct_resize",
        "seg": {"control_path": str(control), "weight": 1.0},
        "edge": None,
        "depth": None,
        "blur": None,
    }
    args.update(extra)
    (sample / "sample_args.json").write_text(json.dumps(args))
    overrides = ["model.config.teacher_forcing_frames_per_chunk=1"]
    if window is not None:
        overrides += [
            f"model.config.kv_cache_inference_size={window}",
            f"model.config.attention_sink_size={sinks}",
        ]
    (run / "setup_args.json").write_text(
        json.dumps({"experiment_overrides": overrides})
    )
    return sample


def test_reference_run_reads_request_and_history(tmp_path: Path) -> None:
    ref = ReferenceRun.load(_sample(tmp_path))
    assert (ref.name, ref.seed, ref.num_frames, ref.fps, ref.hint) == (
        "clip",
        7,
        201,
        30.0,
        "seg",
    )
    assert ref.num_latents == 51
    assert ref.history == HistoryProfile(30, 3)
    assert ref.control_path == tmp_path / "seg.mp4"
    assert ref.video_path == tmp_path / "run" / "clip" / "vision.mp4"
    assert ref.dump_path is None
    assert ref.emphasize_control_in_prompt is False


def test_reference_run_retains_explicit_control_emphasis(tmp_path: Path) -> None:
    ref = ReferenceRun.load(
        _sample(tmp_path, emphasize_control_in_prompt=True, aspect_ratio="16,9")
    )
    assert ref.emphasize_control_in_prompt is True
    assert ref.prompt_aspect_ratio == "16,9"


def test_legacy_reference_without_emphasis_keeps_baseline(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    args_path = sample / "sample_args.json"
    args = json.loads(args_path.read_text())
    del args["emphasize_control_in_prompt"]
    args_path.write_text(json.dumps(args))
    assert ReferenceRun.load(sample).emphasize_control_in_prompt is False


def test_probe_forwards_reference_emphasis_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cosmos3_nano_sim_transfer.probes import parity

    ref = ReferenceRun.load(
        _sample(tmp_path, emphasize_control_in_prompt=True, aspect_ratio="16,9")
    )
    probe = Probe.__new__(Probe)
    probe.args = Namespace(max_latents=None)
    probe.release = tmp_path / "release"
    recorded = []

    class StopBeforeGeneration(Exception):
        pass

    def tokenize(caption: str, **kwargs):
        recorded.append(kwargs)
        raise StopBeforeGeneration

    monkeypatch.setattr(parity, "prompt_token_ids", tokenize)
    with pytest.raises(StopBeforeGeneration):
        probe.run(ref, tmp_path / "probe")
    assert recorded[0]["hint"] == "seg"
    assert recorded[0]["emphasize_control_in_prompt"] is True
    assert recorded[0]["aspect_ratio"] == "16,9"


def test_full_history_keeps_every_latent_in_view(tmp_path: Path) -> None:
    ref = ReferenceRun.load(_sample(tmp_path, window="null", sinks="0"))
    assert ref.history == HistoryProfile(51, 0)


def test_missing_history_needs_an_override(tmp_path: Path) -> None:
    sample = _sample(tmp_path, window=None)
    with pytest.raises(ValueError, match="--history"):
        ReferenceRun.load(sample)
    assert ReferenceRun.load(sample, history="full").history == HistoryProfile(51, 0)
    assert ReferenceRun.load(sample, history="51/1").history == HistoryProfile(51, 1)


def test_dump_fallback_and_a_missing_control_input(tmp_path: Path) -> None:
    sample = _sample(tmp_path)
    (sample.parent / "ref.pt").write_bytes(b"")
    assert ReferenceRun.load(sample).dump_path == sample.parent / "ref.pt"
    (tmp_path / "seg.mp4").unlink()
    with pytest.raises(FileNotFoundError, match="--control clip="):
        ReferenceRun.load(sample)
    copy = sample / "control_seg.mp4"
    assert ReferenceRun.load(sample, control=copy).control_path == copy


@pytest.mark.parametrize(
    "extra",
    [
        {"guidance": 3.0},
        {"num_conditional_frames": 1},
        {"emphasize_control_in_prompt": "true"},
        {"aspect_ratio": 1.77},
        {"transfer_resize_mode": "crop"},
        {"control_guidance": 2.0},
        {"depth": {"control_path": "d.mp4"}},
        {"num_frames": 200},
    ],
)
def test_unsupported_requests_are_refused(tmp_path: Path, extra: dict) -> None:
    with pytest.raises(ValueError):
        ReferenceRun.load(_sample(tmp_path, **extra))


def test_history_parsing() -> None:
    assert parse_history("full", 151) == HistoryProfile(151, 0)
    assert parse_history("30/3", 151) == HistoryProfile(30, 3)
    setup = {"experiment_overrides": ["model.config.kv_cache_inference_size=None"]}
    assert history_from_setup(setup, 12) == HistoryProfile(12, 0)


def test_frame_psnr_and_cap() -> None:
    a = torch.zeros(4, 4, 3, dtype=torch.uint8)
    b = a.clone()
    assert frame_psnr(a, b) == PSNR_CAP
    b[0, 0, 0] = 255  # one channel of one pixel off by 255: mse = 255^2 / 48
    assert frame_psnr(a, b) == pytest.approx(10 * math.log10(48))


def test_latent_psnr_uses_the_reference_peak() -> None:
    ref = torch.tensor([2.0, -4.0])
    assert latent_psnr(ref + 0.4, ref) == pytest.approx(10 * math.log10(16 / 0.16))


def test_per_second_takes_the_median_of_each_second() -> None:
    values = [float(i) for i in range(61)]
    assert per_second(values, 30.0) == [14.5, 44.5, 60.0]


def test_against_floor_reports_each_second() -> None:
    result = against_floor([33.0, 30.0, 27.0], [32.5, 30.2, 27.0])
    assert result["delta_per_second"] == pytest.approx([0.5, -0.2, 0.0])
    assert result["all_seconds_at_or_above_floor"] is False
    assert result["worst_delta"] == pytest.approx(-0.2)


def test_to_uint8_rounds_and_clamps() -> None:
    pixels = torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0, 3.0])
    assert to_uint8(pixels).tolist() == [0, 0, 128, 191, 255, 255]


def test_unknown_decoder_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown decoder"):
        LatentDecoder("vqgan", device="cpu")


def test_labels_name_the_history(tmp_path: Path) -> None:
    assert ReferenceRun.load(_sample(tmp_path)).label == "clip_w30s3"
    full = ReferenceRun.load(_sample(tmp_path / "full", window="null", sinks="0"))
    assert full.full_history and full.label == "clip_full"


def _seeds(latents: int, sink: int, control: int | None, clean: int | None) -> dict:
    seeds = []
    for k in range(latents):
        seeds.append({"frame_idx": 2 * k, "sink_tokens": sink, "max_tokens": control})
        seeds.append({"frame_idx": 2 * k + 1, "sink_tokens": sink, "max_tokens": clean})
    return {"seeds": seeds}


def test_reference_history_limits_follow_its_window(tmp_path: Path) -> None:
    ref = ReferenceRun.load(_sample(tmp_path))  # window 30, 3 sinks
    tokens = 390
    good = _seeds(3, 6 * tokens, 52 * tokens, 53 * tokens)
    assert _reference_history_limits(good, ref, tokens)[
        "reference_history_limits_follow_window"
    ]
    off_by_one = _seeds(3, 6 * tokens, 54 * tokens, 55 * tokens)
    assert not _reference_history_limits(off_by_one, ref, tokens)[
        "reference_history_limits_follow_window"
    ]
    full = ReferenceRun.load(_sample(tmp_path / "full", window="null", sinks="0"))
    assert _reference_history_limits(_seeds(3, 0, None, None), full, tokens)[
        "reference_history_limits_follow_window"
    ]
    assert _reference_history_limits({"seeds": [{"frame_idx": 0}]}, ref, tokens) == {}
