# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
from omnidreams.interactive_drive.cli import build_parser
from omnidreams.interactive_drive.cli_args import arg_was_explicit

pytestmark = pytest.mark.ci_cpu


def test_offload_text_encoder_flag_defaults_disabled() -> None:
    args = build_parser().parse_args([])

    assert args.offload_text_encoder is False


def test_offload_text_encoder_flag_enables() -> None:
    args = build_parser().parse_args(["--offload-text-encoder"])

    assert args.offload_text_encoder is True


def test_postprocess_preset_defaults_disabled() -> None:
    args = build_parser().parse_args([])

    assert args.postprocess_preset == ""


def test_postprocess_preset_accepts_rtx_super_resolution() -> None:
    args = build_parser().parse_args(["--postprocess-preset", "rtx-super-resolution"])

    assert args.postprocess_preset == "rtx-super-resolution"


def test_parser_records_explicit_arg_destinations() -> None:
    args = build_parser().parse_args(
        [
            "--manifest",
            "example_world_model_perf.yaml",
            "--offload-text-encoder",
            "--no-bev",
        ]
    )

    assert arg_was_explicit(args, "manifest")
    assert arg_was_explicit(args, "offload_text_encoder")
    assert arg_was_explicit(args, "bev")
    assert not arg_was_explicit(args, "camera")
