# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parity against a dumped cosmos-framework run, on a GPU (manual).

Runs :mod:`cosmos3_nano_sim_transfer.probes.parity` on one reference sample
folder that has a tensor dump (see the probe's docstring), on the GPU type the
dump was made on, and checks the acceptance:

- prompt ids, text and vision positions and initial noise equal to the dump,
  and the dump's history limits following the reference's window;
- teacher-forced steps (the reference's own history replayed) within one
  step's numerics of the reference: at least :data:`TEACHER_FORCED_MIN_DB`
  for every latent after the first, :data:`FIRST_LATENT_MIN_DB` for the first,
  which has no history to anchor it. Measured on an RTX PRO 6000 against
  cosmos-framework eager: 47 to 53 dB with the right history, both full and
  window 30 / 3 sinks; keeping full history on the windowed dump drops to
  37.5 dB within 20 latents of the first eviction;
- when a second cosmos-framework run of the same request is given, the free
  rollout on the dump's own control latents as close to the reference,
  second by second, as that run is (within :data:`FLOOR_TOLERANCE_DB`).

The package's own full path (its own control encoding) is reported against
the same floor but not asserted: its control latents differ from
cosmos-framework's in the last bits, and the free rollout amplifies that.
Needs about 40 GB of GPU memory.

Environment: ``COSMOS3_TRANSFER_RELEASE`` (release snapshot),
``COSMOS3_TRANSFER_PARITY_REF`` (sample folder), optionally
``COSMOS3_TRANSFER_PARITY_FLOOR`` (the other run's ``vision.mp4``),
``COSMOS3_TRANSFER_VAE`` (``Wan2.2_VAE.pth``) and ``COSMOS3_TRANSFER_PARITY_OUT``.
Run with ``pytest -p no:manual_marker -m manual``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.manual

TEACHER_FORCED_MIN_DB = 44.0
"""Latent PSNR every teacher-forced latent after the first must reach."""

FIRST_LATENT_MIN_DB = 30.0
"""Latent PSNR of the first latent (measured 35.7 to 37.3 dB; attention kernels
alone move it by about 2.4 dB)."""

FLOOR_TOLERANCE_DB = 0.5
"""Measurement allowance below the floor: per-second values of equivalent runs
differ by a few tenths of a dB (decoding a video file instead of the frames in
memory alone moves them by up to 0.3 dB)."""

RELEASE = os.environ.get("COSMOS3_TRANSFER_RELEASE")
REF = os.environ.get("COSMOS3_TRANSFER_PARITY_REF")


@pytest.mark.skipif(
    not (RELEASE and REF and torch.cuda.is_available()),
    reason="needs a GPU, COSMOS3_TRANSFER_RELEASE and COSMOS3_TRANSFER_PARITY_REF",
)
def test_parity_against_cosmos_framework(tmp_path: Path) -> None:
    from cosmos3_nano_sim_transfer.probes.parity import ReferenceRun, main

    assert REF is not None and RELEASE is not None
    ref = ReferenceRun.load(REF)
    label = ref.label
    argv = ["--release", RELEASE, "--ref", REF]
    argv += ["--out", os.environ.get("COSMOS3_TRANSFER_PARITY_OUT", str(tmp_path))]
    if vae := os.environ.get("COSMOS3_TRANSFER_VAE"):
        argv += ["--vae", vae]
    if floor := os.environ.get("COSMOS3_TRANSFER_PARITY_FLOOR"):
        argv += ["--floor", f"{label}={floor}"]
    (metrics,) = main(argv)

    checks = metrics["checks"]
    assert checks["prompt_ids_equal"]
    assert checks["text_positions_equal"]
    assert checks["vision_positions_max_abs_diff"] == 0.0
    assert all(checks["initial_noise_bit_equal"].values())
    if ref.full_history:
        assert checks.get("reference_history_limits_follow_window", True)
    else:
        # A windowed reference must show its limits: the dump's history limits
        # are what tie the teacher-forced bar to the right window.
        assert checks.get("reference_history_limits_follow_window") is True, (
            "the dump records no history limits for a windowed reference"
        )
    forced = metrics["teacher_forced"]["latent_psnr_vs_dump"]
    assert forced[0] >= FIRST_LATENT_MIN_DB, f"latent 0: {forced[0]:.1f} dB"
    worst = min(range(1, len(forced)), key=forced.__getitem__)
    assert forced[worst] >= TEACHER_FORCED_MIN_DB, (
        f"latent {worst}: {forced[worst]:.1f} dB"
    )
    assert metrics["ref_controls"]["finite"] and metrics["own_path"]["finite"]
    if floor:
        worst = metrics["ref_controls"]["vs_floor"]["worst_delta"]
        assert worst >= -FLOOR_TOLERANCE_DB, f"{worst:.2f} dB below the floor"
