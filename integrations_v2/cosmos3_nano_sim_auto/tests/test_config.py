# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for the settings the checkpoint owns.

The values themselves are pinned where they are read, beside the caller that
would be wrong without them, since a constant asserted to equal itself says
nothing. What is worth testing here is the trim, which has an off-by-one that
would cost a visible amount of quality and report nothing.
"""

from __future__ import annotations

import pytest
from cosmos3_nano_sim_auto.config import denoise_schedule

pytestmark = pytest.mark.ci_cpu


def test_trimming_takes_the_first_sigmas_and_not_a_sample() -> None:
    """The sampler appends 0.0 and walks pairs, so a prefix is N whole steps.

    Two sigmas become ``[1.0, 0.9375, 0.0]``: two steps, the last of them the
    jump to clean that ends every schedule. Any other subset would either change
    where the schedule starts or leave the latent short of clean.
    """
    full = (1.0, 0.9375, 0.8333, 0.625)

    assert denoise_schedule(full, 2) == [1.0, 0.9375]


def test_leaving_the_schedule_alone_says_so_rather_than_copying_it() -> None:
    assert denoise_schedule((1.0, 0.9375), None) is None


def test_a_schedule_cannot_be_asked_for_more_steps_than_it_was_distilled_for() -> None:
    """Slicing would silently return the four it has and report five."""
    with pytest.raises(ValueError, match="cannot be lengthened"):
        denoise_schedule((1.0, 0.9375, 0.8333, 0.625), 5)
