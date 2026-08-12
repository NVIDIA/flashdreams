# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import numpy as np
import pytest
from omnidreams.interactive_drive.visual_flare import (
    CollisionVisualFlare,
    VisualFlareEventQueue,
    darken_rgb,
)

pytestmark = pytest.mark.ci_cpu


def test_collision_visual_flare_fades_in_holds_and_fades_out() -> None:
    flare = CollisionVisualFlare(
        fade_in_s=0.1,
        hold_s=0.1,
        fade_out_s=0.2,
        peak_opacity=0.8,
    )
    flare.trigger(now=10.0)

    assert flare.opacity(now=10.0) == 0.0
    assert flare.opacity(now=10.05) == pytest.approx(0.4)
    assert flare.opacity(now=10.15) == pytest.approx(0.8)
    assert flare.opacity(now=10.30) == pytest.approx(0.4)
    assert flare.opacity(now=10.41) == 0.0


def test_visual_flare_event_waits_for_target_frame_then_expires() -> None:
    triggers: list[str] = []
    events = VisualFlareEventQueue(duration_s=0.4)
    events.schedule(chunk_index=3, frame_index=5)

    events.update(lambda: triggers.append("triggered"), now=10.0)
    events.update(
        lambda: triggers.append("triggered"),
        displayed_position=(3, 4),
        now=10.1,
    )
    assert triggers == []
    assert len(events) == 1

    events.update(
        lambda: triggers.append("triggered"),
        displayed_position=(3, 5),
        now=10.2,
    )
    assert triggers == ["triggered"]
    assert len(events) == 1

    events.update(lambda: triggers.append("triggered"), now=10.61)
    assert triggers == ["triggered"]
    assert len(events) == 0


def test_darken_rgb_is_harsh_and_does_not_mutate_source() -> None:
    source = np.full((2, 2, 3), 200, dtype=np.uint8)

    darkened = darken_rgb(source, 0.88)

    assert np.all(darkened == 24)
    assert np.all(source == 200)


def test_darken_rgb_preserves_rgba_alpha() -> None:
    source = np.full((1, 1, 4), 200, dtype=np.uint8)

    darkened = darken_rgb(source, 0.5)

    np.testing.assert_array_equal(darkened[0, 0], [100, 100, 100, 200])


def test_collision_visual_flare_default_remains_visible_at_peak() -> None:
    flare = CollisionVisualFlare()
    flare.trigger(now=10.0)

    assert flare.opacity(now=10.06) == pytest.approx(0.72)
