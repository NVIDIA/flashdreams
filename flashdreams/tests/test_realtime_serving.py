# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from flashdreams.runtime.keyboard import KeyboardState

pytestmark = pytest.mark.ci_cpu


def test_keyboard_state_resolves_the_latest_opposing_key() -> None:
    state = KeyboardState()

    assert state.apply_event(event="keydown", key="ArrowLeft")
    assert state.apply_event(event="keydown", key="d")

    assert state.snapshot() == frozenset({"a", "d"})
    assert state.resolved_effective_keys() == frozenset({"d"})
