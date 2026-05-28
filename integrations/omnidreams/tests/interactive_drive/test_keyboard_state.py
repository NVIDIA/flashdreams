# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for :class:`KeyboardState`'s :class:`RuntimeControls` contract.

The display loop depends on the rising-edge consume semantics for reset:
exactly one ``consume_reset_request`` call returns ``True`` per call to
``request_reset``, and rapid presses must coalesce so a single reset isn't
processed twice.
"""

from omnidreams.interactive_drive.input.keyboard import KeyboardState


def test_consume_reset_request_returns_false_when_no_reset_pending() -> None:
    keyboard = KeyboardState()
    assert keyboard.consume_reset_request() is False


def test_consume_reset_request_returns_true_once_per_request() -> None:
    keyboard = KeyboardState()
    keyboard.request_reset()
    assert keyboard.consume_reset_request() is True
    assert keyboard.consume_reset_request() is False


def test_repeated_request_reset_coalesces_to_one_consume() -> None:
    """Multiple presses of ``r`` between consumes must not double-fire.

    The loop tears down and rebuilds sim/pipeline on every ``True``; if
    rapid presses produced multiple ``True`` returns, the user would see
    the loading frame N times for N presses instead of once.
    """
    keyboard = KeyboardState()
    keyboard.request_reset()
    keyboard.request_reset()
    keyboard.request_reset()
    assert keyboard.consume_reset_request() is True
    assert keyboard.consume_reset_request() is False


def test_view_mode_reflects_set_view_mode() -> None:
    keyboard = KeyboardState()
    assert keyboard.view_mode == "rgb"
    keyboard.set_view_mode("hdmap")
    assert keyboard.view_mode == "hdmap"
