# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors
# SPDX-License-Identifier: Apache-2.0

"""Run with python3 -m unittest discover -s ros2/flashdreams_bridge."""

import json
from pathlib import Path
import unittest

import pytest

from ros2.flashdreams_bridge.cockpit_joy import CockpitJoy

pytestmark = pytest.mark.ci_cpu


def sample(joy, *pressed, steer=0.0, brake=0.0, throttle=0.0):
    buttons = [int(i in pressed) for i in range(21)]
    return joy.update([steer, 0.0, 0.0, 0.0, brake, throttle], buttons)


class TestCockpitJoy(unittest.TestCase):
    def test_layout_keeps_twist_off_the_gamepad_topic(self):
        layout = Path(__file__).resolve().parents[1] / "examples" / "jlu_wireframe_layout.json"
        panels = json.loads(layout.read_text())["configById"]
        teleop_topics = {
            panel["topic"] for name, panel in panels.items() if name.startswith("Teleop!")
        }
        self.assertEqual(teleop_topics, {"/cmd_vel"})
        self.assertEqual(panels["RawMessages!joy"]["topicPath"], "/joy")

    def test_sticks_triggers_and_handbrake(self):
        joy = CockpitJoy()
        self.assertEqual(sample(joy, steer=0.05, throttle=-0.02).steer, 0.0)
        self.assertEqual(sample(joy, throttle=-0.02).throttle, 0.0)
        state = sample(joy, steer=0.5125, brake=-0.51, throttle=-1.0)
        self.assertAlmostEqual(state.steer, 0.35 * 0.5 + 0.65 * 0.5**2.2)
        self.assertEqual((state.gear, state.throttle, state.brake), ("D", 1.0, 0.5))
        self.assertEqual(sample(joy, steer=-1.0).steer, -1.0)
        self.assertEqual(sample(joy, 0).brake, 1.0)
        self.assertTrue(sample(joy, 0).handbrake)
        self.assertFalse(sample(joy, brake=-1.0).handbrake)
        self.assertEqual(sample(joy).brake, 0.0)

    def test_gear_edges_and_direct_selection(self):
        joy = CockpitJoy()
        self.assertEqual(sample(joy, 2).gear, "N")  # X down one, held
        self.assertEqual(sample(joy, 2, throttle=-1.0).throttle, 0.0)
        self.assertEqual(sample(joy, 2).gear, "N")
        sample(joy)
        self.assertEqual(sample(joy, 2).gear, "R")
        self.assertEqual(sample(joy, 1).gear, "N")
        for button, gear in [(11, "D"), (12, "R"), (13, "P"), (14, "N")]:
            state = sample(joy, button, throttle=-1.0)
            self.assertEqual(state.gear, gear)
            self.assertEqual(state.throttle, int(gear in ("D", "R")))
            self.assertEqual(state.brake, int(gear == "P"))
            self.assertEqual(sample(joy).gear, gear)

    def test_reset_and_camera_edges(self):
        joy = CockpitJoy()
        state = sample(joy, 3, 9)
        self.assertTrue(state.reset)
        self.assertEqual(state.camera_delta, -1)
        state = sample(joy, 3, 9)
        self.assertFalse(state.reset)
        self.assertEqual(state.camera_delta, 0)
        sample(joy)
        state = sample(joy, 3, 10)
        self.assertTrue(state.reset)
        self.assertEqual(state.camera_delta, 1)

    def test_invalid_samples_do_not_release_buttons(self):
        joy = CockpitJoy()
        sample(joy, 2)
        for axes, buttons in [
            ([], []),
            ([0] * 6, [0] * 14),
            ([2] * 6, [0] * 15),
            ([float("nan")] * 6, [0] * 15),
            ([0] * 6, [2] * 15),
        ]:
            with self.assertRaises(ValueError):
                joy.update(axes, buttons)
        self.assertEqual(sample(joy, 2).gear, "N")

    def test_release_preserves_gear_and_rearms_button_edges(self):
        joy = CockpitJoy()
        for dpad, gear, shift, next_gear in [
            (13, "P", 1, "R"),
            (12, "R", 1, "N"),
            (14, "N", 2, "R"),
            (11, "D", 2, "N"),
        ]:
            sample(joy, dpad, shift, 3)
            joy.release()
            self.assertEqual(joy.gear, gear)
            state = sample(joy, shift, 3)
            self.assertEqual(state.gear, next_gear)
            self.assertTrue(state.reset)
