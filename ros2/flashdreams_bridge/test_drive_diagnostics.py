# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock, patch

import pytest

try:
    from builtin_interfaces.msg import Time
    from rclpy.clock import ClockType
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Source ROS 2 Humble first") from exc

from ros2.flashdreams_bridge import drive_diagnostics

pytestmark = pytest.mark.ci_cpu


class TestDriveDiagnostics(unittest.TestCase):
    def test_burst_pause_and_episode_reset(self):
        node = Mock(episode="first", last_error="", needs_reset=False, joy=None, active_request=None)
        node.get_clock.return_value.now.return_value.to_msg.return_value = Time()
        diag = drive_diagnostics.DriveDiagnostics(node)
        self.assertEqual(node.create_timer.call_args.kwargs["clock"].clock_type, ClockType.STEADY_TIME)

        def snapshot():
            diag.publish()
            return [
                {item.key: item.value for item in status.values}
                for status in diag.publisher.publish.call_args.args[0].status
            ]

        with patch.object(drive_diagnostics, "monotonic", return_value=10.0) as now:
            diag.observe(100000, ["front", "back"], reset=True)
            snapshot()
            now.return_value = 12.0
            for source in (200000, 300000, 400000, 500000):
                diag.observe(source, ["front", "back"], reset=False)
            diag.step_done(1.6)
            drive, front, back = snapshot()
            self.assertEqual(front, back)
            self.assertEqual(float(front["publish_hz_wall"]), 2.0)
            self.assertEqual(float(front["sample_hz_sim"]), 10.0)
            self.assertEqual(float(front["realtime_factor"]), 0.2)
            self.assertEqual(float(drive["last_step_wall_s"]), 1.6)
            for t in range(13, 19):
                now.return_value = t
                snapshot()
            self.assertEqual(float(snapshot()[0]["publish_hz_wall"]), 0.0)
            diag.observe(100000, ["other"], reset=True)
            drive, camera = snapshot()
            self.assertEqual(camera["episode_frames"], "1")
            self.assertNotIn("sample_hz_sim", camera)
