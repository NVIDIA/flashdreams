# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors

from io import StringIO
import json
import unittest
from unittest.mock import Mock, patch

import pytest

try:
    from rclpy.logging import LoggingSeverity as Severity
except ModuleNotFoundError as exc:
    raise unittest.SkipTest("Source ROS 2 Humble first") from exc

from ros2.flashdreams_bridge import model_logs

pytestmark = pytest.mark.ci_cpu


class TestModelLogs(unittest.TestCase):
    def test_model_unit_scope_clock_levels_and_shutdown(self):
        def row(text, priority):
            return json.dumps(
                {
                    "MESSAGE": text,
                    "PRIORITY": priority,
                    "__REALTIME_TIMESTAMP": "1700000000123456",
                }
            )

        lines = [
            "unreadable journal",
            "unreadable journal",
            row("2026-11-24 15:01:02.645 | WARNING  | physics:step:43 - paused", 6),
            row("Traceback (most recent call last):", 6),
            row("    demo()", 6),
            row("OSError: address in use", 6),
            row([116, 255], 2),
        ]
        node = Mock()
        with patch.object(model_logs.subprocess, "Popen") as popen:
            model_logs.ModelLogs(node, "").close()
            popen.assert_not_called()
            popen.return_value.stdout = StringIO("\n".join(lines))
            relay = model_logs.ModelLogs(node, model_logs.MODEL_UNIT)
            relay.thread.join(timeout=2)
            relay.close()
            relay.close()
        command = popen.call_args.args[0]
        self.assertIn("--user", command)
        self.assertIn("--user-unit=flashdreams-ros-model.service", command)
        messages = [call.args[0] for call in relay.publisher.publish.call_args_list]
        self.assertEqual(
            [msg.level for msg in messages],
            [
                Severity.WARN,
                Severity.ERROR,
                Severity.INFO,
                Severity.ERROR,
                Severity.FATAL,
            ],
        )
        self.assertTrue(all(msg.name == model_logs.MODEL_UNIT for msg in messages))
        self.assertTrue(
            all(
                msg.stamp.sec == 1700000000 and msg.stamp.nanosec == 123456000
                for msg in messages
            )
        )
        self.assertEqual(messages[2].msg, "    demo()")
        self.assertEqual(messages[-1].msg, "t�")
        self.assertEqual(
            node.get_logger().warning.call_count, 2
        )  # One parse warning, one close notice.
        popen.return_value.terminate.assert_called_once()
