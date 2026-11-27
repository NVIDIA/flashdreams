# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors
# SPDX-License-Identifier: Apache-2.0
"""Run with Humble and the built bridge overlay sourced, using /usr/bin/python3."""

from queue import Empty, Queue
from threading import Semaphore
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import pytest

try:
    from flashdreams_bridge.joy_driver import JoyDriver
except ModuleNotFoundError as exc:
    if exc.name not in ("flashdreams_bridge", "rclpy", "flashdreams_interfaces"):
        raise
    raise unittest.SkipTest("Source the ROS 2 bridge overlay first") from exc

pytestmark = pytest.mark.ci_cpu


class TestJoyDriver(unittest.TestCase):
    def test_edges_survive_slow_steps_and_disconnect_preserves_park(self):
        node = Mock(episode="initial")
        node.declare_parameter.return_value.value = 0.75
        node.context.ok.return_value = True
        requests, finish = Queue(), Semaphore(0)

        def exchange(request, response):
            requests.put(request)
            if not finish.acquire(timeout=5):
                raise TimeoutError("Test did not release the pending step")
            node.episode = request["episode_id"]
            response.code = "OK"
            response.next_step = request.get("expected_step", -1) + 1
            return response

        node.exchange = exchange
        with patch("flashdreams_bridge.joy_driver.monotonic", return_value=100) as now:
            driver = JoyDriver(node)

            def send(*held):
                driver.on_joy(
                    SimpleNamespace(
                        axes=[0.0] * 6, buttons=[int(i in held) for i in range(21)]
                    )
                )

            try:
                send()
                self.assertEqual(requests.get(timeout=2)["type"], "reset")
                finish.release()
                self.assertEqual(requests.get(timeout=2)["expected_step"], 0)
                send(0, 3)  # Tap A and Y while the simulation is busy.
                send()
                finish.release()
                self.assertEqual(requests.get(timeout=2)["type"], "reset")
                finish.release()
                request = requests.get(timeout=2)
                self.assertEqual((request["expected_step"], request["brake"]), (0, 1.0))
                finish.release()
                request = requests.get(timeout=2)
                self.assertEqual((request["expected_step"], request["brake"]), (1, 0.0))
                send(13)  # Park. Missed heartbeat must pause the simulation.
                now.return_value += 2
                finish.release()
                with self.assertRaises(Empty):
                    requests.get(timeout=0.1)
                send()
                request = requests.get(timeout=2)
                self.assertEqual(
                    (request["expected_step"], request["manual_gear"]), (2, "P")
                )
                send(3)
                driver.on_joy(SimpleNamespace(axes=[], buttons=[]))
                finish.release()
                with self.assertRaises(Empty):
                    requests.get(timeout=0.1)
                send()
                self.assertEqual(requests.get(timeout=2)["expected_step"], 3)
            finally:
                node.context.ok.return_value = False
                finish.release()
                driver.close()


if __name__ == "__main__":
    unittest.main()
