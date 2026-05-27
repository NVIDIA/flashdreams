# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from __future__ import annotations

import unittest

from interactive_drive.controls import KeyboardDriveController


class KeyboardDriveControllerTest(unittest.TestCase):
    def test_samples_expected_controls(self) -> None:
        controller = KeyboardDriveController()
        controller.on_key("w", True)
        controller.on_key("a", True)
        sample = controller.sample()
        self.assertEqual(sample.throttle, 1.0)
        self.assertEqual(sample.brake, 0.0)
        self.assertEqual(sample.steer, 1.0)

        controller.on_key("w", False)
        controller.on_key("space", True)
        sample = controller.sample()
        self.assertEqual(sample.throttle, 0.0)
        self.assertEqual(sample.brake, 1.0)


if __name__ == "__main__":
    unittest.main()
