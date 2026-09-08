# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Interactive configuration tool for driving input devices.

Provides ``interactive-drive-configuration``: a small Tkinter wizard that
auto-detects a connected steering wheel or game controller, calibrates its
axes by capturing live input, and writes a local profile YAML the demo
runtime discovers automatically. No device make/model is hardcoded; the
only product-identifying string (the evdev device name used for detection)
is captured at runtime and written solely into the user's local profile.
"""
