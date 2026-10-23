# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NULL-model entry point for the imgui ui demo application."""

from imgui_ui_demo.window_size_app import WindowSizeApplication

from flashdreams.api_v2.application import IApplication


def create_app() -> IApplication:
    """Return a new window-size application."""
    return WindowSizeApplication()
