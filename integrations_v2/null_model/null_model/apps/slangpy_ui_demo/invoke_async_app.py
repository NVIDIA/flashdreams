# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NULL-model entry point for the slangpy ui demo application."""

from slangpy_ui_demo.invoke_async_app import ColorToggleApplication

from flashdreams.api_v2.application import IApplication


def create_app() -> IApplication:
    """Return a new color-toggle application."""
    return ColorToggleApplication()
