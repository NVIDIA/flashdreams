# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NULL-model entry point for the color fade application."""

from color_fade.app import ColorFadeApplication

from flashdreams.api_v2.application import IApplication


def create_app() -> IApplication:
    """Return a new colour fade application."""
    return ColorFadeApplication()
