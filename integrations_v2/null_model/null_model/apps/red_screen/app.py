# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NULL-model entry point for the red screen application."""

from red_screen.app import RedScreenApplication

from flashdreams.api_v2.application import IApplication


def create_app() -> IApplication:
    """Return a new red screen application."""
    return RedScreenApplication()
