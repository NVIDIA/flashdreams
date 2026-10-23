# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NULL-model entry point for the imgui ui demo application."""

from imgui_ui_demo.query_string_app import QueryStringApplication

from flashdreams.api_v2.application import IApplication


def create_app() -> IApplication:
    """Return a new query-string application."""
    return QueryStringApplication()
