# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NULL-model entry point for the slangpy ui demo application."""

from slangpy_ui_demo.model_output_app import ModelOutputApplication

from flashdreams.api_v2.application import IApplication


def create_app() -> IApplication:
    """Return a new model-output application."""
    return ModelOutputApplication()
