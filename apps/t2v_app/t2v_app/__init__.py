# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text-to-video pipeline application for ``flashdreams-app``."""

from flashdreams.core.pipeline_presets import PipelineProvider

from .provider import add_arguments, create_app

__all__ = ["PipelineProvider", "add_arguments", "create_app"]
