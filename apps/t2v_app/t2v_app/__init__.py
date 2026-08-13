# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text-to-video pipeline application for ``flashdreams-app``."""

from .provider import create_app_spec, parse_options

__all__ = ["create_app_spec", "parse_options"]
