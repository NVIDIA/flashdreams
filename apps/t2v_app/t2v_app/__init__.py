# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text-to-video pipeline application for ``flashdreams-app``."""

from .provider import add_arguments, create_app_spec

__all__ = ["add_arguments", "create_app_spec"]
