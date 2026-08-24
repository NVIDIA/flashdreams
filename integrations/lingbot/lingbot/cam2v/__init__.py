# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot specialization of the shared camera-to-video application."""

from .app import LingbotCam2VApplication, create_app

__all__ = ["LingbotCam2VApplication", "create_app"]
