# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lingbot camera-to-video application for the FlashDreams v2 API."""

from .app import LingbotCam2VApplication, create_app

__all__ = ["LingbotCam2VApplication", "create_app"]
