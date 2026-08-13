# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concrete triangle model application."""

from __future__ import annotations

from flashdreams.runtime import InferenceConfig
from triangle_app import TriangleApp

from .model import TriangleRuntime

MODEL_ID = "triangle-model"


class TriangleModel(TriangleApp):
    model_id = MODEL_ID

    def create_runtime(self, config: InferenceConfig) -> TriangleRuntime:
        self.validate_config(config)
        return TriangleRuntime()


__all__ = ["MODEL_ID", "TriangleModel"]
