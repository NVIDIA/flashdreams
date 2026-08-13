# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level runner for one application and IO handler."""

from __future__ import annotations

from dataclasses import dataclass

from flashdreams.runtime.application import FlashDreamsApplication
from flashdreams.runtime.demo.contracts import DemoSpec, OutputSpec
from flashdreams.runtime.io_handler import IOHandler


@dataclass(frozen=True, slots=True)
class ApplicationRunner:
    """Bind an application to polymorphic input/output behavior."""

    application: FlashDreamsApplication
    io_handler: IOHandler

    def run(self) -> object:
        return self.io_handler.run(self)

    def create_driver_spec(self, output: OutputSpec) -> DemoSpec:
        """Build the internal contract consumed by existing session drivers."""
        return DemoSpec(
            model_id=self.application.model_id,
            input_mode=self.io_handler.input_mode,
            output=output,
            scenario=self.application.scenario,
            config=self.application.config,
            metadata={"realtime": self.io_handler.realtime},
        )


__all__ = ["ApplicationRunner"]
