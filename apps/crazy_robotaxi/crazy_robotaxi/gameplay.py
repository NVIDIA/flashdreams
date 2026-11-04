# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Robotaxi controls and live-edit behavior for the shared gameplay process."""

from functools import partial
from typing import cast

from omnidreams_game_engine.gameplay import GameplayProcess, SimulationDriver
from omnidreams_game_engine.simulation.ego_vehicle_kinematics import (
    EgoVehicleKinematics,
)
from omnidreams_game_engine.timeline import WorldSample

from crazy_robotaxi.controls import (
    BoundActionState,
    ControlsConfig,
    taxi_driver_command,
)
from crazy_robotaxi.factory import build_taxi_engine


class TaxiSimulationDriver(SimulationDriver):
    """Apply taxi controls and retain live-edit presentation for each physics step."""

    def __init__(self, engine, driver, controls):
        super().__init__(engine, driver)
        self.actions = BoundActionState(controls)
        self.live_edit = getattr(engine, "live_edit", None)

    def apply(self, events):
        for action in self.actions.apply(events):
            if action in {"style", "weather", "coins", "obstacle"}:
                self.request_action(action)
        return super().apply(events)

    def restore_input(self, events):
        self.actions.apply(events)
        super().restore_input(events)

    def request_action(self, value):
        if self.live_edit is not None:
            self.live_edit.request_action(value)

    def advance(self):
        step = self.engine.advance((taxi_driver_command(self.driver.command()),))
        return WorldSample(
            step,
            presentation=None
            if self.live_edit is None
            else self.live_edit.capture_frame(step.trajectory),
        )


def _build_simulation(engine_options, driver, controls, observe_physics):
    engine = build_taxi_engine(**engine_options, render_conditions=False)
    if observe_physics:
        cast(EgoVehicleKinematics, engine.simulation).set_physx_debug_enabled(True)
    return TaxiSimulationDriver(engine, driver, controls)



def build_gameplay_process(
    engine_options,
    driver,
    controls=ControlsConfig(),
    observer=None,
    observe_physics=False,
):
    """Launch the shared physics owner with Robotaxi's engine and controls."""
    return GameplayProcess(
        partial(_build_simulation, engine_options, driver, controls, observe_physics),
        observer,
    )
