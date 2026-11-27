# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors
# SPDX-License-Identifier: Apache-2.0
"""Drive the Interactive Drive vehicle on a flat plane without camera generation."""

import argparse
from contextlib import closing
from pathlib import Path
import signal
from threading import Event
from time import monotonic

from interactive_drive.config import AppConfig
from interactive_drive.math3d import rig_pose_from_state
from interactive_drive.ros_drive import (
    driver_command,
    finish_step,
    publish_frame,
    publish_initial_observation,
    report_model_errors,
)
from interactive_drive.ros_endpoint import DriveEndpoint
from interactive_drive.simulation.ego_vehicle_kinematics import integrate_vehicle
from interactive_drive.types import VehicleState


def serve(endpoint, stop):
    config = AppConfig(scene_path=Path("."), physics=False)
    interval_us = 100_000
    dt = interval_us / 1_000_000
    while request := endpoint.receive(stop.is_set):
        episode = request["episode_id"]
        if request["type"] == "reset":
            state = VehicleState(
                x_m=0.0, y_m=0.0, z_m=0.0, yaw_rad=0.0, speed_mps=0.0, steer_rad=0.0
            )
            timestamp = 0
            publish_initial_observation(
                endpoint, episode, timestamp, rig_pose_from_state(0, 0, 0, 0), state
            )
            continue
        # DriveEndpoint admits a step only after a successful reset.
        try:
            command = driver_command(request, config)
        except ValueError as exc:
            endpoint.send(
                {"type": "error", "code": "INVALID_REQUEST", "message": str(exc)}
            )
            continue
        with report_model_errors(endpoint):
            started = monotonic()
            state = integrate_vehicle(state, command, dt, config.vehicle)
            pose = rig_pose_from_state(
                state.x_m,
                state.y_m,
                state.z_m,
                state.yaw_rad,
                state.pitch_rad + state.suspension_pitch_rad,
                state.roll_rad + state.suspension_roll_rad,
            )
            # Cap this CPU trial at real time, including after input resumes.
            if stop.wait(max(0.0, dt - (monotonic() - started))):
                return
            previous = timestamp
            timestamp += interval_us
            publish_frame(
                endpoint, episode, request["expected_step"], timestamp, pose, state
            )
            finish_step(endpoint, request, command, previous, timestamp, timestamp, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--socket",
        type=Path,
        required=True,
        help="Same Unix socket path as the ROS bridge",
    )
    args = parser.parse_args()
    stop = Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    with closing(DriveEndpoint(args.socket)) as endpoint:
        print(f"Flat-ground vehicle dynamics ready on {args.socket}", flush=True)
        serve(endpoint, stop)


if __name__ == "__main__":
    main()
