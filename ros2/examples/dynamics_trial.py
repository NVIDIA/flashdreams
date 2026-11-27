# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the ROS manual-control path on flat ground without a video model."""

import argparse
import math
from pathlib import Path

import numpy as np
from interactive_drive.config import AppConfig
from interactive_drive.ros_drive import driver_command
from interactive_drive.simulation.ego_vehicle_kinematics import EgoVehicleKinematics
from interactive_drive.types import VehicleState

# Segment, observation count at 10 Hz, selected gear, front-wheel rad, pedals.
PHASES = (
    ("Straight", 40, "D", 0.0, 0.65, 0.0),
    ("Left turn", 32, "D", 0.24, 0.25, 0.0),
    ("Brake", 15, "D", 0.0, 0.0, 0.7),
    ("Reverse", 28, "R", -0.18, 0.6, 0.0),
    ("Brake", 10, "R", 0.0, 0.0, 0.7),
    ("Park", 6, "P", 0.0, 0.0, 0.0),
)


def trial():
    config = AppConfig(scene_path=Path("."))
    sim = EgoVehicleKinematics(VehicleState(0, 0, 0, 0, 0, 0), config.vehicle, None, 0)
    segments = []
    for name, frames, gear, angle, throttle, brake in PHASES:
        command = driver_command(
            dict(
                manual_gear=gear,
                reverse=gear == "R",
                steering_angle_rad=angle,
                throttle=throttle,
                brake=brake,
            ),
            config,
        )
        chunk = sim.pose_chunk(command, frames, 0.1, 0.0)
        end = chunk.boundary_state_after_chunk
        states = (*chunk.vehicle_states, end)
        start = states[0]
        times = np.append(chunk.timestamps_us, chunk.timestamps_us[-1] + 100_000) / 1e6
        speeds = np.array([s.speed_mps for s in states])
        dx, dy = end.x_m - start.x_m, end.y_m - start.y_m
        assert all(math.isclose(s.steer_rad, angle) for s in states[1:])
        if name == "Straight":
            assert np.all(np.diff(speeds) > 0) and dx > 0
            assert abs(dy) < 1e-6 and abs(end.yaw_rad) < 1e-6
        elif name == "Left turn":
            assert end.yaw_rad > start.yaw_rad and dy > 0 and np.all(speeds > 0)
        elif name == "Brake":
            assert np.all(np.diff(abs(speeds)) <= 0) and end.speed_mps == 0.0
            assert np.all(speeds * start.speed_mps >= 0)
        elif name == "Reverse":
            assert start.speed_mps == 0 and np.all(np.diff(speeds) < 0)
            assert dx * math.cos(start.yaw_rad) + dy * math.sin(start.yaw_rad) < 0
        elif name == "Park":
            assert np.all(speeds == 0) and math.hypot(dx, dy) < 1e-4
        print(
            f"{times[-1]:5.1f}s  {name:9}  {gear}  wheel={end.steer_rad:+.2f} rad  "
            f"speed={end.speed_mps:+.2f} m/s  xy=({end.x_m:.2f}, {end.y_m:.2f}) m  "
            f"heading={math.degrees(end.yaw_rad):.1f} deg"
        )
        segments.append((name, gear, times, states))
    return segments


def plot(segments, output):
    import matplotlib.pyplot as plt

    colors = {
        "Straight": "#185991",
        "Left turn": "#12756c",
        "Brake": "#a34b17",
        "Reverse": "#983c72",
        "Park": "#5e6776",
    }
    fig = plt.figure(figsize=(11, 5.1), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
    path = fig.add_subplot(grid[:, 0], aspect="equal")
    speed = fig.add_subplot(grid[0, 1])
    steer = fig.add_subplot(grid[1, 1], sharex=speed)
    for name, gear, times, states in segments:
        color = colors[name]
        path.plot(
            [s.x_m for s in states],
            [s.y_m for s in states],
            lw=3.2,
            color=color,
            label=None if (name, gear) == ("Brake", "R") else name,
        )
        speed.plot(times, [s.speed_mps for s in states], color=color, lw=2.5)
        steer.plot(times, [s.steer_rad for s in states], color=color, lw=2.5)
        if name in ("Straight", "Left turn", "Reverse"):
            s = states[len(states) // 2]
            path.quiver(
                s.x_m,
                s.y_m,
                math.cos(s.yaw_rad),
                math.sin(s.yaw_rad),
                color="#263240",
                scale=13,
                width=0.008,
                zorder=4,
            )
    origin = segments[0][-1][0]
    final = segments[-1][-1][-1]
    path.scatter([origin.x_m], [origin.y_m], s=52, color="#185991", zorder=5)
    path.scatter(
        [final.x_m], [final.y_m], s=62, marker="s", color=colors["Park"], zorder=5
    )
    path.set(
        xlabel="Forward at start (m)",
        ylabel="Left at start (m)",
        title="Trajectory and vehicle heading",
    )
    path.legend(loc="upper left", fontsize=9, frameon=False)
    speed.set(title="Longitudinal speed", ylabel="m/s")
    steer.set(title="Front-wheel angle", ylabel="rad", xlabel="Simulation time (s)")
    for ax in (path, speed, steer):
        ax.set_axisbelow(True)
        ax.grid(color="#dce1e6", lw=0.7)
        ax.spines[["right", "top"]].set_visible(False)
    for gear in ("D", "R", "P"):
        times = [t for _, selected, t, _ in segments if selected == gear]
        speed.text(
            (times[0][0] + times[-1][-1]) / 2,
            0.94,
            gear,
            ha="center",
            va="top",
            transform=speed.get_xaxis_transform(),
            color="#344455",
        )
        if gear != "D":
            for ax in (speed, steer):
                ax.axvline(times[0][0], color="#84909c", lw=0.9, ls=":")
    fig.savefig(output, dpi=165)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot", type=Path, help="Write a trajectory and telemetry plot"
    )
    options = parser.parse_args()
    result = trial()
    if options.plot:
        plot(result, options.plot)
