# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors

"""Receive the remote cockpit's Foxglove WebSocket controls on ROS 2 /joy."""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp")
    socket_path = runtime_dir / f"flashdreams-{os.getuid()}" / "drive.sock"
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "address",
                default_value="127.0.0.1",
                description="WebSocket listen address; use the simulation host's LAN IP for direct access.",
            ),
            DeclareLaunchArgument(
                "port", default_value="8765", description="WebSocket TCP port."
            ),
            DeclareLaunchArgument(
                "model_log_unit", default_value="flashdreams-ros-model.service",
                description="User systemd unit whose model logs appear on /rosout (empty disables).",
            ),
            DeclareLaunchArgument(
                "socket_path",
                default_value=str(socket_path),
                description="The model's --ros-socket path.",
            ),
            Node(
                package="flashdreams_bridge",
                executable="bridge",
                output="screen",
                parameters=[
                    {
                        "control_mode": "joy",
                        "model_log_unit": ParameterValue(LaunchConfiguration("model_log_unit"), value_type=str),
                        "socket_path": ParameterValue(
                            LaunchConfiguration("socket_path"), value_type=str
                        ),
                    }
                ],
            ),
            Node(
                package="flashdreams_bridge",
                executable="camera_preview",
                output="screen",
            ),
            Node(
                package="foxglove_bridge",
                executable="foxglove_bridge",
                output="screen",
                parameters=[
                    {
                        "address": ParameterValue(
                            LaunchConfiguration("address"), value_type=str
                        ),
                        "port": ParameterValue(
                            LaunchConfiguration("port"), value_type=int
                        ),
                        "capabilities": ["clientPublish"],
                        "client_topic_whitelist": ["^/joy$"],
                        "topic_whitelist": [
                            "^/joy$",
                            "^/sim/.*$",
                            "^/tf$",
                            "^/tf_static$",
                            "^/clock$",
                            "^/rosout$",
                            "^/diagnostics$",
                        ],
                        "sysinfo": False,
                    }
                ],
            ),
        ]
    )
