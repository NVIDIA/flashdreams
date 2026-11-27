# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors

# Sampled at the same simulation time as odometry and images.
std_msgs/Header header
float64 speed_mps          # Signed longitudinal speed, positive forward.
float64 steering_angle_rad # Front wheel angle, positive left.
float64 yaw_rate_radps     # Heading rate, positive counterclockwise.
# Settings used to integrate up to this sample, after the gear interlocks.
string gear                # P, R, N, D.
float64 throttle           # Pedal fraction [0, 1].
float64 brake              # Pedal fraction [0, 1].
