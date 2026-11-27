# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 FlashDreams contributors

# Original F-theta calibration at its native resolution. TF supplies extrinsics.
std_msgs/Header header
uint32 calibration_width
uint32 calibration_height
uint32 image_width
uint32 image_height
float64 cx
float64 cy
float64[] polynomial
bool is_backward_polynomial
float64[3] linear_cde
