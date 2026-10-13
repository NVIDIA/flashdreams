# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Configuration for rendering HD maps.
"""

# Settings aligned with dataset_rds_hq_mv.json
# From https://github.com/nv-tlabs/Cosmos-Drive-Dreams/tree/main/cosmos-drive-dreams-toolkits
SETTINGS = {
    "INPUT_POSE_FPS": 30,  # Target pose FPS after interpolation
    "INPUT_LIDAR_FPS": 10,
    "GT_VIDEO_FPS": 30,
    "SOURCE_POSE_FPS": 10,  # Original ego pose FPS in clipGT
    "SOURCE_VIDEO_FPS": 30,  # Original video FPS in clipGT
    "SOURCE_OBSTACLE_FPS": 10,  # Original obstacle FPS in clipGT
    "COSMOS_RESOLUTION": [720, 1280],
    "RESIZE_RESOLUTION": [720, 1280],
    "TARGET_CHUNK_FRAME": 121,
    "OVERLAP_FRAME": 0,
    "TARGET_RENDER_FPS": 30,
    "MAX_CHUNK": 10,
}
