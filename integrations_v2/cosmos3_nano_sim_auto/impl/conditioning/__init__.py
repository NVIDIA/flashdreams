# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Everything the model is told: the prompt, the map, and where the car is.

Control reaches the rollout through :mod:`.control_provider`, which is the one
seam here: it does not care whether a span of map was read from a video or
drawn by :mod:`.ludus_controls` as the run went.
"""
