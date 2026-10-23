# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The model's own implementation, below the two front ends.

Nothing outside this package should need a name from here. The two things that
are meant to be reached from elsewhere are the registered applications in
:mod:`cosmos3_nano_sim_auto.apps` and the settings in :mod:`cosmos3_nano_sim_auto.config`,
and what is under here is what they are built out of.
"""
