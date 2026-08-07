# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Interactive driving demo, built on the shared local-window presenter.

Holds the demo's chrome today: driving widgets stacked as
:class:`~flashdreams.serving.presentation.HudOverlay` layers over the shared
presenter. The engine -- loop, simulation, scene loading -- still lives in
``omnidreams.interactive_drive`` and moves here as it stops being
model-specific.
"""
