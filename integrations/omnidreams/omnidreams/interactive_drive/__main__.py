# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# ``python -m omnidreams.interactive_drive`` and the ``interactive-drive`` console
# script both go through the demo wrapper so the same flags work in both
# the supervised HUD path and the bare backend path. The HUD is on by
# default; pass ``--no-hud`` to bypass it and fall through to the bare
# slangpy Vulkan window. Browser / remote streaming use cases live in
# the separate ``omnidreams.webrtc.server`` entry point.
from omnidreams.interactive_drive.demo import main

if __name__ == "__main__":
    main()
