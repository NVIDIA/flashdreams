# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# ``python -m interactive_drive`` and the ``interactive-drive`` console
# script both go through the demo wrapper so the same flags work in both
# the supervised HUD path and the bare backend path. The HUD is on by
# default; pass ``--no-hud`` (or ``--stream-mjpeg HOST:PORT``) to bypass
# it and fall through to the original cli backend.
from interactive_drive.demo import main

if __name__ == "__main__":
    main()
