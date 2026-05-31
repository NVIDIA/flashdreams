#!/usr/bin/env bash
set -euo pipefail

## only needed once for steering wheel configuration
# uv run --package flashdreams-omnidreams interactive-drive-configure-wheel

uv sync --package flashdreams-omnidreams --extra interactive-drive
uv run --package flashdreams-omnidreams python integrations/omnidreams/omnidreams_singleview/tools/sync_thirdparty.py sync

INTERACTIVE_DRIVE_PROFILE_INPUT_TO_PRESENT=1 \
uv run --no-sync --package flashdreams-omnidreams interactive-drive --autoload-scene --manifest example_world_model_perf.yaml \
   --scene  $HOME/cvpr_scenes/clipgt-e2993759-36e1-4d97-868f-e2a737f1eb68.usdz

# clipgt-e2993759-36e1-4d97-868f-e2a737f1eb68.usdz
# clipgt-7bd1eb2f-c375-44ee-b4ca-55473e0773a9.usdz, night scene
