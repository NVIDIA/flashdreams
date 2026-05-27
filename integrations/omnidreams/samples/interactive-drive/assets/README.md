# Assets

This directory contains repo-local binary assets needed to run the self-contained demo.

## Scenes

Staged under `scenes/` by `interactive-drive-prepare`, which pulls the USDZ
from [`nvidia/omni-dreams-scenes`](https://huggingface.co/datasets/nvidia/omni-dreams-scenes)
on Hugging Face. The default scene is
`scenes/clipgt-01d503d4-449b-46fc-8d78-9085e70d3554.usdz` and auto-stages
on first `interactive-drive` launch; run
`uv run --package omnidreams-interactive-drive interactive-drive-prepare --scene-uuid <clipgt-...>`
to stage a different scene from the dataset.

The default GUI launch configs and README examples use the default scene
so the demo no longer depends on a scene path outside the current workspace.
