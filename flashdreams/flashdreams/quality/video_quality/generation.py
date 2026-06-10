# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generator seam: turn a scenario's conditioning into a clip to evaluate.

The runner calls a :class:`Generator` to produce the RGB clip it will score,
replacing #317's evaluate-only ``NotImplementedError``. The real backends (gRPC
replay against the omnidreams world-model server, or direct in-process
``pipeline.generate``) land later; the skeleton ships a deterministic synthetic
passthrough so the pipeline runs end-to-end without a GPU or model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from flashdreams.quality.video_quality.manifest import VideoQualityCase
from flashdreams.quality.video_quality.metrics import VideoMetricsInput, synthetic_video


@dataclass(frozen=True)
class GenerationRequest:
    """Everything a generator needs to produce one scenario's clip."""

    scenario_id: str
    conditioning: dict[str, Any] = field(default_factory=dict)
    source: dict[str, Any] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Generator(Protocol):
    """Pluggable clip generator. Real backends (gRPC / in-process) land later."""

    name: str

    def generate(self, request: GenerationRequest) -> VideoMetricsInput:
        """Produce an RGB clip for ``request``."""
        ...


class SyntheticPassthroughGenerator:
    """Placeholder generator: deterministic synthetic clip, no model/GPU."""

    name = "synthetic_passthrough"

    def generate(self, request: GenerationRequest) -> VideoMetricsInput:
        source = request.source
        pattern = str(
            request.conditioning.get("pattern")
            or source.get("pattern")
            or "textured_motion"
        )
        return synthetic_video(
            pattern,
            frames=int(source.get("frames", 16)),
            height=int(source.get("height", 64)),
            width=int(source.get("width", 64)),
            fps=float(source.get("fps", 8)),
            seed=int(source.get("seed", request.generation.get("seed", 0))),
        )


class OmnidreamsGenerator:
    """Real batch HD-map -> RGB generator via the in-process ``flashdreams-run``
    OmnidreamsRunner path (no gRPC server needed).

    GPU-only: needs CUDA, the ``flashdreams-omnidreams`` package, and the gated
    HF checkpoints (``nvidia/omni-dreams-models``) + sample data. It is registered
    so the benchmark can select it with ``--generator omnidreams`` on the GPU
    pool, while ``synthetic_passthrough`` stays the default for CPU/CI runs.

    Asset convention (per scenario): ``assets.conditioning`` is the HD-map video
    path and ``assets.first_frame`` is the seed frame. With neither set, the demo
    ``example_data`` path lazy-fetches a bundled clip from Hugging Face.

    Layering note: omnidreams is an integration package, so it is imported lazily;
    this backend could later move behind an entry-point plugin.
    """

    name = "omnidreams"
    default_recipe = "omnidreams-sv-2steps-chunk2-loc6-lightvae-lighttae-perf"

    def __init__(self, *, recipe: str | None = None, total_blocks: int = 20) -> None:
        self._recipe = recipe or self.default_recipe
        self._total_blocks = total_blocks

    def generate(self, request: GenerationRequest) -> VideoMetricsInput:
        import dataclasses  # noqa: PLC0415
        import tempfile  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        import mediapy as media  # noqa: PLC0415

        from omnidreams.config import OMNIDREAMS_RUNNERS  # noqa: PLC0415

        recipe = str(request.generation.get("recipe") or self._recipe)
        try:
            base = OMNIDREAMS_RUNNERS[recipe]
        except KeyError as exc:
            raise KeyError(
                f"Unknown omnidreams recipe {recipe!r}; registered: "
                f"{tuple(sorted(OMNIDREAMS_RUNNERS))}"
            ) from exc

        total_blocks = int(request.generation.get("total_blocks", self._total_blocks))
        output_dir = Path(tempfile.mkdtemp(prefix="omnidreams_gen_"))
        overrides: dict[str, object] = {
            "output_dir": output_dir,
            "total_blocks": total_blocks,
        }

        hdmap = request.conditioning.get("conditioning")
        first_frame = request.conditioning.get("first_frame")
        if hdmap:
            if not first_frame:
                raise ValueError(
                    f"{request.scenario_id}: omnidreams generator needs a first_frame "
                    "asset alongside the HD-map conditioning video."
                )
            overrides["hdmap_video_paths"] = (Path(hdmap),)
            overrides["first_frame_paths"] = (Path(first_frame),)
        else:
            overrides["example_data"] = True
            uuid = request.generation.get("example_data_uuid")
            if uuid:
                overrides["example_data_uuid"] = str(uuid)

        prompt = request.conditioning.get("prompt")
        if prompt:
            overrides["prompt"] = str(prompt)

        cfg = dataclasses.replace(base, **overrides)
        runner = cfg.setup()
        runner.run()

        canvas = media.read_video(str(output_dir / f"{recipe}.mp4"))
        # The runner stacks [HD-map (top), generated (bottom)] vertically; the
        # generated RGB is the lower ``pixel_height`` rows.
        generated = np.ascontiguousarray(canvas[:, cfg.pixel_height :, :, :3])
        return VideoMetricsInput(frames=generated, fps=float(cfg.output_fps))


_GENERATORS: dict[str, Generator] = {
    generator.name: generator
    for generator in (SyntheticPassthroughGenerator(), OmnidreamsGenerator())
}


def get_generator(name: str) -> Generator:
    """Return a registered generator backend."""
    try:
        return _GENERATORS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown generator {name!r}; registered: {tuple(sorted(_GENERATORS))}"
        ) from exc


def request_from_case(case: VideoQualityCase) -> GenerationRequest:
    """Build a generation request from a manifest case's conditioning."""
    conditioning = {
        "prompt": case.assets.prompt,
        "first_frame": case.assets.first_frame,
        "conditioning": case.assets.conditioning,
    }
    return GenerationRequest(
        scenario_id=case.id,
        conditioning={k: v for k, v in conditioning.items() if v is not None},
        source=dict(case.source),
        generation=dict(case.generation),
    )
