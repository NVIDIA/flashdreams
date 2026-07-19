# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reference template integration runner for ``flashdreams-run``.

The template integration ships toy ``Conv3d`` networks and a synthetic
control input -- no prompt, no first frame; outputs are diagnostic
tensors. New runners should mirror this control flow:

1. ``Runner.__init__`` is inherited (eagerly builds the pipeline).
2. :meth:`TemplateRunner.run` resolves runner-config inputs, calls
   ``self.pipeline.initialize_cache(...)``, loops ``generate`` +
   ``finalize``, and persists outputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from loguru import logger

from flashdreams.infra.pipeline import StreamInferencePipeline
from flashdreams.infra.runner import Runner, RunnerConfig
from flashdreams.infra.runner_io import ensure_output_dir, runner_artifact_path
from flashdreams.recipes.template.encoder import TemplateControlEncoder
from flashdreams.recipes.template.transformer import (
    TemplateTransformer,
    TemplateTransformerConfig,
)


@dataclass(kw_only=True)
class TemplateRunnerConfig(RunnerConfig):
    """Runner config for any template variant (offline / AR / AR-compiled)."""

    _target: type["TemplateRunner"] = field(default_factory=lambda: TemplateRunner)

    num_ar_steps: int = 1
    """How many AR steps to roll. Pin per-variant: 1 for the offline /
    bidirectional preset (one chunk covers ``window_size_t``), 2+ for
    the streaming AR presets (so the KV cache exercises both the
    filling and the steady-state code paths)."""

    height: int = 6
    """Pre-patchify latent height. Must be divisible by
    ``transformer.patch_size[1]``. Defaults match the smoke-test
    fixture in ``tests/test_template.py``."""

    width: int = 4
    """Pre-patchify latent width. Same divisibility rule as
    :attr:`height`."""

    batch_size: int = 1
    """Batch elements. The template DiT has no batch-shape constraint."""

    n_context_tokens: int = 4
    """Length of the synthetic context-token sequence fed to
    ``context_encoder``."""

    seed: int = 42
    """Seeds a :class:`torch.Generator` so two runs with the same
    config produce bit-identical context + control. The pipeline's
    own ``diffusion_model.seed`` controls the noise sample."""


class TemplateRunner(Runner[TemplateRunnerConfig, StreamInferencePipeline]):
    """End-to-end driver for any template-integration variant."""

    def run(self) -> None:
        """Roll one rollout and dump the output tensor."""
        cfg = self.config
        transformer = self.pipeline.diffusion_model.transformer
        assert isinstance(transformer, TemplateTransformer), (
            f"TemplateRunner expected TemplateTransformer; "
            f"got {type(transformer).__name__}."
        )
        tcfg: TemplateTransformerConfig = transformer.config
        device = torch.device(cfg.device)

        # Read control_channels off the live encoder so encoder
        # overrides via ``derive_config`` flow through. ``encoder=None``
        # is the no-control case; the 8 here is just a placeholder that
        # the ``control = None`` short-circuit below discards.
        if isinstance(self.pipeline.encoder, TemplateControlEncoder):
            control_channels = self.pipeline.encoder.config.control_channels
        else:
            control_channels = 8

        inputs = _make_synthetic_inputs(
            tcfg=tcfg,
            batch_size=cfg.batch_size,
            height=cfg.height,
            width=cfg.width,
            n_context_tokens=cfg.n_context_tokens,
            control_channels=control_channels,
            device=device,
            seed=cfg.seed,
        )

        transformer_context: dict[str, object] = {
            "context": inputs["context"],
            "height": cfg.height,
            "width": cfg.width,
        }
        if tcfg.guidance_scale > 1.0:
            transformer_context["negative_context"] = inputs["negative_context"]

        cache = self.pipeline.initialize_cache(transformer_context=transformer_context)

        outputs: list[torch.Tensor] = []
        # Pipeline asserts encoder presence ⇔ control input; mirror the
        # branch here so the AR loop hands a coherent ``input`` per step.
        control = inputs["control"] if self.pipeline.encoder is not None else None
        for ar_idx in range(cfg.num_ar_steps):
            out = self.pipeline.generate(ar_idx, cache, input=control)
            outputs.append(out)
            # Skip ``finalize`` on the last step (canonical pattern).
            if ar_idx < cfg.num_ar_steps - 1:
                self.pipeline.finalize(ar_idx, cache)

        # Persist only on rank 0; under CP every rank holds the same
        # gathered output. Force CPU before save so the pickle is portable.
        if not self.is_rank_zero:
            return
        stacked = torch.stack(outputs, dim=0).cpu()
        ensure_output_dir(cfg.output_dir)
        out_path = runner_artifact_path(cfg.output_dir, cfg.runner_name, "pt")
        torch.save(stacked, out_path)

        logger.info(
            f"[{cfg.runner_name}] wrote {tuple(stacked.shape)} "
            f"({stacked.dtype}) to {out_path.resolve()}"
        )


def _make_synthetic_inputs(
    *,
    tcfg: TemplateTransformerConfig,
    batch_size: int,
    height: int,
    width: int,
    n_context_tokens: int,
    control_channels: int,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Build deterministic random context + control tensors.

    Uses the same seeded generator path as
    ``tests/test_template.py::_make_inputs`` so equal seeds reproduce
    the smoke test's input distribution.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    dtype = tcfg.dtype
    return dict(
        context=torch.randn(
            batch_size,
            n_context_tokens,
            tcfg.network.context_channels,
            device=device,
            generator=gen,
            dtype=dtype,
        ),
        negative_context=torch.randn(
            batch_size,
            n_context_tokens,
            tcfg.network.context_channels,
            device=device,
            generator=gen,
            dtype=dtype,
        ),
        control=torch.randn(
            batch_size,
            control_channels,
            tcfg.len_t,
            height,
            width,
            device=device,
            generator=gen,
            dtype=dtype,
        ),
    )


__all__ = [
    "TemplateRunner",
    "TemplateRunnerConfig",
]
