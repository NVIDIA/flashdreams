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

"""End-to-end smoke tests + manual driver for the ``template`` recipe.

``test_template_cp_equivalence`` is a two-invocation test: the
single-GPU branch writes a reference tensor to
``$TEMPLATE_CP_REF_PATH`` (default ``<tmpdir>/flashdreams/cp_reference.pt``),
then a torchrun launch reads it and asserts equivalence.

.. code-block:: bash

    uv run --extra dev pytest \
        flashdreams/tests/test_template.py::test_template_cp_equivalence -v
    uv run --extra dev torchrun --nproc_per_node=2 -m pytest \
        flashdreams/tests/test_template.py::test_template_cp_equivalence -v
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from flashdreams.core.distributed.context_parallel import cat_outputs_cp
from flashdreams.infra.config import derive_config
from flashdreams.infra.pipeline import (
    StreamInferencePipeline,
    StreamInferencePipelineConfig,
)
from flashdreams.recipes.template.config import (
    build_cfg_autoregressive,
    build_cfg_offline,
    with_compile_and_cuda_graph,
)
from flashdreams.recipes.template.transformer import (
    TemplateTransformer,
    TemplateTransformerCache,
    TemplateTransformerConfig,
)

# Asymmetric ``H != W != len_t`` to catch transpose / reshape bugs
# a square shape would hide. ``H`` / ``W`` must be divisible by
# ``TemplateTransformerConfig.patch_size``.
TEMPLATE_DEFAULT_BATCH_SIZE = 1
TEMPLATE_DEFAULT_HEIGHT = 6
TEMPLATE_DEFAULT_WIDTH = 4

_CP_REFERENCE_PATH = Path(
    os.environ.get(
        "TEMPLATE_CP_REF_PATH",
        str(Path(tempfile.gettempdir()) / "flashdreams" / "cp_reference.pt"),
    )
)
"""Shared reference tensor for :func:`test_template_cp_equivalence`.

Set ``TEMPLATE_CP_REF_PATH`` to a shared-FS path when the two
invocations don't share ``/tmp``."""


## Shared helpers


def _make_inputs(
    cfg: TemplateTransformerConfig,
    *,
    batch_size: int,
    height: int,
    width: int,
    control_channels: int,
    context_channels: int,
    n_context_tokens: int,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Build deterministic per-rollout inputs for one rank.

    Uses a seeded :class:`torch.Generator` so repeated calls with the
    same ``seed`` are bit-identical — required for the eager-vs-compiled
    and CP-equivalence checks.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    dtype = cfg.dtype
    return dict(
        context_embeddings=torch.randn(
            batch_size,
            n_context_tokens,
            context_channels,
            device=device,
            generator=gen,
            dtype=dtype,
        ),
        negative_context_embeddings=torch.randn(
            batch_size,
            n_context_tokens,
            context_channels,
            device=device,
            generator=gen,
            dtype=dtype,
        ),
        control_pre_patchify=torch.randn(
            batch_size,
            control_channels,
            cfg.len_t,
            height,
            width,
            device=device,
            generator=gen,
            dtype=dtype,
        ),
    )


def _run_rollout(
    *,
    pipeline: StreamInferencePipeline,
    num_ar_steps: int,
    use_cfg: bool,
    inputs: dict[str, torch.Tensor],
    height: int,
    width: int,
    use_control: bool = True,
) -> list[torch.Tensor]:
    """Initialize the cache, run ``num_ar_steps``, and return per-step outputs.

    Args:
        use_control: When ``False``, forwards ``input=None`` to
            ``pipeline.generate``. The pipeline must be built with
            ``encoder=None`` or ``generate`` asserts.
    """
    transformer_context: dict[str, object] = {
        "context": inputs["context_embeddings"],
        "height": height,
        "width": width,
    }
    if use_cfg:
        transformer_context["negative_context"] = inputs["negative_context_embeddings"]

    cache = pipeline.initialize_cache(transformer_context=transformer_context)
    assert isinstance(cache.transformer_cache, TemplateTransformerCache)
    if use_cfg:
        assert cache.transformer_cache.network_cache_uncond is not None
    else:
        assert cache.transformer_cache.network_cache_uncond is None

    control = inputs["control_pre_patchify"] if use_control else None
    outputs: list[torch.Tensor] = []
    for ar_idx in range(num_ar_steps):
        out = pipeline.generate(ar_idx, cache, input=control)
        outputs.append(out)
        # Skip ``finalize`` on the last step — the canonical "hot loop,
        # optional last finalize" pattern.
        if ar_idx < num_ar_steps - 1:
            pipeline.finalize(ar_idx, cache)
    return outputs


## Basic smoke tests

# ``RingAttention`` dispatches to CUDA-only SDPA backends.
_CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="template recipe uses RingAttention which requires CUDA.",
)


@_CUDA_REQUIRED
@pytest.mark.parametrize("seed", [0, 42])
def test_template_bidirectional(seed: int) -> None:
    """``build_cfg_offline`` runs a single AR step end-to-end."""
    config = build_cfg_offline(seed=seed)
    pipeline = config.setup()
    assert isinstance(pipeline, StreamInferencePipeline)
    transformer = pipeline.diffusion_model.transformer
    assert isinstance(transformer, TemplateTransformer)
    cfg = transformer.config

    device = torch.device("cuda")
    pipeline.to(device).eval()

    H, W = TEMPLATE_DEFAULT_HEIGHT, TEMPLATE_DEFAULT_WIDTH
    B = TEMPLATE_DEFAULT_BATCH_SIZE
    inputs = _make_inputs(
        cfg,
        batch_size=B,
        height=H,
        width=W,
        control_channels=8,
        context_channels=16,
        n_context_tokens=4,
        device=device,
        seed=seed,
    )
    outputs = _run_rollout(
        pipeline=pipeline,
        num_ar_steps=1,
        use_cfg=False,
        inputs=inputs,
        height=H,
        width=W,
    )

    assert len(outputs) == 1
    assert outputs[0].shape == (B, 3, cfg.len_t, H, W)
    assert outputs[0].device.type == device.type
    assert torch.isfinite(outputs[0]).all()


@_CUDA_REQUIRED
def test_template_no_control() -> None:
    """Rollout with no encoder + ``input=None`` skips the control bias.

    Covers the ``TemplateDiT.forward`` branch that skips
    ``self.input_proj(control)`` when no control tensor is supplied,
    and asserts the no-control output disagrees with the matching
    control-on rollout.
    """
    from typing import cast

    base = build_cfg_offline(seed=0)
    config = cast(StreamInferencePipelineConfig, derive_config(base, encoder=None))
    # Seed before each ``setup()`` so the no-control and with-control
    # pipelines initialise identical weights — the assertion below only
    # isolates the control branch when the underlying networks match
    # bit-for-bit.
    torch.manual_seed(0)
    pipeline = config.setup()
    assert isinstance(pipeline, StreamInferencePipeline)
    assert pipeline.encoder is None
    transformer = pipeline.diffusion_model.transformer
    assert isinstance(transformer, TemplateTransformer)
    cfg = transformer.config

    device = torch.device("cuda")
    pipeline.to(device).eval()

    H, W = TEMPLATE_DEFAULT_HEIGHT, TEMPLATE_DEFAULT_WIDTH
    B = TEMPLATE_DEFAULT_BATCH_SIZE
    inputs = _make_inputs(
        cfg,
        batch_size=B,
        height=H,
        width=W,
        control_channels=8,
        context_channels=16,
        n_context_tokens=4,
        device=device,
        seed=0,
    )
    outputs_no_ctrl = _run_rollout(
        pipeline=pipeline,
        num_ar_steps=1,
        use_cfg=False,
        inputs=inputs,
        height=H,
        width=W,
        use_control=False,
    )
    assert len(outputs_no_ctrl) == 1
    assert outputs_no_ctrl[0].shape == (B, 3, cfg.len_t, H, W)
    assert torch.isfinite(outputs_no_ctrl[0]).all()

    # A matching control-on rollout (same seed, same inputs) must
    # differ — otherwise the control bias is silently dropped.
    torch.manual_seed(0)
    with_ctrl_pipeline = build_cfg_offline(seed=0).setup().to(device).eval()
    outputs_with_ctrl = _run_rollout(
        pipeline=with_ctrl_pipeline,
        num_ar_steps=1,
        use_cfg=False,
        inputs=inputs,
        height=H,
        width=W,
        use_control=True,
    )
    assert not torch.allclose(
        outputs_no_ctrl[0], outputs_with_ctrl[0], rtol=1e-2, atol=1e-2
    ), "control-off and control-on rollouts produced identical outputs"


@_CUDA_REQUIRED
def test_template_streaming_cfg() -> None:
    """``build_cfg_autoregressive`` runs multi-step AR with CFG on."""
    config = _with_cfg(build_cfg_autoregressive(seed=0), guidance_scale=2.0)
    pipeline = config.setup()
    transformer = pipeline.diffusion_model.transformer
    assert isinstance(transformer, TemplateTransformer)
    cfg = transformer.config
    # Exercise both the filling and steady-state KV-cache phases.
    num_ar_steps = (cfg.sink_size_t + cfg.window_size_t) // cfg.len_t + 2
    device = torch.device("cuda")
    pipeline.to(device).eval()

    H, W = TEMPLATE_DEFAULT_HEIGHT, TEMPLATE_DEFAULT_WIDTH
    B = TEMPLATE_DEFAULT_BATCH_SIZE
    inputs = _make_inputs(
        cfg,
        batch_size=B,
        height=H,
        width=W,
        control_channels=8,
        context_channels=16,
        n_context_tokens=4,
        device=device,
        seed=0,
    )
    outputs = _run_rollout(
        pipeline=pipeline,
        num_ar_steps=num_ar_steps,
        use_cfg=True,
        inputs=inputs,
        height=H,
        width=W,
    )

    assert len(outputs) == num_ar_steps
    for out in outputs:
        assert out.shape == (B, 3, cfg.len_t, H, W)
        assert torch.isfinite(out).all()
    assert not torch.allclose(outputs[0], outputs[1])


@_CUDA_REQUIRED
def test_template_cfg_rejects_missing_negative_context() -> None:
    """CFG on without negative context must fail at cache build time."""
    config = _with_cfg(build_cfg_autoregressive(seed=0), guidance_scale=2.0)
    pipeline = config.setup()
    pipeline.to("cuda").eval()
    transformer = pipeline.diffusion_model.transformer
    assert isinstance(transformer, TemplateTransformer)

    B = TEMPLATE_DEFAULT_BATCH_SIZE
    dtype = transformer.config.dtype
    context = torch.randn(B, 4, 16, device="cuda", dtype=dtype)
    with pytest.raises(AssertionError, match="negative_context_embeddings"):
        pipeline.initialize_cache(
            transformer_context={
                "context": context,
                "height": TEMPLATE_DEFAULT_HEIGHT,
                "width": TEMPLATE_DEFAULT_WIDTH,
            }
        )


@_CUDA_REQUIRED
def test_template_latent_shape_respects_cp_size() -> None:
    """``latent_shape`` reflects the per-rollout, per-rank ``L`` partition.

    Populated lazily by
    :meth:`TemplateTransformer.initialize_autoregressive_cache`;
    reading it earlier asserts.
    """
    device = torch.device("cuda")
    pipeline = build_cfg_offline().setup().to(device).eval()
    transformer = pipeline.diffusion_model.transformer
    assert isinstance(transformer, TemplateTransformer)
    cfg = transformer.config
    assert isinstance(cfg, TemplateTransformerConfig)

    with pytest.raises(AssertionError, match="initialize_autoregressive_cache"):
        _ = transformer.latent_shape

    H, W = TEMPLATE_DEFAULT_HEIGHT, TEMPLATE_DEFAULT_WIDTH
    B = TEMPLATE_DEFAULT_BATCH_SIZE
    context = torch.randn(B, 4, 16, device=device, dtype=cfg.dtype)
    pipeline.initialize_cache(
        transformer_context={"context": context, "height": H, "width": W}
    )
    # cp_size == 1 here; torch.distributed isn't initialised.
    kt, kh, kw = cfg.patch_size
    L = (cfg.len_t // kt) * (H // kh) * (W // kw)
    expected = (B, L // transformer._cp_size, cfg.network.in_channels)
    assert transformer.latent_shape == expected


## torch.compile + CUDAGraphWrapper equivalence


def _with_cfg(
    base: StreamInferencePipelineConfig, *, guidance_scale: float
) -> StreamInferencePipelineConfig:
    """Return ``base`` with ``guidance_scale`` patched onto the transformer."""
    from typing import cast

    return cast(
        StreamInferencePipelineConfig,
        derive_config(
            base,
            diffusion_model=dict(
                transformer=dict(guidance_scale=guidance_scale),
            ),
        ),
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="torch.compile + CUDAGraphWrapper equivalence requires CUDA.",
)
def test_template_compile_and_cudagraph_equivalence() -> None:
    """Eager vs ``torch.compile`` + ``CUDAGraphWrapper`` agree numerically.

    Uses streaming + CFG and runs past ``_cuda_graph_capture_ar_idx``
    so both the ``drain`` (filling) and captured (steady) paths fire.
    """
    device = torch.device("cuda")
    seed = 0

    base = _with_cfg(build_cfg_autoregressive(seed=seed), guidance_scale=2.0)
    fast = with_compile_and_cuda_graph(base)

    torch.manual_seed(seed)
    eager_pipeline = base.setup().to(device).eval()

    torch.manual_seed(seed)
    fast_pipeline = fast.setup().to(device).eval()

    # The fast pipeline's state_dict keys differ (compile wraps into
    # ``OptimizedModule``, CUDAGraphWrapper adds submodules), but both
    # rollouts use the same raw network constructed under the same
    # ``manual_seed``, so underlying weights match bit-for-bit. Assert
    # that here so any regression is caught early.
    eager_raw = eager_pipeline.diffusion_model.transformer.network
    fast_raw = fast_pipeline.diffusion_model.transformer.network
    fast_raw_inner = getattr(fast_raw, "_orig_mod", fast_raw)
    assert isinstance(eager_raw, torch.nn.Module)
    assert isinstance(fast_raw_inner, torch.nn.Module)
    for (name_e, p_e), (name_f, p_f) in zip(
        eager_raw.named_parameters(),
        fast_raw_inner.named_parameters(),
        strict=True,
    ):
        assert name_e == name_f, (name_e, name_f)
        torch.testing.assert_close(
            p_e.detach(),
            p_f.detach(),
            rtol=0,
            atol=0,
            msg=f"seeded init diverged on parameter {name_e}",
        )

    cfg = eager_pipeline.diffusion_model.transformer.config
    assert isinstance(cfg, TemplateTransformerConfig)
    # Cross the filling → steady boundary with at least one steady
    # step so both ``drain`` and captured ``__call__`` fire.
    num_ar_steps = (cfg.sink_size_t + cfg.window_size_t) // cfg.len_t + 2

    H, W = TEMPLATE_DEFAULT_HEIGHT, TEMPLATE_DEFAULT_WIDTH
    B = TEMPLATE_DEFAULT_BATCH_SIZE
    inputs = _make_inputs(
        cfg,
        batch_size=B,
        height=H,
        width=W,
        control_channels=8,
        context_channels=16,
        n_context_tokens=4,
        device=device,
        seed=seed,
    )

    eager_outputs = _run_rollout(
        pipeline=eager_pipeline,
        num_ar_steps=num_ar_steps,
        use_cfg=True,
        inputs=inputs,
        height=H,
        width=W,
    )
    fast_outputs = _run_rollout(
        pipeline=fast_pipeline,
        num_ar_steps=num_ar_steps,
        use_cfg=True,
        inputs=inputs,
        height=H,
        width=W,
    )

    assert len(eager_outputs) == len(fast_outputs) == num_ar_steps
    # bf16 + Inductor TF32 drift; output ``std ~ 50`` and observed
    # ``max_abs ~ 1.5`` (~3%). Tolerances below admit that floor while
    # still catching a real regression.
    for ar_idx, (eager, fast_out) in enumerate(zip(eager_outputs, fast_outputs)):
        assert fast_out.shape == eager.shape
        assert torch.isfinite(fast_out).all(), f"NaN at AR step {ar_idx} (fast)"
        assert torch.isfinite(eager).all(), f"NaN at AR step {ar_idx} (eager)"
        torch.testing.assert_close(
            fast_out,
            eager,
            rtol=5e-2,
            atol=2.0,
            msg=f"mismatch at AR step {ar_idx}",
        )


## CP equivalence (two-invocation; see module docstring)


def _cp_one_predict_flow(
    *,
    device: torch.device,
    seed: int,
    ar_idx: int,
    timestep_value: float,
) -> torch.Tensor:
    """Build a transformer + gather one ``predict_flow`` pass.

    Rank-agnostic: the transformer auto-detects ``cp_size`` from the
    active process group. All tensors that must match across ranks are
    drawn globally from a fixed seed, then CP-split inside the helper.
    Returns the gathered flow tensor ``[B, L, C]`` — pre-unpatchify,
    post-attention; the boundary where CP equivalence is meaningful.
    """
    from flashdreams.core.distributed.context_parallel import split_inputs_cp

    base = _with_cfg(build_cfg_autoregressive(seed=seed), guidance_scale=2.0)
    torch.manual_seed(seed)
    pipeline = base.setup().to(device).eval()
    transformer = pipeline.diffusion_model.transformer
    assert isinstance(transformer, TemplateTransformer)
    cfg = transformer.config

    gen = torch.Generator(device=device).manual_seed(seed + 17)
    B = TEMPLATE_DEFAULT_BATCH_SIZE
    H, W = TEMPLATE_DEFAULT_HEIGHT, TEMPLATE_DEFAULT_WIDTH
    kt, kh, kw = cfg.patch_size
    patch_volume = kt * kh * kw
    network_in_ch = cfg.network.in_channels
    # Pre-patchify ``control_global`` carries ``raw_in_ch``; the patch
    # fold lives in the channel dim downstream.
    raw_in_ch = network_in_ch // patch_volume
    L_post_patchify = (cfg.len_t // kt) * (H // kh) * (W // kw)
    dtype = cfg.dtype
    context = torch.randn(B, 4, 16, device=device, generator=gen, dtype=dtype)
    neg_context = torch.randn(B, 4, 16, device=device, generator=gen, dtype=dtype)
    control_global = torch.randn(
        B,
        raw_in_ch,
        cfg.len_t,
        H,
        W,
        device=device,
        generator=gen,
        dtype=dtype,
    )
    noisy_global = torch.randn(
        B,
        L_post_patchify,
        network_in_ch,
        device=device,
        generator=gen,
        dtype=dtype,
    )

    cache = pipeline.initialize_cache(
        transformer_context={
            "context": context,
            "negative_context": neg_context,
            "height": H,
            "width": W,
        }
    )
    transformer_cache = cache.transformer_cache
    assert isinstance(transformer_cache, TemplateTransformerCache)

    control_local = transformer.patchify_and_maybe_split_cp(control_global)
    noisy_local = split_inputs_cp(
        noisy_global, seq_dim=1, cp_group=transformer._cp_group
    )

    transformer_cache.start(ar_idx)
    flow_local = transformer.predict_flow(
        noisy_latent=noisy_local,
        timestep=torch.tensor(timestep_value, device=device),
        cache=transformer_cache,
        input=control_local,
    )
    transformer_cache.finalize(ar_idx)

    return cat_outputs_cp(flow_local, seq_dim=1, cp_group=transformer._cp_group)


def test_template_cp_equivalence() -> None:
    """Non-distributed vs ``cp_size=world_size`` produce the same global output.

    Two-invocation protocol (see module docstring):

    - ``WORLD_SIZE == 1``: write the global flow tensor to
      :data:`_CP_REFERENCE_PATH`.
    - ``WORLD_SIZE >= 2`` (under ``torchrun``): compare rank 0's
      gathered tensor to the reference.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    if not torch.cuda.is_available():
        pytest.skip("CP equivalence requires CUDA.")

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    if world_size == 1:
        flow_global = _cp_one_predict_flow(
            device=device,
            seed=0,
            ar_idx=0,
            timestep_value=500.0,
        )
        _CP_REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"flow_global": flow_global.detach().cpu()},
            _CP_REFERENCE_PATH,
        )
        assert torch.isfinite(flow_global).all()
        return

    # Check the reference file *before* initialising NCCL so a missing
    # reference doesn't leak the process group via ``pytest.skip``.
    if not _CP_REFERENCE_PATH.exists():
        pytest.skip(
            f"CP reference {_CP_REFERENCE_PATH} not found — run the "
            f"single-GPU branch (plain pytest, no torchrun) first."
        )

    assert dist.is_nccl_available()
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )

    # Anything that can raise after ``init_process_group`` must run in a
    # ``try``/``finally`` — otherwise an assertion failure (or any other
    # exception) would leak the process group into the next test in the
    # same torchrun worker.
    try:
        flow_global = _cp_one_predict_flow(
            device=device,
            seed=0,
            ar_idx=0,
            timestep_value=500.0,
        )

        if rank == 0:
            reference = torch.load(_CP_REFERENCE_PATH, weights_only=True)["flow_global"]
            # bf16 + ring-attention fp32 LSE merge gives ~2-3 decimal
            # digits at the gathered flow; 2e-2 catches real shard bugs
            # without flaking on merge drift.
            torch.testing.assert_close(
                flow_global.detach().cpu(),
                reference,
                rtol=2e-2,
                atol=2e-2,
            )
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    # Lightweight manual driver. Every test here requires CUDA.
    if torch.cuda.is_available():
        torch.manual_seed(0)
        test_template_latent_shape_respects_cp_size()
        print("[OK] latent_shape CP-aware")
        test_template_bidirectional(seed=0)
        print("[OK] bidirectional")
        test_template_no_control()
        print("[OK] no-control rollout")
        test_template_streaming_cfg()
        print("[OK] streaming + CFG")
        test_template_cfg_rejects_missing_negative_context()
        print("[OK] CFG uncond-context assertion")
        test_template_compile_and_cudagraph_equivalence()
        print("[OK] compile + CUDAGraphWrapper equivalence")
    else:
        print("[SKIP] CUDA-only tests (no CUDA device available)")
