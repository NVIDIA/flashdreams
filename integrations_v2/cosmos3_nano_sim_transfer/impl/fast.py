# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fast path of the Transfer rollout: K/V written in place, CUDA graphs, optional compile.

:class:`FastTransferRollout` runs the passes of
:class:`~cosmos3_nano_sim_transfer.impl.rollout.TransferRollout` (control pass,
denoise passes, clean replay) on the same weights, positions, noise and
history, with three changes that do not change the math:

- **One fixed K/V buffer per layer.** It holds the prompt's K/V right-aligned
  in the first ``text_slots`` positions, then the bounded stream cache (the
  ``BlockKVCache`` storage is a view of the rest). A pass writes its own K/V
  straight into its slot (the control's, or the RGB slot, which the denoise
  passes use as scratch until the replay fills it) and attends over one
  contiguous slice of the buffer. The eager rollout concatenates prompt,
  history and current K/V for every layer of every pass instead (about 3.5 GB
  of copies per pass at a 30/3 history). Attention runs the same SDPA kernel
  over the same keys in the same order, so the result is the eager result.
- **CUDA graphs.** A pass's slices depend only on its kind (control, denoise,
  replay) and its write slot: a session's first ``window_frames - 1`` latents
  each have their own slots, and from then on every latent uses the same ones
  (the window rolls in place). Each such geometry is captured once, after one
  eager call, and replayed from then on; all graphs share one memory pool,
  since passes run one at a time and their outputs are copied out at once. A
  warm-up of ``1 + graph_warmup_iters`` sessions (:attr:`FastTransferRollout.warmup_sessions`)
  captures every geometry.
- **Compile (optional).** The work around attention (norms, projections,
  RoPE, MLP) is compiled with ``torch.compile`` as one function for all
  layers; attention and the cache writes stay as they are. Inductor keeps
  eager's intermediate bfloat16 roundings inside its fused kernels
  (:data:`COMPILE_OPTIONS`), but fused reductions still round differently, so
  compiled output is close to eager, not equal to it.

The prompt must fit ``text_slots`` (the export's ``text_cache_max_len``).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F
from cosmos_interactive._network import CosmosMoTNetwork, _apply_rope
from cosmos_interactive._patchify import unpatchify_latent
from torch import Tensor

from ..config import HistoryProfile, TransferModelConfig
from .rollout import LATENT_HW, TransferRollout

ACCELERATIONS = ("eager", "graphs", "compile")
"""How a rollout runs: ``eager`` (:class:`TransferRollout`), ``graphs``
(:class:`FastTransferRollout` with CUDA graphs) or ``compile`` (graphs over
compiled layers)."""


COMPILE_OPTIONS: dict[str, str | int | bool | Callable[..., Any]] = {
    "emulate_precision_casts": True
}
"""Inductor options of the compiled layers: round to bfloat16 where eager does,
inside fused kernels too. Without it a teacher-forced step lands about 6 dB
further from eager at the first latent and 9 dB at the worst later one."""


def _rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """``cosmos_interactive``'s RMSNorm, as a function."""
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x.to(dtype)) * weight


def _layer_functions(
    *, heads: int, kv_heads: int, head_dim: int, eps: float
) -> tuple[Callable[..., tuple[Tensor, Tensor, Tensor]], Callable[..., Tensor]]:
    """The generation half of a layer around attention, with the weights as arguments.

    ``pre`` is the generation half's input norm, Q/K/V projections, QK norms and
    RoPE; ``post`` the output projection, residual, post-attention norm, SwiGLU
    MLP and residual. They are the operations of ``_MoTLayer.gen_compute`` in
    the same order, so eager results are bit-equal; taking the weights as
    arguments lets one compiled function serve every layer.
    """

    def pre(
        h: Tensor,
        cos: Tensor,
        sin: Tensor,
        norm: Tensor,
        wq: Tensor,
        bq: Tensor | None,
        wk: Tensor,
        bk: Tensor | None,
        wv: Tensor,
        bv: Tensor | None,
        q_norm: Tensor,
        k_norm: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        b, s, _ = h.shape
        hn = _rms_norm(h, norm, eps)
        q = _rms_norm(F.linear(hn, wq, bq).view(b, s, heads, head_dim), q_norm, eps)
        k = _rms_norm(F.linear(hn, wk, bk).view(b, s, kv_heads, head_dim), k_norm, eps)
        v = F.linear(hn, wv, bv).view(b, s, kv_heads, head_dim).transpose(1, 2)
        q, k = _apply_rope(q.transpose(1, 2), k.transpose(1, 2), cos, sin)
        return q, k, v

    def post(
        h: Tensor,
        attn: Tensor,
        wo: Tensor,
        bo: Tensor | None,
        norm: Tensor,
        w_gate: Tensor,
        w_up: Tensor,
        w_down: Tensor,
    ) -> Tensor:
        b, s, _ = h.shape
        h = h + F.linear(attn.transpose(1, 2).reshape(b, s, -1), wo, bo)
        hn = _rms_norm(h, norm, eps)
        return h + F.linear(F.silu(F.linear(hn, w_gate)) * F.linear(hn, w_up), w_down)

    return pre, post


class _PassGraph:
    """One pass geometry as a CUDA graph in a shared memory pool.

    The first ``warmup`` calls run eagerly; the next one captures ``fn`` on
    static copies of its tensor arguments and replays it. Later calls copy
    their arguments into the static inputs, replay and return a copy of the
    output (or ``None``), so a later graph in the pool cannot overwrite it.
    """

    def __init__(
        self, fn: Callable[..., Tensor | None], pool: Any, warmup: int
    ) -> None:
        self.fn = fn
        self.pool = pool
        self.warmup = int(warmup)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.inputs: list[Tensor] = []
        self.output: Tensor | None = None

    def __call__(self, *args: Tensor) -> Tensor | None:
        if self.graph is None and self.warmup > 0:
            self.warmup -= 1
            return self.fn(*args)
        if self.graph is None:
            self.inputs = [arg.clone() for arg in args]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(
                graph, pool=self.pool, capture_error_mode="thread_local"
            ):
                self.output = self.fn(*self.inputs)
            self.graph = graph
        else:
            for slot, arg in zip(self.inputs, args, strict=True):
                slot.copy_(arg)
        self.graph.replay()
        return None if self.output is None else self.output.clone()


def _layer_weights(layer: Any) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """The generation half's weights of one layer in ``pre``/``post`` argument order."""
    gen = layer.gen
    attn = gen.self_attn
    mlp = gen.mlp
    for linear in (mlp.gate_proj, mlp.up_proj, mlp.down_proj):
        if linear.bias is not None:
            raise ValueError("the fast path expects an MLP without biases")
    pre = (
        gen.input_layernorm.weight,
        attn.q_proj.weight,
        attn.q_proj.bias,
        attn.k_proj.weight,
        attn.k_proj.bias,
        attn.v_proj.weight,
        attn.v_proj.bias,
        attn.q_norm.weight,
        attn.k_norm.weight,
    )
    post = (
        attn.o_proj.weight,
        attn.o_proj.bias,
        gen.post_attention_layernorm.weight,
        mlp.gate_proj.weight,
        mlp.up_proj.weight,
        mlp.down_proj.weight,
    )
    return pre, post


class FastTransferRollout(TransferRollout):
    """:class:`TransferRollout` on fixed K/V buffers, with CUDA graphs and optional compile.

    Takes the arguments of :class:`TransferRollout` plus:

    Args:
        cuda_graphs: Replay each pass geometry as a captured graph (CUDA only).
        graph_fill: Capture the geometries of a filling window too; ``False``
            captures only the full window's and runs a session's first
            ``window_frames - 1`` latents eagerly.
        compile_layers: Compile the work around attention (see the module docstring).
        text_slots: Prompt tokens the buffers hold; the export's
            ``text_cache_max_len`` when ``None``.
        graph_warmup_iters: Eager calls of a pass geometry before it is captured.

    Raises:
        ValueError: Graphs on a device other than CUDA, or grouped-query heads
            expanded by hand (``native_gqa_attention=False``).
    """

    def __init__(
        self,
        network: CosmosMoTNetwork,
        model: TransferModelConfig,
        *,
        history: HistoryProfile,
        latent_hw: tuple[int, int] = LATENT_HW,
        fps: float = 30.0,
        native_gqa_attention: bool = True,
        noise_dtype: torch.dtype = torch.bfloat16,
        cuda_graphs: bool = True,
        graph_fill: bool = True,
        compile_layers: bool = False,
        text_slots: int | None = None,
        graph_warmup_iters: int = 1,
    ) -> None:
        if not native_gqa_attention:
            raise ValueError(
                "the fast path attends with SDPA's native grouped-query heads"
            )
        super().__init__(
            network,
            model,
            history=history,
            latent_hw=latent_hw,
            fps=fps,
            native_gqa_attention=True,
            noise_dtype=noise_dtype,
        )
        if cuda_graphs and self.device.type != "cuda":
            raise ValueError(f"CUDA graphs need a CUDA device, not {self.device}")
        self.cuda_graphs = bool(cuda_graphs)
        self.graph_fill = bool(graph_fill)
        self.compiled = bool(compile_layers)
        self.graph_warmup_iters = int(graph_warmup_iters)
        self.text_slots = int(
            model.text_cache_max_len if text_slots is None else text_slots
        )
        if self.text_slots < 1:
            raise ValueError(f"text_slots must be positive, got {self.text_slots}")
        llm = network.config.spec.llm
        gen0 = self.cache.gen[0]
        self.cache_tokens = int(gen0._k.shape[2])
        shape = (
            1,
            llm.num_key_value_heads,
            self.text_slots + self.cache_tokens,
            llm.head_dim,
        )
        self.k_buffers: list[Tensor] = []
        self.v_buffers: list[Tensor] = []
        for gen in self.cache.gen:
            k = torch.zeros(shape, device=self.device, dtype=gen._k.dtype)
            v = torch.zeros(shape, device=self.device, dtype=gen._v.dtype)
            # The bounded cache keeps its bookkeeping and rolls in place in the buffer.
            gen._k = k[:, :, self.text_slots :]
            gen._v = v[:, :, self.text_slots :]
            self.k_buffers.append(k)
            self.v_buffers.append(v)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()  # the cache's first storage is unused now
        attn0 = self.layers[0].gen.self_attn
        pre, post = _layer_functions(
            heads=attn0.num_heads,
            kv_heads=attn0.num_kv_heads,
            head_dim=attn0.head_dim,
            eps=float(self.layers[0].gen.input_layernorm.eps),
        )
        for layer in self.layers:
            norms = (
                layer.gen.input_layernorm,
                layer.gen.post_attention_layernorm,
                layer.gen.self_attn.q_norm,
                layer.gen.self_attn.k_norm,
            )
            if any(float(n.eps) != float(norms[0].eps) for n in norms):
                raise ValueError("the fast path expects one RMSNorm epsilon")
        if self.compiled:
            pre = torch.compile(pre, dynamic=False, options=COMPILE_OPTIONS)
            post = torch.compile(post, dynamic=False, options=COMPILE_OPTIONS)
        self._pre = pre
        self._post = post
        self._weights = [_layer_weights(layer) for layer in self.layers]
        block = 2 * self.tokens_per_latent
        # Write slots of a full window: the control at the last block, the RGB after it.
        self.steady_control_start = self.cache_tokens - block
        self.steady_rgb_start = self.cache_tokens - self.tokens_per_latent
        self.text_start = self.text_slots
        """Buffer position of the first prompt token."""
        self._buffered_text: Tensor | None = None
        self._graphs: dict[tuple[str, int], _PassGraph] = {}
        self._pool = torch.cuda.graph_pool_handle() if self.cuda_graphs else None

    @property
    def warmup_latents(self) -> int:
        """Latents per warm-up session: the window fills, then the full
        window's geometries run their eager calls and are captured."""
        if not self.cuda_graphs:
            return 3
        return self.history.window_frames + self.graph_warmup_iters

    @property
    def warmup_sessions(self) -> int:
        """Warm-up sessions that capture every geometry: a filling window's
        control and replay geometries run once per session."""
        if not (self.cuda_graphs and self.graph_fill):
            return 1
        return 1 + self.graph_warmup_iters

    @property
    def graphs_captured(self) -> int:
        """Pass geometries replaying a captured graph."""
        return sum(1 for graph in self._graphs.values() if graph.graph is not None)

    @torch.no_grad()
    def start(self, text_ids: Tensor, *, seed: int) -> None:
        """See :meth:`TransferRollout.start`; the prompt must fit ``text_slots``.

        Raises:
            ValueError: The prompt is empty or longer than ``text_slots``.
        """
        ids = torch.as_tensor(text_ids, dtype=torch.long).reshape(-1).cpu()
        if ids.numel() > self.text_slots:
            raise ValueError(
                f"the prompt has {ids.numel()} tokens; the fast path holds "
                f"{self.text_slots} (use the eager rollout for longer prompts)"
            )
        super().start(ids, seed=seed)
        if self._buffered_text is None or not torch.equal(self._buffered_text, ids):
            n = int(ids.numel())
            start = self.text_slots - n
            for i, (k, v) in enumerate(
                zip(self.k_buffers, self.v_buffers, strict=True)
            ):
                text_k, text_v = self.cache.text_k[i], self.cache.text_v[i]
                assert text_k is not None and text_v is not None
                k[:, :, start : self.text_slots].copy_(text_k)
                v[:, :, start : self.text_slots].copy_(text_v)
                self.cache.text_k[i] = k[:, :, start : self.text_slots]
                self.cache.text_v[i] = v[:, :, start : self.text_slots]
            if start != self.text_start:
                # The slices the graphs read start elsewhere now.
                self._graphs.clear()
            self.text_start = start
            self._buffered_text = ids

    def _layers(
        self, h: Tensor, cos: Tensor, sin: Tensor, write: int, *, clean: bool
    ) -> Tensor:
        """All layers over ``h``, writing its K/V at cache slot ``write``.

        A clean pass only needs the K/V of the last layer, not its output.
        """
        s = int(h.shape[1])
        begin = self.text_start
        slot = self.text_slots + write
        end = slot + s
        last = len(self._weights) - 1
        for i, ((pre_w, post_w), k_buf, v_buf) in enumerate(
            zip(self._weights, self.k_buffers, self.v_buffers, strict=True)
        ):
            q, k, v = self._pre(h, cos, sin, *pre_w)
            k_buf[:, :, slot:end].copy_(k)
            v_buf[:, :, slot:end].copy_(v)
            if clean and i == last:
                break
            attn = F.scaled_dot_product_attention(
                q, k_buf[:, :, begin:end], v_buf[:, :, begin:end], enable_gqa=True
            )
            h = self._post(h, attn, *post_w)
        return h

    def _clean_pass(self, write: int) -> Callable[[Tensor, Tensor, Tensor], None]:
        def run(tokens: Tensor, cos: Tensor, sin: Tensor) -> None:
            h = self.net.vae2llm(tokens.to(self.dtype))[None]
            self._layers(h, cos, sin, write, clean=True)

        return run

    def _denoise_pass(
        self, write: int
    ) -> Callable[[Tensor, Tensor, Tensor, Tensor], Tensor]:
        def run(tokens: Tensor, timestep: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
            h = self.net.vae2llm(tokens.to(self.dtype))
            h = h + self.net.embed_uniform_timestep(timestep, h.shape[0]).to(self.dtype)
            h = self._layers(h[None], cos, sin, write, clean=False)
            return self.net.llm2vae(self.net.norm_moe_gen(h))[0]

        return run

    def _graphed(
        self, kind: str, write: int, build: Callable[[int], Callable[..., Any]]
    ) -> Callable[..., Any]:
        """The pass of ``kind`` writing at cache slot ``write``, as a CUDA graph when enabled."""
        steady = write == (
            self.steady_control_start if kind == "control" else self.steady_rgb_start
        )
        if not self.cuda_graphs or not (steady or self.graph_fill):
            return build(write)
        graph = self._graphs.get((kind, write))
        if graph is None:
            graph = _PassGraph(build(write), self._pool, self.graph_warmup_iters)
            self._graphs[(kind, write)] = graph
        return graph

    def _write_clean(
        self, tokens: Tensor, cos: Tensor, sin: Tensor, start: int
    ) -> None:
        kind = "replay" if start % (2 * self.tokens_per_latent) else "control"
        tokens = tokens.to(device=self.device)
        self._graphed(kind, start, self._clean_pass)(tokens, cos, sin)

    def _flow(
        self, latent: Tensor, timestep: Tensor, cos: Tensor, sin: Tensor, end: int
    ) -> Tensor:
        tokens = self._tokens(latent).to(device=self.device)
        timestep = torch.as_tensor(timestep, device=self.device, dtype=torch.float32)
        flow = self._graphed("denoise", end, self._denoise_pass)(
            tokens, timestep, cos, sin
        )
        return unpatchify_latent(
            flow,
            channels=self.channels,
            grid_t=1,
            grid_h=self.grid_hw[0],
            grid_w=self.grid_hw[1],
            patch_spatial=self.patch,
        )


def make_rollout(
    accel: str,
    network: CosmosMoTNetwork,
    model: TransferModelConfig,
    **kwargs: Any,
) -> TransferRollout:
    """A rollout for ``accel`` (one of :data:`ACCELERATIONS`)."""
    if accel == "eager":
        return TransferRollout(network, model, **kwargs)
    if accel in ("graphs", "compile"):
        return FastTransferRollout(
            network,
            model,
            cuda_graphs=True,
            compile_layers=accel == "compile",
            **kwargs,
        )
    raise ValueError(f"unknown acceleration {accel!r}; use one of {ACCELERATIONS}")


__all__ = ["ACCELERATIONS", "COMPILE_OPTIONS", "FastTransferRollout", "make_rollout"]
