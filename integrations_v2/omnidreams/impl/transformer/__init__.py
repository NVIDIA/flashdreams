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

"""Multi-view, HDMap-conditioned Cosmos DiT for streaming omnidreams."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn.functional as F
from omnidreams.impl.native.acceleration import (
    NativeAccelerationConfig,
    NativeAccelerationMode,
    NativeAccelerationUnavailable,
    NativeBackendSelection,
    require_extension_symbols,
)
from torch import Tensor

from flashdreams.core.attention.rope import (
    KVCacheRelativeRotaryPositionEmbedding3D,
    RotaryPositionEmbedding3D,
)
from flashdreams.core.checkpoint.load import load_checkpoint
from flashdreams.core.distributed.context_parallel import split_inputs_cp
from flashdreams.infra.acceleration.cuda_graph_dispatch import (
    CUDAGraphDispatch,
    cuda_graph_capture_ar_index,
)
from flashdreams.infra.compile import compile_module
from flashdreams.infra.diffusion.transformer import (
    Transformer,
    TransformerAutoregressiveCache,
    TransformerConfig,
)

from .context_parallel import (
    HierarchicalCPGroups,
    create_hierarchical_cp_groups,
)
from .modules import AttentionBackend
from .network import (
    CosmosDiTNetwork,
    CosmosDiTNetworkCache,
    CosmosDiTNetworkConfig,
)

## Default camera names / view-index mapping

DEFAULT_CAMERAS: tuple[str, ...] = (
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_right_70fov",
    "camera_rear_tele_30fov",
    "camera_rear_left_70fov",
    "camera_cross_left_120fov",
    "camera_front_tele_30fov",
)

DEFAULT_CAMERA_VIEW_MAPPING: dict[str, int] = dict(
    zip(DEFAULT_CAMERAS, range(len(DEFAULT_CAMERAS)))
)


## Per-rollout cache


@dataclass(kw_only=True)
class TextEditGuidance:
    """Transient old/new prompt guidance for a mid-rollout text edit."""

    scale: float
    """Strength applied to the new-minus-old flow direction."""

    chunks_remaining: int
    """Number of upcoming autoregressive chunks to guide."""

    kv_old: list[tuple[Tensor, Tensor]] = field(default_factory=list)
    """Per-block cross-attention K/V for the pre-edit prompt."""

    kv_new: list[tuple[Tensor, Tensor]] = field(default_factory=list)
    """Per-block cross-attention K/V for the post-edit prompt."""

    use_lora: bool = False
    """Whether a distilled LoRA realizes this window with one forward."""


@dataclass(kw_only=True)
class CosmosTransformerCache(TransformerAutoregressiveCache):
    """Long-lived AR cache for the Cosmos transformer."""

    network_cache: CosmosDiTNetworkCache
    """Per-block self-attn KV + (text-only) cross-attn KV."""

    network_cache_uncond: CosmosDiTNetworkCache | None = None
    """Unconditional cache for CFG; ``None`` disables CFG."""

    rope_adapter: RotaryPositionEmbedding3D | KVCacheRelativeRotaryPositionEmbedding3D
    """3D RoPE adapter, advanced via ``shift_t`` each step."""

    rope_freqs: Tensor | None = None
    """Self-attention RoPE frequencies for the current AR step.
    Shape ``[L, 1, 1, head_dim // 2]`` after CP. Recomputed once per
    AR step in :meth:`start` and reused across cond and uncond branches
    (and across all scheduler steps within the AR step)."""

    image: Tensor
    """First-frame VAE latent, T-padded to ``len_t`` and patchified.
    Injects into the noisy / predicted latent at AR step 0."""

    mask_first_block: Tensor
    """``[B, V, T, 1, H, W]`` mask with ones on the first temporal latent
    frame; used at AR step 0."""

    mask_other_blocks: Tensor
    """All-zero counterpart used at AR step >= 1."""

    view_indices: Tensor | None = None
    """Per-view index tensor for AdaLN view modulation; ``None`` when
    ``num_views == 1``."""

    autoregressive_index: int = -1
    """AR step index for the chunk currently being processed; ``-1`` before the first ``start``."""

    text_edit_guidance: TextEditGuidance | None = None
    """Transient text guidance, or ``None`` when no edit window is active."""

    def start(self, autoregressive_index: int) -> None:
        guidance = self.text_edit_guidance
        if guidance is not None and autoregressive_index > self.autoregressive_index:
            if guidance.chunks_remaining <= 0:
                self.text_edit_guidance = None
            else:
                guidance.chunks_remaining -= 1
        # Hoist KV pre-update and RoPE shift out of the graph-captured forward
        # (predict_flow runs eager_mode=False; cond/uncond share rope_freqs).
        self.rope_freqs = self.rope_adapter.shift_t(autoregressive_index)

        self.autoregressive_index = autoregressive_index
        self.network_cache.before_update(autoregressive_index)
        if self.network_cache_uncond is not None:
            self.network_cache_uncond.before_update(autoregressive_index)

    def finalize(self, autoregressive_index: int) -> None:
        self.network_cache.after_update(autoregressive_index)
        if self.network_cache_uncond is not None:
            self.network_cache_uncond.after_update(autoregressive_index)


## Config


@dataclass(kw_only=True)
class CosmosTransformerConfig(TransformerConfig):
    """Config for the Cosmos transformer.

    Bakes in the temporal layout (``len_t``, ``window_size_t``,
    ``sink_size_t``), CFG / compile knobs, and the multi-view layout.
    Per-rollout spatial layout (``height``, ``width``) is supplied to
    :meth:`CosmosTransformer.initialize_autoregressive_cache` so one
    instance can serve multiple resolutions. CP size is auto-detected
    from ``torch.distributed.get_world_size()`` at construction; build
    the pipeline under a ``torch.distributed`` initialization with the
    desired world size to opt in.

    Builders are responsible for wiring the embedded
    :class:`CosmosDiTNetworkConfig`:

    - ``network.additional_concat_ch`` — ``16`` for the Wan-VAE HDMap
      branch, ``192`` for the pixel-shuffle HDMap branch, ``0`` to
      disable HDMap conditioning.
    - ``network.enable_cross_view_attn`` — ``True`` iff ``num_views > 1``.
    """

    _target: type["CosmosTransformer"] = field(
        default_factory=lambda: CosmosTransformer
    )

    network: CosmosDiTNetworkConfig = field(default_factory=CosmosDiTNetworkConfig)
    """Backbone Cosmos DiT network config."""

    dtype: torch.dtype = torch.bfloat16
    """Network parameter / activation dtype."""

    checkpoint_path: str | None = None
    """Optional path to a pretrained checkpoint; ``None`` keeps the random init."""

    state_dict_transform: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None
    """Pre-load state-dict remap. Defaults to a ``net.`` prefix stripper."""

    batch_shape: tuple[int, ...] = (1,)
    """Batch dims of the latent (excluding ``V, T, HW, D``)."""

    num_views: int = 1
    """Number of camera views; >1 enables cross-view attention."""

    len_t: int = 4
    """Latent frames per AR chunk."""

    h_extrapolation_ratio: float = 3.0
    """RoPE extrapolation along H (3.0 @ 720p)."""

    w_extrapolation_ratio: float = 3.0
    """RoPE extrapolation along W."""

    window_size_t: int = 8
    """Self-attention sliding window (pre-patchify T)."""

    sink_size_t: int = 0
    """Sink-token count (pre-patchify T)."""

    early_short_history_block_count: int | None = None
    """Number of initial blocks limited to one chunk of visual history."""

    compile_network: bool = True
    """``torch.compile`` the network."""

    use_cuda_graph: bool = True
    """Wrap in ``CUDAGraphWrapper`` for steady-state replay. Caller must
    keep non-staged inputs at stable storage addresses across calls."""

    cuda_graph_warmup_iters: int = 2
    """Eager calls before capture (>= 2 to drain Inductor autotune)."""

    skip_finalize_kv_cache: bool = False
    """Skip the KV cache finalize step."""

    native_dit_acceleration: NativeAccelerationMode = "disabled"
    """Native optimized DiT policy: ``disabled``, ``auto``, or ``required``."""

    native_dit_build_root: str | None = None
    """Optional native extension build/cache root."""

    native_dit_max_jobs: int | str | None = None
    """Optional PyTorch/Ninja job cap for the native DiT build."""

    native_dit_verbose_build: bool = False
    """Forward verbose build output from the native extension loader."""

    native_dit_backend: Literal["fp8_kvcache_cudnn", "bf16"] = "fp8_kvcache_cudnn"
    """Optimized native DiT compute backend."""

    native_dit_attention_backend: str = "auto"
    """Optimized native attention backend.

    ``auto`` selects the current default, which resolves to the portable cuDNN
    FP8 SDPA path. Set ``sage2``, ``sparge``, ``sage3``, or ``sage3_fp8``
    explicitly to opt into an experimental attention backend. The framework
    ``self_attention_backend="sage2"`` setting also selects ``sage2`` when
    native DiT acceleration is enabled. With ``native_dit_acceleration="auto"``,
    an unavailable Native Sage2 kernel falls back to framework Sage2. An
    explicit Native backend remains a Native-only override: for example,
    framework ``self_attention_backend="omnidreams"`` plus Native ``sage2``
    falls back to the framework's cuDNN attention when Native DiT is unavailable.
    """

    native_dit_sparge_topk: float | None = None
    """Optional Sparge self-attention top-k ratio.

    ``None`` uses ``0.25`` for Sparge and Sparge/SageAttention-3 hybrid runs.
    """

    native_dit_sparge_hybrid_period: int | None = None
    """Optional Sparge/SageAttention-3 hybrid period.

    ``None`` uses ``0``. Set a value greater than ``1`` with
    ``native_dit_attention_backend="sparge"`` to enable the hybrid schedule.
    """

    native_dit_sparge_hybrid_phase: int | None = None
    """Optional Sparge hybrid phase. ``None`` uses backend defaults."""

    guidance_scale: float = 1.0
    """CFG scale. ``1.0`` disables CFG; ``> 1.0`` requires negative text embeddings."""

    @property
    def requires_negative_text_embeddings(self) -> bool:
        """Whether cache initialization must receive negative text embeddings."""
        return self.guidance_scale > 1.0


## Default state-dict transform


def _strip_net_prefix(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Strip the ``net.`` prefix added by the upstream training stack."""
    out: dict[str, Tensor] = {}
    for k, v in state_dict.items():
        out[k[len("net.") :] if k.startswith("net.") else k] = v
    return out


@dataclass(frozen=True)
class _AttentionPolicy:
    """Resolved framework fallback and Native DiT attention choices.

    ``AttentionBackend.OMNIDREAMS`` names the original framework attention
    implementation, whose SDPA kernel is cuDNN. An explicit
    ``native_dit_attention_backend`` remains a Native-only override, so the
    legacy ``OMNIDREAMS`` + Native ``sage2`` combination deliberately falls
    back to framework cuDNN when automatic Native DiT selection is unavailable.
    """

    framework_backend: AttentionBackend
    """Framework attention backend used when Native DiT is not selected."""

    native_backend: str
    """Resolved Native attention backend passed to the Native executor."""

    native_backend_was_explicit: bool
    """Whether the Native backend came from a non-``auto`` override."""

    def cuda_graph_incompatibility(
        self,
        *,
        config: CosmosTransformerConfig,
        native_selected: bool,
    ) -> str | None:
        """Return why the effective attention path cannot use CUDA Graph.

        Args:
            config: Transformer configuration containing the graph setting.
            native_selected: Whether Native DiT is the effective execution path.

        Returns:
            An actionable incompatibility reason, or ``None`` when graph replay
            is supported by the effective path.
        """

        if not config.use_cuda_graph:
            return None
        if native_selected and self.native_backend == "sage2":
            return (
                "OmniDreams Native SageAttention 2 is incompatible with "
                "use_cuda_graph=True because replay can reuse stale attention "
                "outputs. Set use_cuda_graph=False or select another Native "
                "attention backend."
            )
        if not native_selected and self.framework_backend is AttentionBackend.SAGE2:
            return (
                "OmniDreams framework SageAttention 2 is incompatible with "
                "use_cuda_graph=True. Set use_cuda_graph=False or select another "
                "self_attention_backend."
            )
        return None

    def validate_cuda_graph(
        self, *, config: CosmosTransformerConfig, native_selected: bool
    ) -> None:
        """Reject Sage2 replay on the effective execution path.

        Native selection is resolved before this check when acceleration is
        enabled. That lets an unavailable Native-only Sage2 override fall back
        to the graph-safe framework cuDNN path, while the unified Sage2 switch
        still rejects its framework Sage2 fallback.

        Args:
            config: Transformer configuration containing the graph setting.
            native_selected: Whether Native DiT is the effective execution path.

        Raises:
            ValueError: If the effective Sage2 path cannot replay safely.
        """

        incompatibility = self.cuda_graph_incompatibility(
            config=config,
            native_selected=native_selected,
        )
        if incompatibility is not None:
            raise ValueError(incompatibility)


def _resolve_attention_policy(config: CosmosTransformerConfig) -> _AttentionPolicy:
    """Resolve framework and Native DiT attention settings once.

    The framework ``SAGE2`` setting is the unified Sage2 switch: Native DiT
    inherits it when enabled, while ``auto`` selection can fall back to the
    same framework Sage2 implementation. Other explicit Native backends retain
    their existing override semantics and framework fallback.

    Args:
        config: Transformer configuration containing both attention settings.

    Returns:
        The normalized framework fallback and Native attention selection.
    """

    framework_backend = AttentionBackend(config.network.self_attention_backend)
    configured_native_backend = config.native_dit_attention_backend
    native_backend_was_explicit = configured_native_backend != "auto"

    if configured_native_backend == "auto":
        native_backend = (
            "sage2" if framework_backend is AttentionBackend.SAGE2 else "cudnn"
        )
    else:
        native_backend = configured_native_backend

    return _AttentionPolicy(
        framework_backend=framework_backend,
        native_backend=native_backend,
        native_backend_was_explicit=native_backend_was_explicit,
    )


def _native_dit_sage2_availability(
    extension: Any,
    *,
    device: torch.device,
) -> tuple[bool, str]:
    """Check that Native DiT and its Sage2 kernel can run on a target device.

    Args:
        extension: Loaded OmniDreams Native extension.
        device: CUDA device selected for the model.

    Returns:
        Whether Native Sage2 is available and an explanatory reason.
    """

    symbols_available, reason = require_extension_symbols(
        "optimized_dit_forward",
        "sage2_is_built",
        "sage2_is_runtime_supported",
    )(extension)
    if not symbols_available:
        return False, reason

    is_built = extension.sage2_is_built
    is_runtime_supported = extension.sage2_is_runtime_supported
    if not callable(is_built) or not callable(is_runtime_supported):
        return False, "native Sage2 availability probes are not callable"
    try:
        if not bool(is_built()):
            return False, "native extension was built with Sage2 stubs"
    except Exception as exc:
        return False, f"native_extension.sage2_is_built() failed: {exc}"

    if not torch.cuda.is_available():
        return False, "CUDA is unavailable for Native Sage2"
    if device.type != "cuda":
        return False, f"target device {device} is not a CUDA device"
    if device.index is None:
        return False, "target CUDA device must have an explicit index"
    device_index = device.index
    try:
        if not bool(is_runtime_supported(device_index)):
            return (
                False,
                f"CUDA device {device_index} is not enabled for Native Sage2",
            )
    except Exception as exc:
        return False, (f"native_extension.sage2_is_runtime_supported() failed: {exc}")
    return True, f"Native DiT Sage2 is available on CUDA device {device_index}"


## Transformer


class CosmosTransformer(Transformer[CosmosTransformerCache]):
    """Multi-view, HDMap-conditioned Cosmos DiT as an infra transformer."""

    network: CosmosDiTNetwork

    def __init__(self, config: CosmosTransformerConfig) -> None:
        attention_policy = _resolve_attention_policy(config)
        if attention_policy.framework_backend is AttentionBackend.SAGE2:
            attention_policy.validate_cuda_graph(
                config=config,
                native_selected=False,
            )
            if (
                config.native_dit_acceleration != "disabled"
                and attention_policy.native_backend_was_explicit
                and attention_policy.native_backend != "sage2"
            ):
                raise ValueError(
                    "Conflicting OmniDreams attention backends: "
                    "self_attention_backend='sage2' requires "
                    "native_dit_attention_backend='auto' or 'sage2' when native "
                    "DiT acceleration is enabled."
                )
            if (
                config.native_dit_acceleration == "required"
                and config.native_dit_backend != "fp8_kvcache_cudnn"
            ):
                raise ValueError(
                    "OmniDreams Native SageAttention 2 requires "
                    "native_dit_backend='fp8_kvcache_cudnn'."
                )
        super().__init__(config)
        self.config: CosmosTransformerConfig = config
        self._attention_policy = attention_policy

        # Auto-detect CP world size from torch.distributed; non-distributed -> singleton groups.
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            self.cp_groups = create_hierarchical_cp_groups(
                world_size=world_size,
                rank=torch.distributed.get_rank(),
                V=config.num_views,
                T=config.len_t,
                single_group_as_none=True,
            )
        else:
            self.cp_groups = HierarchicalCPGroups(rank=0)

        # Pre-patchify temporal divisibility check; per-rollout
        # (height, width) is populated by initialize_autoregressive_cache.
        kt = config.network.patch_temporal
        assert config.len_t % kt == 0, (
            f"len_t ({config.len_t}) must be divisible by patch_temporal ({kt})."
        )
        if (
            not config.network.apply_rope_before_kvcache
            and config.native_dit_acceleration != "disabled"
        ):
            raise ValueError(
                "Cache-relative RoPE is not supported by native DiT acceleration; "
                "set native_dit_acceleration='disabled'"
            )
        if config.early_short_history_block_count is not None:
            if config.native_dit_acceleration != "disabled":
                raise ValueError(
                    "Early short history is not supported by native DiT acceleration; "
                    "set native_dit_acceleration='disabled'"
                )
            if not (
                1 <= config.early_short_history_block_count <= config.network.num_blocks
            ):
                raise ValueError(
                    "early_short_history_block_count must be between 1 and "
                    f"num_blocks ({config.network.num_blocks})"
                )
            if config.window_size_t < config.len_t:
                raise ValueError(
                    "Early short history requires window_size_t to be at least len_t"
                )
        self._output_height: int | None = None
        self._output_width: int | None = None

        self.network = CosmosDiTNetwork(config=config.network)
        self.network = self.network.to(dtype=config.dtype)
        self.network.eval()
        self.network.set_context_parallel_group(
            self_attn_group=self.cp_groups.THW_group,
            cross_view_attn_group=self.cp_groups.V_group,
        )

        if config.checkpoint_path is not None:
            transform = config.state_dict_transform or _strip_net_prefix
            state_dict = load_checkpoint(config.checkpoint_path)
            state_dict = transform(state_dict)
            self.network.load_state_dict(state_dict)
        self.network.update_parameters_after_loading_checkpoint()

        self._optimized_dit_executor: Any | None = None
        self._optimized_dit_selection: NativeBackendSelection | None = None
        # Pipeline setup constructs modules before moving them to the requested
        # device, so device-dependent Sage2 selection is finalized at cache init.
        self._native_sage2_selection_pending = (
            config.native_dit_acceleration != "disabled"
            and attention_policy.native_backend == "sage2"
        )
        self._native_sage2_selection_device: torch.device | None = None
        if (
            config.native_dit_acceleration != "disabled"
            and not self._native_sage2_selection_pending
        ):
            self._configure_optimized_dit_from_config(target_device=self.device)

        if (
            config.compile_network
            and self._optimized_dit_executor is None
            and not self._native_sage2_selection_pending
        ):
            self.network = compile_module(self.network)

        # Cond and CFG-uncond branches each get their own CUDA-graph wrapper
        # since each mutates an independent rolling KV cache.
        self._use_cuda_graph = config.use_cuda_graph
        self._cuda_graph_capture_ar_idx = cuda_graph_capture_ar_index(
            sink_size_t=config.sink_size_t,
            window_size_t=config.window_size_t,
            len_t=config.len_t,
        )
        self._reset_framework_dispatch()

        # Single view: flatten latent to 4D [B, V, L, D] so CP applies on L
        # directly. Multi-view: keep 5D [B, V, T, HW, D] for hierarchical CP.
        self.flatten_thw = config.num_views == 1
        self._finalizing_kv_cache = False
        self._text_edit_lora: Any | None = None

    def set_text_edit_lora(self, edit_lora: Any | None) -> None:
        """Attach a graph-safe distilled text-edit LoRA hook."""
        self._text_edit_lora = edit_lora

    def _configure_optimized_dit_from_config(
        self,
        *,
        target_device: torch.device,
    ) -> None:
        """Select Native DiT using the model's target CUDA device.

        Args:
            target_device: Device that will own model parameters and caches.
        """
        from omnidreams.impl.native import omnidreams_singleview

        attention_policy = getattr(self, "_attention_policy", None)
        if attention_policy is None:
            # Some focused tests construct the transformer with ``__new__`` to
            # isolate Native backend selection from model allocation.
            attention_policy = _resolve_attention_policy(self.config)
            self._attention_policy = attention_policy
        attention_backend = attention_policy.native_backend
        if attention_backend == "sage2":
            incompatibility = self._native_sage2_config_incompatibility()
            if incompatibility is not None:
                if self.config.native_dit_acceleration == "required":
                    raise ValueError(incompatibility)
                self._optimized_dit_selection = NativeBackendSelection(
                    component="optimized_dit",
                    mode=self.config.native_dit_acceleration,
                    enabled=False,
                    reason=incompatibility,
                )
                attention_policy.validate_cuda_graph(
                    config=self.config,
                    native_selected=False,
                )
                return
        native_graph_incompatibility = attention_policy.cuda_graph_incompatibility(
            config=self.config,
            native_selected=True,
        )
        if (
            native_graph_incompatibility is not None
            and self.config.native_dit_acceleration == "auto"
        ):
            self._optimized_dit_selection = NativeBackendSelection(
                component="optimized_dit",
                mode=self.config.native_dit_acceleration,
                enabled=False,
                reason=native_graph_incompatibility,
            )
            attention_policy.validate_cuda_graph(
                config=self.config,
                native_selected=False,
            )
            return

        helper = omnidreams_singleview.load_python_module("optimized_dit")
        availability_check = (
            (
                lambda extension: _native_dit_sage2_availability(
                    extension,
                    device=target_device,
                )
            )
            if attention_backend == "sage2"
            else require_extension_symbols("optimized_dit_forward")
        )
        native_config = NativeAccelerationConfig(
            mode=self.config.native_dit_acceleration,
            build_root=self.config.native_dit_build_root,
            max_jobs=self.config.native_dit_max_jobs,
            verbose_build=self.config.native_dit_verbose_build,
        )
        selection = omnidreams_singleview.select_backend(
            "optimized_dit",
            native_config,
            availability_check=availability_check,
        )
        self._optimized_dit_selection = selection
        if not selection.enabled:
            attention_policy.validate_cuda_graph(
                config=self.config,
                native_selected=False,
            )
            return
        attention_policy.validate_cuda_graph(
            config=self.config,
            native_selected=True,
        )
        self._optimized_dit_executor = helper.OptimizedDiTExecutor(
            self,
            selection.require_extension(),
            dit_backend=self.config.native_dit_backend,
            attention_backend=attention_backend,
            sparge_topk=self.config.native_dit_sparge_topk,
            sparge_hybrid_period=self.config.native_dit_sparge_hybrid_period,
            sparge_hybrid_phase=self.config.native_dit_sparge_hybrid_phase,
        )

    def _ensure_optimized_dit_configured_for_target_device(self) -> None:
        """Resolve pending Native Sage2 selection on the model's target device."""
        target_device = self.device
        if not self._native_sage2_selection_pending:
            if (
                self._native_sage2_selection_device is not None
                and target_device != self._native_sage2_selection_device
            ):
                raise RuntimeError(
                    "OmniDreams Native Sage2 selection was finalized for "
                    f"{self._native_sage2_selection_device}, but the model is now "
                    f"on {target_device}. Move the model to its final device before "
                    "initializing the first autoregressive cache, or construct a "
                    "new pipeline."
                )
            return

        if target_device.type != "cuda":
            reason = f"target device {target_device} is not a CUDA device"
            if self.config.native_dit_acceleration == "required":
                raise NativeAccelerationUnavailable(reason)
            self._optimized_dit_selection = NativeBackendSelection(
                component="optimized_dit",
                mode=self.config.native_dit_acceleration,
                enabled=False,
                reason=reason,
            )
            self._attention_policy.validate_cuda_graph(
                config=self.config,
                native_selected=False,
            )
        else:
            with torch.cuda.device(target_device):
                self._configure_optimized_dit_from_config(
                    target_device=target_device,
                )

        if self._optimized_dit_executor is None and self.config.compile_network:
            self.network = compile_module(self.network)
            self._reset_framework_dispatch()
        self._native_sage2_selection_device = target_device
        self._native_sage2_selection_pending = False

    def _reset_framework_dispatch(self) -> None:
        """Point framework dispatch at the current, possibly compiled network."""
        self._cuda_graph_dispatch = CUDAGraphDispatch(
            self.network,
            enabled=self.config.use_cuda_graph,
            capture_ar_idx=self._cuda_graph_capture_ar_idx,
            warmup_iters=self.config.cuda_graph_warmup_iters,
        )
        # Compatibility aliases for native acceleration hooks that predate the
        # shared dispatch helper.
        self._network_call = self._cuda_graph_dispatch.cond_call or self.network
        self._network_call_uncond = (
            self._cuda_graph_dispatch.uncond_call or self.network
        )

    def _native_sage2_config_incompatibility(self) -> str | None:
        """Return why this model configuration cannot use Native Sage2."""

        if self.config.native_dit_backend != "fp8_kvcache_cudnn":
            return (
                "OmniDreams Native SageAttention 2 requires "
                "native_dit_backend='fp8_kvcache_cudnn'."
            )
        head_dim = self.config.network.model_channels // self.config.network.num_heads
        if head_dim not in {64, 128}:
            return (
                "OmniDreams Native SageAttention 2 requires head_dim=64 or 128, "
                f"got {head_dim}."
            )
        return None

    ## Patchify / CP plumbing

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """Per-rank latent shape ``[..., V/cp_V, pT/cp_T, HW/cp_HW, D]``.

        Per-rollout ``(height, width)`` is populated by
        :meth:`initialize_autoregressive_cache`; reading earlier asserts.
        """
        assert self._output_height is not None and self._output_width is not None, (
            "latent_shape requires an initialized rollout; call "
            "initialize_autoregressive_cache(..., height=..., width=...) first."
        )
        cfg = self.config
        kt = cfg.network.patch_temporal
        kh = kw = cfg.network.patch_spatial
        D = cfg.network.in_channels * kt * kh * kw
        pT = cfg.len_t // kt
        pH = self._output_height // kh
        pW = self._output_width // kw
        if self.flatten_thw:
            return (
                *cfg.batch_shape,
                cfg.num_views // self.cp_groups.V_size,
                (pT * pH * pW) // self.cp_groups.THW_size,
                D,
            )
        else:
            return (
                *cfg.batch_shape,
                cfg.num_views // self.cp_groups.V_size,
                pT // self.cp_groups.T_size,
                (pH * pW) // self.cp_groups.HW_size,
                D,
            )

    def patchify_and_maybe_split_cp(self, x: Tensor) -> Tensor:
        # x expected to be [B, V, T, C, H, W]
        assert x.ndim == 6, f"x must be a 6D tensor, but got shape {x.shape}"
        x = x.to(device=self.device, dtype=self.config.dtype)

        if self.flatten_thw:
            return self.network.patchify_and_maybe_split_cp(
                x,
                process_groups=[self.cp_groups.V_group, self.cp_groups.THW_group],
                cp_dims=[-3, -2],
                flatten_thw=True,
            )  # [B, V, L, D]
        else:
            return self.network.patchify_and_maybe_split_cp(
                x,
                process_groups=[
                    self.cp_groups.V_group,
                    self.cp_groups.T_group,
                    self.cp_groups.HW_group,
                ],
                cp_dims=[-4, -3, -2],
                flatten_thw=False,
            )  # [B, V, T, HW, D]

    def unpatchify_and_maybe_gather_cp(self, x: Tensor) -> Tensor:
        assert self._output_height is not None and self._output_width is not None, (
            "unpatchify_and_maybe_gather_cp requires an initialized rollout; "
            "call initialize_autoregressive_cache(..., height=..., width=...) first."
        )
        kh = kw = self.config.network.patch_spatial
        pH = self._output_height // kh
        pW = self._output_width // kw
        if self.flatten_thw:
            # x expected to be [B, V, L, D]
            assert x.ndim == 4, f"x must be a 4D tensor, but got shape {x.shape}"
            return self.network.unpatchify_and_maybe_gather_cp(
                pH=pH,
                pW=pW,
                x=x,
                process_groups=[self.cp_groups.V_group, self.cp_groups.THW_group],
                cp_dims=[-3, -2],
                flatten_thw=True,
            )  # [B, V, T, C, H, W]
        else:
            # x expected to be [B, V, T, HW, D]
            assert x.ndim == 5, f"x must be a 5D tensor, but got shape {x.shape}"
            return self.network.unpatchify_and_maybe_gather_cp(
                pH=pH,
                pW=pW,
                x=x,
                process_groups=[
                    self.cp_groups.V_group,
                    self.cp_groups.T_group,
                    self.cp_groups.HW_group,
                ],
                cp_dims=[-4, -3, -2],
                flatten_thw=False,
            )  # [B, V, T, C, H, W]

    ## Condition / cache plumbing

    @torch.no_grad()
    def initialize_autoregressive_cache(
        self,
        *,
        height: int,
        width: int,
        text_embeddings: Tensor,
        image_embeddings: Tensor,
        negative_text_embeddings: Tensor | None = None,
        view_names: list[str] | None = None,
        **_unused: Any,
    ) -> CosmosTransformerCache:
        """Build a fully seeded cache for a new rollout.

        Args:
            height: Pre-patchify latent height (post-VAE).
            width: Pre-patchify latent width (post-VAE).
            text_embeddings: ``[B, V, L, D]`` text embeddings.
            image_embeddings: ``[B, V, 1, C, H, W]`` first-frame VAE latent.
                ``H``/``W`` must equal ``height``/``width``.
            view_names: Length-``V`` view names; required when
                ``num_views > 1``.
        """
        self._ensure_optimized_dit_configured_for_target_device()

        # Stash per-rollout spatial layout (read by latent_shape,
        # unpatchify_and_maybe_gather_cp, and the network-cache / RoPE setup).
        cfg = self.config
        text_embeddings = text_embeddings.to(device=self.device, dtype=cfg.dtype)
        image_embeddings = image_embeddings.to(device=self.device, dtype=cfg.dtype)
        if negative_text_embeddings is not None:
            negative_text_embeddings = negative_text_embeddings.to(
                device=self.device, dtype=cfg.dtype
            )

        kt = cfg.network.patch_temporal
        kh = kw = cfg.network.patch_spatial
        assert height % kh == 0 and width % kw == 0, (
            f"(height, width) = ({height}, {width}) must be divisible by "
            f"patch_spatial ({kh})."
        )
        self._output_height = height
        self._output_width = width
        pT = cfg.len_t // kt
        pH = height // kh
        pW = width // kw

        if self.cp_groups.V_group is not None:
            text_embeddings = split_inputs_cp(
                text_embeddings, seq_dim=1, cp_group=self.cp_groups.V_group
            )
            if negative_text_embeddings is not None:
                negative_text_embeddings = split_inputs_cp(
                    negative_text_embeddings,
                    seq_dim=1,
                    cp_group=self.cp_groups.V_group,
                )

        head_dim = cfg.network.model_channels // cfg.network.num_heads
        rope_kwargs: dict[str, Any] = {
            "len_t": pT,
            "len_h": pH,
            "len_w": pW,
            "head_dim": head_dim,
            "h_extrapolation_ratio": cfg.h_extrapolation_ratio,
            "w_extrapolation_ratio": cfg.w_extrapolation_ratio,
            "device": self.device,
        }
        if cfg.network.apply_rope_before_kvcache:
            rope_adapter = RotaryPositionEmbedding3D(**rope_kwargs)
        else:
            assert cfg.window_size_t % kt == 0 and cfg.sink_size_t % kt == 0, (
                "Cache-relative RoPE requires window_size_t and sink_size_t "
                f"to be divisible by patch_temporal ({kt})"
            )
            rope_adapter = KVCacheRelativeRotaryPositionEmbedding3D(
                **rope_kwargs,
                window_size_t=cfg.window_size_t // kt,
                sink_size_t=cfg.sink_size_t // kt,
            )
        rope_adapter.set_context_parallel_group(cp_group=self.cp_groups.THW_group)

        num_tokens_per_view_per_step = pH * pW
        if self.cp_groups.THW_group is not None:
            num_tokens_per_view_per_step //= self.cp_groups.THW_group.size()
        network_cache_kwargs: dict[str, Any] = {
            "chunk_size": num_tokens_per_view_per_step * pT,
            "window_size": num_tokens_per_view_per_step * cfg.window_size_t,
            "sink_size": num_tokens_per_view_per_step * cfg.sink_size_t,
        }
        if cfg.early_short_history_block_count is not None:
            network_cache_kwargs["early_short_history_block_count"] = (
                cfg.early_short_history_block_count
            )
        network_cache = self.network.initialize_cache(
            **network_cache_kwargs,
            context=text_embeddings,
        )
        network_cache_uncond: CosmosDiTNetworkCache | None = None
        if cfg.requires_negative_text_embeddings:
            assert negative_text_embeddings is not None, (
                f"{type(cfg).__name__}.guidance_scale={cfg.guidance_scale} > 1.0 "
                "requires negative_text_embeddings."
            )
            network_cache_uncond = self.network.initialize_cache(
                **network_cache_kwargs,
                context=negative_text_embeddings,
            )

        view_indices: Tensor | None = None
        if cfg.network.enable_cross_view_attn:
            assert view_names is not None and len(view_names) == cfg.num_views, (
                f"view_names of length {cfg.num_views} required when "
                f"num_views > 1 (got {view_names})"
            )
            batch_size = image_embeddings.shape[0]
            view_indices = torch.tensor(
                [DEFAULT_CAMERA_VIEW_MAPPING[name] for name in view_names],
                device=self.device,
                dtype=torch.long,
            )
            view_indices = view_indices.repeat(batch_size, 1)
            if self.cp_groups.V_group is not None:
                view_indices = split_inputs_cp(
                    view_indices, seq_dim=1, cp_group=self.cp_groups.V_group
                )

        B, V, _, _, H, W = image_embeddings.shape
        assert H == height and W == width, (
            f"image_embeddings spatial dims ({H}, {W}) must match "
            f"(height, width) = ({height}, {width})."
        )
        mask_first_block = torch.zeros(
            B, V, cfg.len_t, 1, H, W, device=self.device, dtype=cfg.dtype
        )
        mask_first_block[:, :, :1, :, :, :] = 1.0
        mask_other_blocks = torch.zeros(
            B, V, cfg.len_t, 1, H, W, device=self.device, dtype=cfg.dtype
        )

        # Pad first-frame image latent along T (zeros for steady state).
        image = F.pad(image_embeddings, (0, 0, 0, 0, 0, 0, 0, cfg.len_t - 1))

        # Patchify image and masks once at rollout start.
        image_patched = self.patchify_and_maybe_split_cp(image)
        mask_first_patched = self.patchify_and_maybe_split_cp(mask_first_block)
        mask_other_patched = self.patchify_and_maybe_split_cp(mask_other_blocks)

        text_edit_lora = getattr(self, "_text_edit_lora", None)
        if text_edit_lora is not None:
            text_edit_lora.set_active(False)

        if self._use_cuda_graph:
            self._cuda_graph_dispatch.reset()

        cache = CosmosTransformerCache(
            network_cache=network_cache,
            network_cache_uncond=network_cache_uncond,
            rope_adapter=rope_adapter,
            image=image_patched,
            mask_first_block=mask_first_patched,
            mask_other_blocks=mask_other_patched,
            view_indices=view_indices,
        )
        if self._optimized_dit_executor is not None:
            self._optimized_dit_executor.after_initialize_autoregressive_cache(cache)
        return cache

    @torch.no_grad()
    def replace_text_embeddings(
        self,
        cache: CosmosTransformerCache,
        text_embeddings: Tensor,
        *,
        guidance_scale: float = 1.0,
        guidance_chunks: int = 0,
    ) -> None:
        """Replace cached text conditioning while retaining visual history.

        Args:
            cache: Live autoregressive cache.
            text_embeddings: Replacement embeddings ``[B, V, L, D]``.
            guidance_scale: New-minus-old edit strength.
            guidance_chunks: Number of upcoming chunks to guide.
        """
        if self._optimized_dit_executor is not None:
            raise NotImplementedError(
                "Text replacement is not available with native DiT acceleration"
            )
        cfg = self.config
        text_embeddings = text_embeddings.to(device=self.device, dtype=cfg.dtype)
        if self.cp_groups.V_group is not None:
            text_embeddings = split_inputs_cp(
                text_embeddings, seq_dim=1, cp_group=self.cp_groups.V_group
            )
        use_guidance = guidance_scale != 1.0 and guidance_chunks > 0
        if use_guidance and cache.network_cache_uncond is not None:
            raise ValueError(
                "Text-edit guidance cannot be combined with negative-prompt CFG"
            )
        block_caches = cache.network_cache.block_caches
        if use_guidance and self._text_edit_lora is not None:
            self.network.replace_text_embeddings(cache.network_cache, text_embeddings)
            self._text_edit_lora.set_active(True)
            cache.text_edit_guidance = TextEditGuidance(
                scale=guidance_scale,
                chunks_remaining=guidance_chunks,
                use_lora=True,
            )
            return
        old = (
            [block.cross_attn.clone_kv() for block in block_caches]
            if use_guidance
            else None
        )
        self.network.replace_text_embeddings(cache.network_cache, text_embeddings)
        if old is None:
            cache.text_edit_guidance = None
            if self._text_edit_lora is not None:
                self._text_edit_lora.set_active(False)
        else:
            cache.text_edit_guidance = TextEditGuidance(
                scale=guidance_scale,
                chunks_remaining=guidance_chunks,
                kv_old=old,
                kv_new=[block.cross_attn.clone_kv() for block in block_caches],
            )

    ## Mask-injection helpers

    def _maybe_inject_image(
        self,
        latent: Tensor,
        cache: CosmosTransformerCache,
    ) -> Tensor:
        """Replace the first-temporal-frame latent with the encoded image at AR step 0."""
        if cache.autoregressive_index != 0:
            return latent
        mask = cache.mask_first_block[..., :1]
        return latent * (1.0 - mask) + cache.image * mask

    def _select_mask(self, cache: CosmosTransformerCache) -> Tensor:
        return (
            cache.mask_first_block
            if cache.autoregressive_index == 0
            else cache.mask_other_blocks
        )

    ## Forward

    def _select_network(self, autoregressive_index: int, *, uncond: bool) -> Any:
        if not self._use_cuda_graph:
            return self.network

        return self._cuda_graph_dispatch.select(
            autoregressive_index,
            uncond=uncond,
        )

    def _predict_branch(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: CosmosTransformerCache,
        network_cache: CosmosDiTNetworkCache,
        input: Tensor | None,
        *,
        uncond: bool,
    ) -> Tensor:
        ar_idx = cache.autoregressive_index
        assert ar_idx >= 0 and cache.rope_freqs is not None, (
            "Cache.start(autoregressive_index) must be called before "
            "predict_flow (DiffusionModel.generate handles this)."
        )
        noisy_latent = self._maybe_inject_image(noisy_latent, cache)
        return self._select_network(ar_idx, uncond=uncond)(
            noisy_latent,
            timesteps=timestep,
            rope_freqs=cache.rope_freqs,
            cache=network_cache,
            condition_video_input_mask=self._select_mask(cache),
            current_chunk_idx=ar_idx,
            hdmap_condition=input,
            view_indices=cache.view_indices,
            eager_mode=False,
        )

    def predict_flow(
        self,
        noisy_latent: Tensor,
        timestep: Tensor,
        cache: CosmosTransformerCache,
        input: Tensor | None = None,
    ) -> Tensor:
        if self._optimized_dit_executor is not None:
            return self._optimized_dit_executor.predict_flow(
                noisy_latent=noisy_latent,
                timestep=timestep,
                cache=cache,
                input=input,
            )
        guidance = cache.text_edit_guidance
        if guidance is not None and guidance.use_lora:
            assert self._text_edit_lora is not None
            self._text_edit_lora.set_active(True)
        elif self._text_edit_lora is not None and self._text_edit_lora.active:
            self._text_edit_lora.set_active(False)
        if (
            guidance is not None
            and not guidance.use_lora
            and not self._finalizing_kv_cache
            and cache.network_cache_uncond is None
        ):
            blocks = cache.network_cache.block_caches
            for block, (key, value) in zip(blocks, guidance.kv_old, strict=True):
                block.cross_attn.overwrite_kv_(key, value)
            flow_old = self._predict_branch(
                noisy_latent=noisy_latent,
                timestep=timestep,
                cache=cache,
                network_cache=cache.network_cache,
                input=input,
                uncond=False,
            )
            for block, (key, value) in zip(blocks, guidance.kv_new, strict=True):
                block.cross_attn.overwrite_kv_(key, value)
            flow_new = self._predict_branch(
                noisy_latent=noisy_latent,
                timestep=timestep,
                cache=cache,
                network_cache=cache.network_cache,
                input=input,
                uncond=False,
            )
            return flow_old + guidance.scale * (flow_new - flow_old)
        flow_cond = self._predict_branch(
            noisy_latent=noisy_latent,
            timestep=timestep,
            cache=cache,
            network_cache=cache.network_cache,
            input=input,
            uncond=False,
        )
        if cache.network_cache_uncond is None:
            return flow_cond
        flow_uncond = self._predict_branch(
            noisy_latent=noisy_latent,
            timestep=timestep,
            cache=cache,
            network_cache=cache.network_cache_uncond,
            input=input,
            uncond=True,
        )
        return flow_uncond + self.config.guidance_scale * (flow_cond - flow_uncond)

    def postprocess_clean_latent(
        self,
        clean_latent: Tensor,
        cache: CosmosTransformerCache,
        input: Tensor | None = None,
    ) -> Tensor:
        return self._maybe_inject_image(clean_latent, cache)

    def finalize_kv_cache(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        try:
            if not self.config.skip_finalize_kv_cache:
                self._finalizing_kv_cache = True
                try:
                    super().finalize_kv_cache(*args, **kwargs)
                finally:
                    self._finalizing_kv_cache = False
        finally:
            if self._optimized_dit_executor is not None:
                self._optimized_dit_executor.after_finalize_kv_cache()
