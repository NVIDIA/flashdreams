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

"""Vendor-side tensor dump harness mirroring :mod:`hy_worldplay._debug_dump`.

Monkey-patches vendor's ``CausalCameraPRopeWanAttnProcessor2_0``
(attention) and ``WanTransformer3DModel`` (top-level forward) so
records match native's call-site names + keys
(``(name, chunk_idx, step_idx, block_idx)``) and the diff script can
align them. No-op unless ``HY_DEBUG_DUMP`` is set; pair with
:mod:`run_vendor_use_kv_cache` so the architectures match.
"""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator

import torch
from torch import Tensor

_DUMP_ENV_VAR = "HY_DEBUG_DUMP"
_lock = threading.Lock()
_context: dict[str, Any] = {}


def enabled() -> bool:
    return bool(os.environ.get(_DUMP_ENV_VAR, ""))


def _dump_path() -> str:
    val = os.environ.get(_DUMP_ENV_VAR, "")
    if not val:
        return ""
    if val in {"1", "true", "True", "yes", "on"}:
        return os.path.abspath("hy_debug_dump.jsonl")
    return os.path.abspath(val)


def set_context(**kwargs: Any) -> None:
    with _lock:
        _context.update(kwargs)


def clear_context(*keys: str) -> None:
    with _lock:
        if not keys:
            _context.clear()
        else:
            for k in keys:
                _context.pop(k, None)


@contextmanager
def context(**kwargs: Any) -> Iterator[None]:
    old = {k: _context.get(k) for k in kwargs}
    set_context(**kwargs)
    try:
        yield
    finally:
        with _lock:
            for k, v in old.items():
                if v is None:
                    _context.pop(k, None)
                else:
                    _context[k] = v


def _tensor_stats(t: Tensor) -> dict[str, Any]:
    if not isinstance(t, Tensor):
        return {"non_tensor_repr": repr(t)[:200]}
    t32 = t.detach().float() if t.numel() > 0 else t
    flat = t32.reshape(-1)
    n = flat.numel()
    if n == 0:
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "device": str(t.device),
            "numel": 0,
        }
    sample_n = min(32, n)
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "device": str(t.device),
        "numel": n,
        "abs_mean": float(flat.abs().mean().item()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std().item() if n > 1 else 0.0),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "sample": flat[:sample_n].cpu().tolist(),
    }


def dump(name: str, tensor: Tensor | None, **extra: Any) -> None:
    if not enabled():
        return
    if torch.cuda.is_available():
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            pass
    path = _dump_path()
    if not path:
        return

    with _lock:
        record: dict[str, Any] = {"name": name, **_context}
        if tensor is not None:
            record["tensor"] = _tensor_stats(tensor)
        if extra:
            record.update(extra)
        try:
            line = json.dumps(record, default=str)
        except (TypeError, ValueError) as e:
            record["__json_error"] = str(e)
            line = json.dumps({"name": name, "__error": str(e)})

        try:
            with open(path, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass


def _make_patched_attention(original_call):
    """Return a wrapper around ``CausalCameraPRopeWanAttnProcessor2_0.__call__``.

    Vendor's processor handles both prefill (``is_cache=True``,
    matched to native's ``prefill.block.*``) and the main forward
    (``is_cache=False``, matched to ``attn.*``). The wrapper dumps
    Q/K/V + rotary inputs before delegating, and the cache K/V
    after, so the diff script can re-cat memory + current K.
    """

    def wrapper(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        rotary_emb=None,
        kv_cache=None,
        is_cache=False,
        idx=None,
        viewmats=None,
        Ks=None,
        context_frames_list=None,
    ):
        # Pre-call snapshot of inputs + cache state.
        if enabled():
            try:
                # Mirror the processor's own path so dumps are
                # post-norm pre-RoPE (matches native's dump points).
                q_raw_v = attn.to_q(hidden_states)
                k_raw_v = attn.to_k(
                    encoder_hidden_states
                    if encoder_hidden_states is not None
                    else hidden_states
                )
                v_raw_v = attn.to_v(
                    encoder_hidden_states
                    if encoder_hidden_states is not None
                    else hidden_states
                )
                if attn.norm_q is not None:
                    q_raw_v = attn.norm_q(q_raw_v)
                if attn.norm_k is not None:
                    k_raw_v = attn.norm_k(k_raw_v)
                # Dump as ``[B, L, H, D]`` (native's layout); the
                # processor itself transposes to ``[B, H, L, D]``.
                q_btlhd = q_raw_v.unflatten(2, (attn.heads, -1))
                k_btlhd = k_raw_v.unflatten(2, (attn.heads, -1))
                v_btlhd = v_raw_v.unflatten(2, (attn.heads, -1))

                phase = "prefill" if is_cache else "forward"
                set_context(block_idx=idx, phase=phase)
                if is_cache:
                    dump("prefill.block.x_in", hidden_states)
                    dump("prefill.block.q_raw", q_btlhd)
                    dump("prefill.block.k_raw", k_btlhd)
                    dump("prefill.block.v_raw", v_btlhd)
                    if rotary_emb is not None:
                        dump(
                            "prefill.block.rope_freqs",
                            rotary_emb[0]
                            if isinstance(rotary_emb, (tuple, list))
                            else rotary_emb,
                        )
                else:
                    dump("attn.x_in", hidden_states)
                    dump("attn.q_raw", q_btlhd)
                    dump("attn.k_raw", k_btlhd)
                    dump("attn.v_raw", v_btlhd)
                    if rotary_emb is not None:
                        dump(
                            "attn.rope_freqs_full",
                            rotary_emb[0]
                            if isinstance(rotary_emb, (tuple, list))
                            else rotary_emb,
                        )
                    # Cache state entering this forward (chunk-1+).
                    if kv_cache is not None:
                        cache_key = kv_cache.get("k") if kv_cache else None
                        cache_value = kv_cache.get("v") if kv_cache else None
                        if cache_key is not None and not is_cache:
                            cache_key_rope, _ = cache_key.chunk(2, dim=-1)
                            cache_value_rope, _ = cache_value.chunk(2, dim=-1)
                            dump("attn.memory_k_rope_prepend", cache_key_rope)
                            dump("attn.memory_v_rope_prepend", cache_value_rope)
            except Exception as exc:
                dump("attn.dump_error", None, error=repr(exc))

        # Run the real attention processor.
        result = original_call(
            self,
            attn=attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            rotary_emb=rotary_emb,
            kv_cache=kv_cache,
            is_cache=is_cache,
            idx=idx,
            viewmats=viewmats,
            Ks=Ks,
            context_frames_list=context_frames_list,
        )

        # Post-call: capture K/V written into the cache (prefill); the
        # forward path's concat is internal, so the diff script
        # re-cats memory + current K from the pre-call dumps.
        if enabled():
            try:
                _, kv_ret = result
                if is_cache and kv_ret is not None:
                    k_combined = kv_ret.get("k")
                    v_combined = kv_ret.get("v")
                    if k_combined is not None:
                        k_rope, k_prope = k_combined.chunk(2, dim=-1)
                        v_rope, v_prope = v_combined.chunk(2, dim=-1)
                        # Transpose [B, H, L, D] -> [B, L, H, D] for parity.
                        dump(
                            "prefill.block.k_rope_written",
                            k_rope.transpose(1, 2).contiguous(),
                        )
                        dump(
                            "prefill.block.v_rope_written",
                            v_rope.transpose(1, 2).contiguous(),
                        )
                        dump(
                            "prefill.block.k_prope_written",
                            k_prope.transpose(1, 2).contiguous(),
                        )
                        dump(
                            "prefill.block.v_prope_written",
                            v_prope.transpose(1, 2).contiguous(),
                        )
            except Exception as exc:
                dump("attn.post_dump_error", None, error=repr(exc))

        return result

    return wrapper


def _make_patched_transformer_forward(original_forward):
    """Wrap ``WanTransformer3DModel.forward`` to bind per-step dump context.

    Derives ``ar_idx`` from ``current_start`` because vendor's forward
    signature doesn't carry it explicitly. The diff script primarily
    matches on ``(name, block_idx, phase)``, so minor ar_idx slippage
    is tolerated.
    """

    def wrapper(self, *args, **kwargs):
        if enabled():
            try:
                timestep = kwargs.get("timestep")
                current_start = kwargs.get("current_start", 0)
                current_end = kwargs.get("current_end", 0)
                is_cache = kwargs.get("is_cache", False)
                # Tokens per frame = 880 (vendor hardcoded; matches
                # 704x1280 / patch_size=(1,2,2) = 22 * 40); a 4-frame
                # chunk is 3520 tokens.
                tokens_per_chunk = 4 * 880
                ar_idx = int(current_start) // tokens_per_chunk
                phase = "prefill" if is_cache else "forward"
                set_context(ar_idx=ar_idx, phase=phase)
                dump(
                    "predict_flow.entry",
                    None,
                    timestep_shape=list(timestep.shape)
                    if timestep is not None
                    else None,
                    is_cache=bool(is_cache),
                    current_start=int(current_start),
                    current_end=int(current_end),
                )
                if timestep is not None:
                    dump("predict_flow.timestep", timestep)
                hidden_states = kwargs.get("hidden_states")
                if hidden_states is not None:
                    dump("predict_flow.noisy_latent", hidden_states)
            except Exception as exc:
                dump("predict_flow.dump_error", None, error=repr(exc))

        return original_forward(self, *args, **kwargs)

    return wrapper


def install_patches() -> None:
    """Install monkey-patches on vendor's attention + transformer modules.

    Idempotent (the ``_INSTALLED`` latch short-circuits) and gated on
    ``HY_DEBUG_DUMP`` so the caller can leave the install line on
    without side effects.
    """
    if not enabled():
        return
    global _INSTALLED
    if _INSTALLED:
        return

    # Lazy import: keeps this module importable without the vendor
    # tree on ``sys.path``.
    from wan.models.dits import arwan_w_action_w_mem_relative_rope as vendor_mod

    proc_cls = vendor_mod.CausalCameraPRopeWanAttnProcessor2_0
    original_call = proc_cls.__call__
    proc_cls.__call__ = _make_patched_attention(original_call)

    transformer_cls = vendor_mod.WanTransformer3DModel
    original_forward = transformer_cls.forward
    transformer_cls.forward = _make_patched_transformer_forward(original_forward)

    _INSTALLED = True
    print(
        f"[dump_patch] installed on {proc_cls.__name__} + "
        f"{transformer_cls.__name__}; dumps -> {_dump_path()}",
        flush=True,
    )


_INSTALLED: bool = False
