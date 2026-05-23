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

"""Phase 2b.6.2 vendor-side dump harness.

Monkey-patches vendor's ``CausalCameraPRopeWanAttnProcessor2_0`` (attention)
and ``WanTransformer3DModel`` (top-level forward) to write the same dump
records as the native HY-WorldPlay code path. Run together with
``run_vendor_use_kv_cache.py`` so the cache-prefill architecture matches
native and the dumps are directly diffable.

Usage::

    HY_DEBUG_DUMP=/tmp/vendor_dump.jsonl USE_KV_CACHE_TRUE=1 \\
        python tests/parity_check/dump_patch_runner.py [generate.py args]

The dump format / call-site names mirror
``hy_worldplay/_debug_dump.py`` so the diff script can match records by
``(name, chunk_idx, step_idx, block_idx)`` keys.

Disabled by default: the patch only installs when ``HY_DEBUG_DUMP`` is a
non-empty env var, so production parity / benchmark runs pay zero
overhead even if this module is imported.
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

    Vendor's attention processor handles BOTH prefill (``is_cache=True``)
    and the main forward (``is_cache=False``) call sites. We dump the
    matched call-site tensors:

    * ``is_cache=True`` -> ``prefill.block.*`` (mirrors native's
      ``HyWorldPlayPRoPESelfAttention.prefill_memory_kv``).
    * ``is_cache=False`` -> ``attn.*`` (mirrors native's
      ``HyWorldPlayPRoPESelfAttention.forward_dual_branch``).

    We dump the inputs (raw Q/K/V, rotary_emb) BEFORE the original call
    runs, and the cache state (cache_key, cache_value) is observable via
    the ``kv_cache`` arg so we dump the pre-concat snapshot too. The
    post-concat / final K is reconstructed from the same monkey-patched
    locals in the wrapper -- it isn't worth a second monkey-patch on
    the attention internals because the wrapper has all inputs needed.
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
        # Capture pre-call snapshot of inputs and cache state.
        if enabled():
            try:
                # Compute Q/K/V via the same path as the attention
                # processor so the dumps reflect the post-norm pre-RoPE
                # tensors that native dumps too.
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
                # Match native's [B, L, H, D] layout for raw dumps (the
                # attention processor immediately transposes to
                # [B, H, L, D]; we dump pre-transpose for parity).
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
                    # Cache state going into this forward (chunk-1+).
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

        # Post-call: capture the K/V written into the cache (prefill) or
        # the (memory + current)-concatenated K (forward). The processor
        # returns ``(hidden_states, kv_cache_return)``. For ``is_cache=True``
        # we read kv_cache_return; for ``is_cache=False`` the concat already
        # happened internally and isn't exposed, so we reconstruct from
        # the cache snapshot above (already dumped) + current K (already
        # dumped) -- the diff script can re-cat them.
        if enabled():
            try:
                _, kv_ret = result
                if is_cache and kv_ret is not None:
                    k_combined = kv_ret.get("k")
                    v_combined = kv_ret.get("v")
                    if k_combined is not None:
                        k_rope, k_prope = k_combined.chunk(2, dim=-1)
                        v_rope, v_prope = v_combined.chunk(2, dim=-1)
                        # Match native layout [B, L, H, D] from [B, H, L, D].
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
    """Wrap ``WanTransformer3DModel.forward`` to bind per-step context.

    Mirrors native's per-step / per-chunk context-binding in
    ``HyWorldPlayWan21Transformer.predict_flow``: dumps the entry stats
    (timestep, current_start/end, is_cache) and sets ``ar_idx``
    inferable from ``current_start // (4 * 880)`` (chunk_size=4 frames *
    880 tokens/frame at vendor's standard resolution).

    The ar_idx tag is derived because vendor's forward signature doesn't
    accept it directly -- the pipeline encodes it implicitly via
    ``current_start`` / ``current_end``. The diff script tolerates
    minor ar_idx misalignments by matching on (name, block_idx, phase)
    primarily.
    """

    def wrapper(self, *args, **kwargs):
        if enabled():
            try:
                timestep = kwargs.get("timestep")
                current_start = kwargs.get("current_start", 0)
                current_end = kwargs.get("current_end", 0)
                is_cache = kwargs.get("is_cache", False)
                # Derive ar_idx + phase from current_start. Tokens per
                # frame = 880 (vendor hardcoded; matches our 704x1280
                # /patch_size=(1,2,2) layout = pph * ppw = 22 * 40). A
                # 4-frame chunk is 3520 tokens, so chunk_i = start // 3520.
                tokens_per_chunk = 4 * 880
                ar_idx = int(current_start) // tokens_per_chunk
                phase = "prefill" if is_cache else "forward"
                set_context(ar_idx=ar_idx, phase=phase)
                dump(
                    "predict_flow.entry",
                    None,
                    timestep_shape=list(timestep.shape) if timestep is not None else None,
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
    """Install monkey-patches on vendor's attention + transformer.

    Idempotent: re-installing the patch (e.g. when the runner imports
    are re-evaluated) wraps the already-wrapped callable, but the
    dumps still produce one record per real call because we set
    ``_INSTALLED`` after the first install and short-circuit. The
    install is a no-op when ``HY_DEBUG_DUMP`` is empty so the
    one-line entry from the runner is safe to leave on.
    """
    if not enabled():
        return
    global _INSTALLED
    if _INSTALLED:
        return

    # Imports deferred until install time so this module can be
    # safely imported even when the vendor tree isn't on sys.path
    # (e.g. during unit tests of this dumper module itself).
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
