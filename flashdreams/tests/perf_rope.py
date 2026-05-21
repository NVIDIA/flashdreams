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

"""A/B benchmark: Triton vs TE ``apply_rotary_pos_emb``.

Production shapes observed in the omnidreams pipeline are listed in
``_PROD_SHAPES`` below. Run with::

    PYTHONPATH=. python tests/perf_rope.py
"""

from __future__ import annotations

import torch

from flashdreams.core.attention.rope_kernel import apply_rotary_pos_emb

try:
    from transformer_engine.pytorch.attention.rope import (
        apply_rotary_pos_emb as te_apply,
    )
except (ImportError, OSError):
    try:
        from transformer_engine.pytorch.attention import (
            apply_rotary_pos_emb as te_apply,
        )
    except (ImportError, OSError):
        te_apply = None


def _bench_ms(fn, *, n_warmup: int = 100, n_iter: int = 2000) -> tuple[float, float]:
    """Return ``(median_ms, p10_ms)`` across ``n_iter`` runs."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iter)]
    for i in range(n_iter):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    median = times[len(times) // 2]
    p10 = times[len(times) // 10]
    return median, p10


_PROD_SHAPES = [
    # (B, S, H, D, interleaved)
    (1, 7040, 16, 128, False),
    (1, 7040, 16, 128, True),
    (1, 7040 * 2, 16, 128, False),
    (1, 7040 * 2, 16, 128, True),
    (1, 7040 * 4, 16, 128, False),
    (1, 7040 * 4, 16, 128, True),
]


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")
    if te_apply is None:
        raise SystemExit("transformer_engine not importable; cannot A/B.")
    # Hoist into a local so the type checker can see the closures
    # below operate on a non-``None`` reference.
    te_call = te_apply
    device = torch.device("cuda")

    header = (
        f"{'shape':<24} {'inter':<6} {'triton_ms':>10} {'p10':>10} "
        f"{'te_ms':>10} {'te_p10':>10} {'tri/te':>8}"
    )
    print(header)
    print("-" * len(header))
    for B, S, H, D, interleaved in _PROD_SHAPES:
        x = torch.randn(B, S, H, D, device=device, dtype=torch.bfloat16)
        # Mirror the ``shift_t`` redundant-copy layout: cat halves for
        # non-interleaved, repeat-interleave odd indices for interleaved.
        raw = torch.randn(S, D // 2, device=device, dtype=torch.float32)
        if interleaved:
            expanded = raw.repeat_interleave(2, dim=-1)
        else:
            expanded = torch.cat([raw, raw], dim=-1)
        freqs = expanded.reshape(S, 1, 1, D)

        triton_x = x.clone()

        def _run_triton():
            apply_rotary_pos_emb(triton_x, freqs, interleaved=interleaved, inplace=True)

        def _run_te():
            te_call(x, freqs, tensor_format="bshd", fused=True, interleaved=interleaved)

        tri_med, tri_p10 = _bench_ms(_run_triton)
        te_med, te_p10 = _bench_ms(_run_te)
        ratio = tri_med / te_med
        print(
            f"({B},{S:>4},{H:>2},{D})  {str(interleaved):<6} "
            f"{tri_med:>10.4f} {tri_p10:>10.4f} "
            f"{te_med:>10.4f} {te_p10:>10.4f} {ratio:>8.2f}"
        )


if __name__ == "__main__":
    main()
