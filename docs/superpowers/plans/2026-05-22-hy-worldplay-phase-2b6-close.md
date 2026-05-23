<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# HY-WorldPlay Phase 2b.6 Close Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close phase 2b.6 by re-baselining the vendor parity reference against `use_kv_cache=True` (Option C in the design spec) and landing the long-deferred cleanup (parity sub-venv removal + `--use-native-pipeline` default flip).

**Architecture:** Native runner's `predict_flow` already mirrors vendor's `use_kv_cache=True` cache-prefill path. Add a small Python helper that runtime-monkey-patches vendor's `WanPipeline.__setattr__` to coerce `use_kv_cache=True`, wire a flag in `run.sh` to invoke it, regenerate the vendor MP4, diff against native, then land the cleanup if parity holds. The KV-prefill executor built across 2b.5b-part1/part2 stays in place.

**Tech Stack:** Python 3.12, `uv`, `pytest`, `torch`, `diffusers`, `imageio[ffmpeg]`, vendor's `wan/generate.py` (cloned at `integrations/hy_worldplay/tests/parity_check/HY-WorldPlay/`), bash, `runpy` (stdlib) for delegating to vendor's `__main__`-only script.

---

## File structure

| Layer | File | Purpose | Status |
|---|---|---|---|
| Parity harness | `integrations/hy_worldplay/tests/parity_check/run_vendor_use_kv_cache.py` | New helper. Subclass `WanPipeline` with `__setattr__` override that coerces `use_kv_cache=True`; rebind the module-level reference; `runpy.run_path` into vendor's `wan/generate.py`. | Create |
| Parity harness | `integrations/hy_worldplay/tests/parity_check/run.sh` | Add `USE_KV_CACHE_TRUE=1` env-var branch that swaps the `wan/generate.py` invocation for the new helper. | Modify |
| Parity harness | `integrations/hy_worldplay/tests/parity_check/README.md` | Document the `USE_KV_CACHE_TRUE=1` mode (when to use it, what baseline it produces). | Modify |
| Tests | `integrations/hy_worldplay/tests/parity_check/test_run_vendor_use_kv_cache.py` | New CPU test. Pin the subclass-`__setattr__` coercion via a tiny stub class that mimics `WanPipeline`'s relevant surface. | Create |
| Tests | `integrations/hy_worldplay/tests/test_runner_config.py` (existing) | Assert the new default `use_native_pipeline=True` after the cleanup flip. | Modify |
| Runner config | `integrations/hy_worldplay/hy_worldplay/config.py` (or wherever `HyWorldPlayWanI2VRunnerConfig` lives) | Flip `use_native_pipeline=True` default. **Gated on parity holding.** | Modify |
| Sub-venv | `integrations/hy_worldplay/tests/parity_check/pyproject.toml`, `uv.lock` | Drop `sageattention`, `cloudpickle`, `accelerate`, `transformers==4.57.6` heavy deps. **Gated on parity holding.** | Modify |
| Docs | `integrations/hy_worldplay/README.md` | Mark 2b.6 closed; promote native invocation as documented default; record final parity number. | Modify |
| Docs | `docs/superpowers/specs/2026-05-20-hy-worldplay-phase-2b-design.md` | Update 2b.6 row + success criteria to reflect close + final parity number. | Modify |

---

## Phase 1: Parity validation (Tasks 1-4)

### Task 1: Add the runtime monkey-patch helper

**Files:**
- Create: `integrations/hy_worldplay/tests/parity_check/run_vendor_use_kv_cache.py`
- Test: `integrations/hy_worldplay/tests/parity_check/test_run_vendor_use_kv_cache.py`

- [ ] **Step 1: Write the failing test**

Save to `integrations/hy_worldplay/tests/parity_check/test_run_vendor_use_kv_cache.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the use_kv_cache=True monkey-patch helper.

The helper at run_vendor_use_kv_cache.py is GPU-only at runtime (it
delegates to vendor's wan/generate.py via runpy), but the
__setattr__ coercion that forces use_kv_cache=True is a pure-Python
class transformation and testable on CPU.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HELPER_PATH = (
    Path(__file__).parent / "run_vendor_use_kv_cache.py"
).resolve()


def _load_helper_module():
    """Import the helper script without executing its __main__ block."""
    spec = importlib.util.spec_from_file_location(
        "hy_worldplay_use_kv_cache_helper", HELPER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_make_subclass_coerces_use_kv_cache_to_true() -> None:
    """The subclass factory intercepts `use_kv_cache=False` assignments.

    We mimic vendor's WanPipeline with a tiny stand-in class so the
    test stays CPU-only and doesn't require the vendor tree to be
    present. The helper's `make_use_kv_cache_true_subclass` is a
    pure transformation: pass any class, get back a subclass whose
    __setattr__ coerces the use_kv_cache attribute to True.
    """
    helper = _load_helper_module()

    class FakeWanPipeline:
        """Stand-in for vendor's WanPipeline."""

        def __init__(self) -> None:
            self.use_kv_cache = True  # init True; predict() reassigns to False

        def predict(self) -> None:
            # Mirrors vendor pipeline_wan_w_mem_relative_rope.py line 707.
            self.use_kv_cache = False

    Patched = helper.make_use_kv_cache_true_subclass(FakeWanPipeline)
    instance = Patched()
    assert instance.use_kv_cache is True
    instance.predict()
    # The False assignment inside predict() is intercepted and coerced.
    assert instance.use_kv_cache is True


def test_make_subclass_preserves_other_attributes() -> None:
    """Only `use_kv_cache` is coerced; other attributes pass through."""
    helper = _load_helper_module()

    class FakeWanPipeline:
        pass

    Patched = helper.make_use_kv_cache_true_subclass(FakeWanPipeline)
    instance = Patched()
    instance.some_other_attr = "hello"
    instance.use_kv_cache = False
    assert instance.some_other_attr == "hello"
    assert instance.use_kv_cache is True


def test_make_subclass_idempotent() -> None:
    """Applying the transform twice doesn't double-wrap or break MRO."""
    helper = _load_helper_module()

    class FakeWanPipeline:
        pass

    OncePatched = helper.make_use_kv_cache_true_subclass(FakeWanPipeline)
    TwicePatched = helper.make_use_kv_cache_true_subclass(OncePatched)
    instance = TwicePatched()
    instance.use_kv_cache = False
    assert instance.use_kv_cache is True
```

- [ ] **Step 2: Run test to verify it fails (helper doesn't exist yet)**

```bash
cd /devwork/flashdreams
.venv/bin/python -m pytest \
    integrations/hy_worldplay/tests/parity_check/test_run_vendor_use_kv_cache.py \
    -v
```

Expected: FAIL with `FileNotFoundError` or `ImportError` because `run_vendor_use_kv_cache.py` doesn't exist yet.

- [ ] **Step 3: Implement the helper**

Save to `integrations/hy_worldplay/tests/parity_check/run_vendor_use_kv_cache.py`:

```python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run vendor's wan/generate.py with WanPipeline.use_kv_cache forced True.

Phase 2b.6 close path (Option C in the phase-2b design spec).

Mirrors the invocation in run.sh's [run] section (torchrun
wan/generate.py ...) but with a runtime monkey-patch that intercepts
WanPipeline.__setattr__ to coerce use_kv_cache=True. This lets us
diff the native HY runner against vendor's cache-prefill code path
(the use_kv_cache=True branch in vendor's pipeline) instead of the
default single-forward-pass branch.

The default use_kv_cache=False assignment lives inside
WanPipeline.predict at line 707 of pipeline_wan_w_mem_relative_rope.py
(mid-execution), so we can't simply set the attribute before
predict() runs. Instead we subclass WanPipeline with an __setattr__
that maps any use_kv_cache assignment to True; replace the
WanPipeline reference in the module's namespace BEFORE
generate.py's `from ... import WanPipeline` resolves; then use
runpy to execute generate.py with the patched class in place.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Type, TypeVar

_T = TypeVar("_T")

_SCRIPT_DIR = Path(__file__).parent
_REPO_DIR = _SCRIPT_DIR / "HY-WorldPlay"


def make_use_kv_cache_true_subclass(base: Type[_T]) -> Type[_T]:
    """Return a subclass of ``base`` that coerces ``use_kv_cache`` to True.

    Vendor's :class:`WanPipeline.predict` hardcodes
    ``self.use_kv_cache = False`` mid-method (line 707 of
    ``pipeline_wan_w_mem_relative_rope.py``); we intercept that
    assignment so the predict body takes the use_kv_cache=True branch
    (cache-prefill + chunk-1-only forward), which is the same
    architecture the native HY-WorldPlay runner already uses.

    Idempotent: applying the transform twice produces a deeper subclass
    chain but every level still coerces correctly through the inherited
    ``__setattr__``.
    """

    class _UseKvCacheTrue(base):  # type: ignore[valid-type, misc]
        def __setattr__(self, name: str, value: object) -> None:
            if name == "use_kv_cache":
                value = True
            super().__setattr__(name, value)

    _UseKvCacheTrue.__name__ = f"_UseKvCacheTrue_{base.__name__}"
    _UseKvCacheTrue.__qualname__ = _UseKvCacheTrue.__name__
    return _UseKvCacheTrue


def _patch_and_run() -> None:
    """Patch the WanPipeline module binding, then delegate to vendor's generate.py."""
    if not _REPO_DIR.exists():
        raise FileNotFoundError(
            f"Vendor tree not found at {_REPO_DIR}. Run "
            f"`bash {_SCRIPT_DIR / 'run.sh'}` once to clone."
        )
    sys.path.insert(0, str(_REPO_DIR))
    sys.path.insert(0, str(_REPO_DIR / "wan"))

    from wan.inference import (  # noqa: WPS433 (deferred import after sys.path)
        pipeline_wan_w_mem_relative_rope as _vendor_pipe_mod,
    )

    _vendor_pipe_mod.WanPipeline = make_use_kv_cache_true_subclass(
        _vendor_pipe_mod.WanPipeline
    )

    runpy.run_path(
        str(_REPO_DIR / "wan" / "generate.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    _patch_and_run()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /devwork/flashdreams
.venv/bin/python -m pytest \
    integrations/hy_worldplay/tests/parity_check/test_run_vendor_use_kv_cache.py \
    -v
```

Expected: PASS for all three tests (coerce_to_true, preserves_other_attributes, idempotent).

- [ ] **Step 5: Run the full HY-WorldPlay CPU test suite (regression bar)**

```bash
cd /devwork/flashdreams
.venv/bin/python -m pytest integrations/hy_worldplay/tests/ -v --tb=short 2>&1 | tail -20
```

Expected: All 99+ existing tests pass + 3 new ones from this task. Final summary line should show no failures.

- [ ] **Step 6: Lint**

```bash
cd /devwork/flashdreams
.venv/bin/python -m ruff check \
    integrations/hy_worldplay/tests/parity_check/run_vendor_use_kv_cache.py \
    integrations/hy_worldplay/tests/parity_check/test_run_vendor_use_kv_cache.py
```

Expected: PASS (no findings). If `ruff` isn't installed in the venv, fall back to the in-IDE ReadLints check; both should be clean.

- [ ] **Step 7: Commit**

```bash
cd /devwork/flashdreams
git add integrations/hy_worldplay/tests/parity_check/run_vendor_use_kv_cache.py \
        integrations/hy_worldplay/tests/parity_check/test_run_vendor_use_kv_cache.py
git commit -m "$(cat <<'EOF'
feat(hy_worldplay): add use_kv_cache=True parity helper (phase 2b.6 close prep)

Adds run_vendor_use_kv_cache.py: a small runtime monkey-patch that
subclasses vendor's WanPipeline with an __setattr__ override mapping
use_kv_cache=False assignments to True, rebinds the module-level
reference, and runpy-delegates into vendor's wan/generate.py. The
helper lets us re-baseline the vendor parity reference against the
use_kv_cache=True code path (cache-prefill + chunk-1-only forward),
which is the architecture the native HY-WorldPlay runner already
mirrors -- so a parity diff against this baseline is the right
acceptance gate for phase 2b.6 (Option C in the design spec).

The pure-Python __setattr__ coercion is CPU-testable via a tiny
WanPipeline stand-in; three new tests pin coerce_to_true,
preserves_other_attributes, and idempotency. GPU execution lands
in subsequent tasks once run.sh's flag wiring is in place.
EOF
)"
```

---

### Task 2: Wire the `USE_KV_CACHE_TRUE=1` env-var branch into `run.sh`

**Files:**
- Modify: `integrations/hy_worldplay/tests/parity_check/run.sh`
- Modify: `integrations/hy_worldplay/tests/parity_check/README.md`

- [ ] **Step 1: Modify run.sh**

Replace the `[run]` block at the bottom of `run.sh` (currently lines ~125-145) with:

```bash
# ----------------------------------------------------------------- benchmark
# Mirrors the upstream invocation in ``HY-WorldPlay/wan/README.md``:
#   PYTHONPATH=$(pwd):$(pwd)/wan torchrun --nproc_per_node=NUM_GPU \
#       wan/generate.py --input "..." --image_path ... \
#       --num_chunk N --pose ... \
#       --ar_model_path .../wan_transformer \
#       --ckpt_path .../wan_distilled_model/model.pt \
#       --out outputs
#
# Set ``USE_KV_CACHE_TRUE=1`` to swap in run_vendor_use_kv_cache.py:
# a runtime monkey-patch that coerces WanPipeline.use_kv_cache=True
# inside predict(). This re-baselines the vendor reference against
# the cache-prefill code path the native HY runner mirrors (phase
# 2b.6 close path; see docs/superpowers/specs/.../phase-2b-design.md).
export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/wan:${PYTHONPATH:-}"

if [[ "${USE_KV_CACHE_TRUE:-0}" == "1" ]]; then
    GENERATE_SCRIPT="${SCRIPT_DIR}/run_vendor_use_kv_cache.py"
    echo "[run] USE_KV_CACHE_TRUE=1 -> wrapping wan/generate.py with ${GENERATE_SCRIPT}"
else
    GENERATE_SCRIPT="${REPO_DIR}/wan/generate.py"
fi

echo "[run] starting upstream WAN-5B benchmark [${NUM_GPU} GPU(s), num_chunk=${NUM_CHUNK}, pose=${POSE}]"
uv run torchrun --nproc_per_node="${NUM_GPU}" "${GENERATE_SCRIPT}" \
    --input "${PROMPT}" \
    --image_path "${IMAGE_PATH}" \
    --num_chunk "${NUM_CHUNK}" \
    --pose "${POSE}" \
    --ar_model_path "${HF_MODELS_DIR}/wan_transformer" \
    --ckpt_path "${HF_MODELS_DIR}/wan_distilled_model/model.pt" \
    --out "${OUTPUT_DIR}"

echo "[run] done; outputs under ${OUTPUT_DIR}"
```

- [ ] **Step 2: Smoke-test the bash routing (no GPU required)**

```bash
cd /devwork/flashdreams/integrations/hy_worldplay/tests/parity_check
# Dry-run by setting all required env vars and intercepting `uv run`:
USE_KV_CACHE_TRUE=1 \
    bash -n run.sh
echo "syntax-only check exit: $?"
```

Expected: exit 0 (bash `-n` is a syntax-only parse; no GPU touched).

- [ ] **Step 3: Update parity_check README**

Append to `integrations/hy_worldplay/tests/parity_check/README.md` (after the existing run instructions):

```markdown
## Re-baselining against vendor's `use_kv_cache=True` code path

Phase 2b.6 closes by validating native parity against vendor's
cache-prefill code path (`use_kv_cache=True`) rather than the
single-forward-pass default. Set `USE_KV_CACHE_TRUE=1` to swap the
default `wan/generate.py` invocation for `run_vendor_use_kv_cache.py`,
which runtime-monkey-patches `WanPipeline.__setattr__` to coerce
`use_kv_cache=True`:

```bash
USE_KV_CACHE_TRUE=1 \
    NUM_CHUNK=2 POSE=w-8 SEED=0 \
    bash integrations/hy_worldplay/tests/parity_check/run.sh
```

The output MP4 lands in `${OUTPUT_DIR}` (default
`HY-WorldPlay/outputs/parity/`). Diff against the native HY runner's
output via `tmp/hy_parity_diff.py` to confirm `mean |Δ| ≤ 5 / 255`.

This mode is the **2b.6 acceptance baseline**. The default
(no-env-var) mode keeps producing the phase-1 `use_kv_cache=False`
baseline so older parity numbers remain comparable.
```

(Adjust the heredoc nesting as needed when actually editing — the inner code fence may need to be escaped or use 4-space indentation depending on the README's existing style.)

- [ ] **Step 4: Commit**

```bash
cd /devwork/flashdreams
git add integrations/hy_worldplay/tests/parity_check/run.sh \
        integrations/hy_worldplay/tests/parity_check/README.md
git commit -m "$(cat <<'EOF'
feat(hy_worldplay): wire USE_KV_CACHE_TRUE=1 into parity_check/run.sh

Adds the env-var-gated branch that swaps the default wan/generate.py
invocation for the new run_vendor_use_kv_cache.py helper. Mode is
opt-in (default behaviour unchanged: the existing
use_kv_cache=False baseline still reproduces phase-1's parity
numbers).

Updates the parity_check README with the new invocation + when to
use it. The phase 2b.6 acceptance gate is `mean |Δ| ≤ 5 / 255`
against this re-baselined vendor reference.
EOF
)"
```

---

### Task 3: Regenerate baselines + diff native against the new vendor reference (GPU)

**Files:**
- Read-only: `integrations/hy_worldplay/tests/parity_check/run.sh`, `tmp/hy_parity_diff.py`
- Outputs: `outputs/parity/vendor_use_kv_cache_true.mp4`, `outputs/parity/native_phase_2b6.mp4`, `outputs/parity/diff_summary.json`

- [ ] **Step 1: Confirm GPU + checkpoint availability**

```bash
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ls -la /devwork/flashdreams/integrations/hy_worldplay/tests/parity_check/HY-WorldPlay/hf_models/wan_transformer/config.json
ls -la /devwork/flashdreams/integrations/hy_worldplay/tests/parity_check/HY-WorldPlay/hf_models/wan_distilled_model/model.pt
```

Expected: at least one GPU with ≥40 GB RAM listed; both checkpoint files exist. If a file is missing, run `bash integrations/hy_worldplay/tests/parity_check/run.sh` once with default settings to trigger the HF download (~52 GiB).

- [ ] **Step 2: Generate the new vendor baseline (use_kv_cache=True)**

```bash
cd /devwork/flashdreams/integrations/hy_worldplay/tests/parity_check

# Mirror the production parity config: 704x1280, num_chunk=2, seed=0.
# pose=w-8 produces 9 keys; vendor consumes the first num_chunk*CHUNK_SIZE=8.
USE_KV_CACHE_TRUE=1 \
    NUM_CHUNK=2 \
    POSE="w-8" \
    SEED=0 \
    OUTPUT_DIR="${PWD}/HY-WorldPlay/outputs/parity_use_kv_cache_true" \
    bash run.sh 2>&1 | tee /tmp/vendor_use_kv_cache_true.log
```

Expected: completes without CUDA errors. The output MP4 should land at `HY-WorldPlay/outputs/parity_use_kv_cache_true/<sanitized-prompt>.mp4`. Find it with:

```bash
ls -la HY-WorldPlay/outputs/parity_use_kv_cache_true/*.mp4
```

If the log contains `use_kv_cache=True` debug prints (vendor's pipeline doesn't print this by default, but the helper can be amended to confirm), good. Otherwise, eyeball the log for chunk-loop progress and confirm the file is non-empty.

Save the MP4 path as a shell variable for the diff step:

```bash
VENDOR_NEW_MP4="$(ls HY-WorldPlay/outputs/parity_use_kv_cache_true/*.mp4 | head -1)"
echo "${VENDOR_NEW_MP4}"
```

- [ ] **Step 3: Generate the native baseline at matching config**

```bash
cd /devwork/flashdreams

# Run from the parity sub-venv since the native path is still gated on
# the heavy deps until the cleanup commit lands (phase 2b.6 step 6+).
# pose=w-7 produces 8 keys natively (matches vendor's 8 consumed keys).
uv run --project integrations/hy_worldplay/tests/parity_check \
    flashdreams-run hy-worldplay-wan-i2v-5b \
    --use-native-pipeline \
    --use-action-conditioning \
    --use-camera-conditioning \
    --use-memory-selection \
    --image-path integrations/hy_worldplay/tests/parity_check/HY-WorldPlay/assets/img/test.png \
    --ar-model-path integrations/hy_worldplay/tests/parity_check/HY-WorldPlay/hf_models/wan_transformer \
    --ckpt-path integrations/hy_worldplay/tests/parity_check/HY-WorldPlay/hf_models/wan_distilled_model/model.pt \
    --hy-worldplay-repo-root integrations/hy_worldplay/tests/parity_check/HY-WorldPlay \
    --num-chunk 2 \
    --pose "w-7" \
    --seed 0 \
    --output-dir outputs/parity_native_2b6 2>&1 | tee /tmp/native_2b6.log

NATIVE_MP4="$(ls outputs/parity_native_2b6/*.mp4 | head -1)"
echo "${NATIVE_MP4}"
```

Expected: completes without CUDA errors; MP4 lands under `outputs/parity_native_2b6/`. Adjust the flag names if the actual CLI surface differs (cross-check with `--help`).

- [ ] **Step 4: Diff the two MP4s**

```bash
cd /devwork/flashdreams
.venv/bin/python tmp/hy_parity_diff.py \
    --vendor-mp4 "${VENDOR_NEW_MP4}" \
    --native-mp4 "${NATIVE_MP4}" \
    --json-out outputs/parity_native_2b6/diff_summary.json 2>&1 | tee /tmp/diff_2b6.log
```

If `tmp/hy_parity_diff.py` doesn't accept those flags exactly (it was written ad-hoc during the 2b.5b-part2-followup parity attempt), invoke it the way the prior diff did and write the summary by hand or amend the script. The key numeric outputs we need are:

- `mean |Δ|` per channel (R / G / B / overall)
- `PSNR`
- per-chunk breakdown (chunk-0 frames 0-12, chunk-1 frames 13-28)

Expected: `mean |Δ|` overall ≤ 5 / 255. If not, see Step 5.

- [ ] **Step 5: Decision gate**

Compare diff result against the acceptance bar:

| Result | Next step |
|---|---|
| `mean |Δ| ≤ 5 / 255` overall | Proceed to Phase 2 (Task 4 onwards). Parity is closed against the cache-prefill baseline. |
| `mean |Δ| ≤ 5 / 255` for chunk-0 but > 5 / 255 for chunk-1 | Likely a remaining implementation bug (per-rollout binding, memory_frame_indices semantics under cache-prefill). Diagnose: dump chunk-1's `memory_frame_indices` from both vendor and native; check that `viewmats` / `Ks` / `action` slicing matches. Land a focused fix as a separate sub-task before Task 4. |
| `mean |Δ| > 5 / 255` for chunk-0 (which already lands at 7-20 from the prior diff) | Unexpected regression. Bisect against the bf8a4ff commit; the three 2b.6 bug-fixes should not regress chunk-0 below the pre-fix state. Re-run the diff after verifying the fixed-bug code is on `HEAD`. |
| Vendor `use_kv_cache=True` run itself crashes or produces visually-broken output | Vendor's cache-prefill mode has its own bug. Fall back: escalate to 2b.6.1 (Option A refactor). Update the spec to reflect this; this task ends with `[ ] Step 5b: spec update + commit + halt Phase 2`. |

- [ ] **Step 6: Commit the diff result (in either branch)**

If parity holds (proceed to Phase 2):

```bash
cd /devwork/flashdreams
git add outputs/parity_native_2b6/diff_summary.json
git commit -m "$(cat <<'EOF'
chore(hy_worldplay): record 2b.6 close diff against use_kv_cache=True baseline

mean |Δ| = <FILL IN>/255 against the new vendor reference (option C
from the phase-2b design spec). Chunk-0 and chunk-1 breakdowns
attached. Parity bar (≤5/255) met; cleanup follows.
EOF
)"
```

If parity doesn't hold (halt or escalate per Step 5):

```bash
cd /devwork/flashdreams
git add outputs/parity_native_2b6/diff_summary.json \
        docs/superpowers/specs/2026-05-20-hy-worldplay-phase-2b-design.md
git commit -m "$(cat <<'EOF'
docs(hy_worldplay): record 2b.6 parity diff + escalate to 2b.6.1

mean |Δ| = <FILL IN>/255 against the use_kv_cache=True vendor
baseline. <One-sentence diagnosis>. Updated the design spec to
move 2b.6.1 (Option A refactor) from "future; not currently planned"
to "in progress" and parked Phase 2 cleanup.
EOF
)"
```

---

## Phase 2: Cleanup (Tasks 4-7; gated on Phase 1 passing)

> **Gate:** Only proceed past this line if Task 3 Step 5 selected the "Proceed to Phase 2" branch.

### Task 4: Flip `use_native_pipeline=True` as the default

**Files:**
- Modify: `integrations/hy_worldplay/hy_worldplay/config.py` (or the file that defines `HyWorldPlayWanI2VRunnerConfig`'s default — cross-check via `grep -n use_native_pipeline integrations/hy_worldplay/hy_worldplay/*.py`)
- Modify: `integrations/hy_worldplay/tests/test_runner_config.py` (or whichever existing test covers the default; grep for `use_native_pipeline` in `tests/`)

- [ ] **Step 1: Locate the current default**

```bash
cd /devwork/flashdreams
grep -rn "use_native_pipeline" integrations/hy_worldplay/hy_worldplay/ \
                              integrations/hy_worldplay/tests/
```

Note the file + line of the dataclass field default. Likely in `config.py`, a line like `use_native_pipeline: bool = False`.

- [ ] **Step 2: Write a failing test asserting the new default**

Find the existing runner-config test that asserts current defaults (likely `test_runner_config.py` or `test_smoke.py`); locate the assertion `assert cfg.use_native_pipeline is False` (or equivalent) and replace it with:

```python
def test_default_uses_native_pipeline() -> None:
    """Phase 2b.6 closing step: native pipeline is the default."""
    from hy_worldplay.config import RUNNER_HY_WORLDPLAY_WAN_I2V_5B

    assert RUNNER_HY_WORLDPLAY_WAN_I2V_5B.use_native_pipeline is True
```

If there's no existing test, add this one to `integrations/hy_worldplay/tests/test_runner_config.py`.

- [ ] **Step 3: Run test to verify it fails**

```bash
cd /devwork/flashdreams
.venv/bin/python -m pytest \
    integrations/hy_worldplay/tests/test_runner_config.py::test_default_uses_native_pipeline \
    -v
```

Expected: FAIL (current default is `False`).

- [ ] **Step 4: Flip the default**

In the dataclass field located in Step 1, change `False` to `True`:

```python
use_native_pipeline: bool = True
```

If the field is set somewhere other than at dataclass declaration (e.g. `__post_init__` override, factory function), patch that path instead.

- [ ] **Step 5: Run test to verify it passes**

```bash
cd /devwork/flashdreams
.venv/bin/python -m pytest \
    integrations/hy_worldplay/tests/test_runner_config.py::test_default_uses_native_pipeline \
    -v
```

Expected: PASS.

- [ ] **Step 6: Run the full HY-WorldPlay CPU test suite (regression bar)**

```bash
cd /devwork/flashdreams
.venv/bin/python -m pytest integrations/hy_worldplay/tests/ -v --tb=short 2>&1 | tail -20
```

Expected: All tests pass. Fix any test that asserted the old default (`False`) by either flipping the assertion to `True` or making it explicit (`use_native_pipeline=False` passed as kwarg in tests that exercise the vendor-wrapper path).

- [ ] **Step 7: Commit**

```bash
cd /devwork/flashdreams
git add integrations/hy_worldplay/hy_worldplay/ \
        integrations/hy_worldplay/tests/
git commit -m "$(cat <<'EOF'
feat(hy_worldplay): flip --use-native-pipeline to default (phase 2b.6 close step 1/3)

Now that the phase 2b.6 parity diff against vendor's use_kv_cache=True
baseline holds at mean |Δ| <= 5/255 (see prior commit's diff_summary.json),
the native runner becomes the default. Vendor-wrapper invocation is
still accessible via `--no-use-native-pipeline` (or `use_native_pipeline=False`
on the config dataclass).

CPU tests for the default value updated. Tests that exercise the
vendor wrapper now set `use_native_pipeline=False` explicitly.
EOF
)"
```

---

### Task 5: Drop the parity sub-venv heavy dependencies

**Files:**
- Modify: `integrations/hy_worldplay/tests/parity_check/pyproject.toml`
- Modify: `integrations/hy_worldplay/tests/parity_check/uv.lock` (regenerated by `uv sync`)

- [ ] **Step 1: Inventory current sub-venv deps**

```bash
cd /devwork/flashdreams/integrations/hy_worldplay/tests/parity_check
grep -E "sageattention|cloudpickle|accelerate|transformers" pyproject.toml
```

Expected: the four heavy deps listed by name.

- [ ] **Step 2: Decide what to keep vs drop**

The parity harness still needs:
- `huggingface_hub` (for the `huggingface-cli` download in `run.sh`)
- `torch` (vendor's `wan/generate.py` imports it)
- `diffusers` (vendor pipeline base class)
- `imageio[ffmpeg]` (if `hy_parity_diff.py` uses it)

Drop:
- `sageattention` (vendor's custom attention; native path uses standard attention)
- `cloudpickle` (vendor's distributed-init dependency)
- `accelerate` (vendor wrapper bootstrap)
- `transformers==4.57.6` (vendor pin)

Edit `pyproject.toml`: remove each of the four lines from the `[project.dependencies]` (or `dependencies = [...]`) array. Keep everything else untouched.

- [ ] **Step 3: Regenerate the lock file**

```bash
cd /devwork/flashdreams/integrations/hy_worldplay/tests/parity_check
uv lock
```

Expected: `uv.lock` shrinks by ~hundreds of lines (the transitive deps of the removed packages also fall away).

- [ ] **Step 4: Run CPU tests from the *main* flashdreams venv**

```bash
cd /devwork/flashdreams
.venv/bin/python -m pytest integrations/hy_worldplay/tests/ -v --tb=short 2>&1 | tail -20
```

Expected: All tests still pass. The main venv has nothing changed; the parity-sub-venv dep removal shouldn't affect it.

- [ ] **Step 5: GPU smoke from the main venv (no sub-venv)**

```bash
cd /devwork/flashdreams
uv run flashdreams-run hy-worldplay-wan-i2v-5b \
    --image-path integrations/hy_worldplay/tests/parity_check/HY-WorldPlay/assets/img/test.png \
    --ar-model-path integrations/hy_worldplay/tests/parity_check/HY-WorldPlay/hf_models/wan_transformer \
    --ckpt-path integrations/hy_worldplay/tests/parity_check/HY-WorldPlay/hf_models/wan_distilled_model/model.pt \
    --hy-worldplay-repo-root integrations/hy_worldplay/tests/parity_check/HY-WorldPlay \
    --num-chunk 1 \
    --pose "w-4" \
    --output-dir outputs/native_smoke_post_cleanup 2>&1 | tee /tmp/native_smoke.log
```

Expected: 1-chunk native rollout completes from the **main** `.venv` (no `--project integrations/hy_worldplay/tests/parity_check` flag), MP4 written. Confirms the heavy deps were genuinely unused on the native path.

- [ ] **Step 6: Commit**

```bash
cd /devwork/flashdreams
git add integrations/hy_worldplay/tests/parity_check/pyproject.toml \
        integrations/hy_worldplay/tests/parity_check/uv.lock
git commit -m "$(cat <<'EOF'
chore(hy_worldplay): drop parity sub-venv heavy deps (phase 2b.6 close step 2/3)

Removes sageattention, cloudpickle, accelerate, and transformers==4.57.6
from the parity sub-venv. With --use-native-pipeline now the default
(prior commit), the native HY runner runs entirely from the main
flashdreams venv and these deps are no longer reachable on the
production path.

The sub-venv stays in place so we can still re-baseline against
vendor's wan/generate.py if needed (e.g. for the deferred 2b.6.1
Option A refactor); deps are now scoped to torch + diffusers +
huggingface_hub + imageio[ffmpeg] -- the minimum to clone, download
checkpoints, and run vendor's generate.py.

Verified with the full CPU test suite (regression bar) and a 1-chunk
GPU smoke from the main venv that the native path doesn't depend on
any of the dropped packages.
EOF
)"
```

---

### Task 6: Update README + design spec to mark 2b.6 closed

**Files:**
- Modify: `integrations/hy_worldplay/README.md`
- Modify: `docs/superpowers/specs/2026-05-20-hy-worldplay-phase-2b-design.md`

- [ ] **Step 1: README updates**

In `integrations/hy_worldplay/README.md`:

- Update the "Run" section to make the native invocation primary (move it before the vendor wrapper example). Document `--no-use-native-pipeline` as the historical fallback.
- In the "Native pipeline (preview)" section, change the heading from "(preview)" to "(default)".
- Replace the "Native path's parity status" prose with a short statement:

```markdown
The native path's parity status as of this release: `mean |Δ| =
<FILL IN> / 255` against vendor's `use_kv_cache=True` baseline at
704x1280 / `num_chunk=2` / `seed=0`, meeting the phase 2b.6
acceptance bar (`≤5 / 255`). The native invocation is now the
default; vendor wrapper stays available via
`--no-use-native-pipeline` for callers that need bit-exact match
against upstream's published `use_kv_cache=False` default (deferred
to 2b.6.1; not currently planned).
```

- In the "Native pipeline (preview)" bullet list, mark **2b.6** as `(landed)` instead of `(partially landed)`. The 2b.6.1 bullet stays as `(future; not currently planned)`.

- In the "Staging plan" list, similarly mark 2b.6 as `(landed)`.

- [ ] **Step 2: Design spec updates**

In `docs/superpowers/specs/2026-05-20-hy-worldplay-phase-2b-design.md`:

- Sub-PR table: change the 2b.6 row's status from `(in progress; close path = Option C)` to `(landed)`. Update its description to record the final parity number.
- Success criteria table: append the final parity number to the 2b.6 row.
- "Sub-PR 2b.6 design (this session)" section: add a brief "Outcome" subsection at the end recording the final parity numbers + which Phase 2 commits closed the cleanup.

- [ ] **Step 3: Commit**

```bash
cd /devwork/flashdreams
git add integrations/hy_worldplay/README.md \
        docs/superpowers/specs/2026-05-20-hy-worldplay-phase-2b-design.md
git commit -m "$(cat <<'EOF'
docs(hy_worldplay): mark phase 2b.6 closed (phase 2b.6 close step 3/3)

Final parity diff against vendor's use_kv_cache=True baseline at
704x1280 / num_chunk=2 / seed=0: mean |Δ| = <FILL IN>/255 (parity
bar: 5/255).

README updates: native invocation promoted to documented default;
vendor wrapper documented as --no-use-native-pipeline fallback;
"Native pipeline (preview)" renamed to "Native pipeline (default)";
phase 2b.6 marked landed across both bullet lists.

Design spec updates: 2b.6 row status flipped to landed with the
final parity number recorded; "Sub-PR 2b.6 design" section gains
an Outcome subsection summarising which commits closed each
sub-step (helper, run.sh wiring, default flip, sub-venv drop,
docs).

Phase 2b is now feature-complete. 2b.6.1 (Option A refactor for
bit-exact match against vendor's use_kv_cache=False default)
remains in the "future; not currently planned" status.
EOF
)"
```

---

### Task 7: Optional — delete the vendor-wrapper runner if no consumer remains

**Files:**
- Potentially delete: `integrations/hy_worldplay/hy_worldplay/_runner.py` (the vendor-wrapper)
- Modify: `integrations/hy_worldplay/hy_worldplay/__init__.py` (drop the export)
- Modify: tests that exercise the wrapper

- [ ] **Step 1: Decide whether to delete or keep**

```bash
cd /devwork/flashdreams
grep -rn "HyWorldPlayWanI2VRunner\b" integrations/hy_worldplay/ \
                                      flashdreams/ \
    | grep -v __pycache__ | grep -v "/_native_runner"
```

If the only references are inside the vendor-wrapper runner itself + the runner-config entry point + the `--no-use-native-pipeline` fallback, the wrapper is still a documented fallback and should stay. **Default: KEEP.** Skip Task 7 entirely.

If references show only test-internal usage and no live integration, the wrapper can be removed.

- [ ] **Step 2 (only if deleting): Remove the wrapper + smoke tests**

```bash
cd /devwork/flashdreams
git rm integrations/hy_worldplay/hy_worldplay/_runner.py
# Edit __init__.py to drop the HyWorldPlayWanI2VRunner export.
# Edit affected tests to drop wrapper-only assertions.
```

Run CPU tests:

```bash
.venv/bin/python -m pytest integrations/hy_worldplay/tests/ -v --tb=short 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 3 (only if deleting): Commit**

```bash
git commit -m "$(cat <<'EOF'
chore(hy_worldplay): drop vendor-wrapper runner (phase 2b.6 close optional cleanup)

With --use-native-pipeline now the default and validated against
vendor's use_kv_cache=True baseline at mean |Δ| ≤ 5/255, the
phase-1 vendor-wrapper runner has no remaining consumer. Removing
HyWorldPlayWanI2VRunner + its dispatch from __init__.py + the
wrapper-only smoke tests.

If a downstream caller needs bit-exact match against vendor's
use_kv_cache=False default, the path forward is 2b.6.1 (Option A
refactor), not reviving the vendor wrapper -- the wrapper depends
on the heavy sub-venv (sageattention / cloudpickle / accelerate)
which was dropped in the prior cleanup commit.
EOF
)"
```

---

## Self-review

1. **Spec coverage:**
   - "Re-baseline vendor with `use_kv_cache=True`" → Tasks 1-3.
   - "Regenerate vendor MP4" → Task 3 Step 2.
   - "Diff native against the new baseline" → Task 3 Step 4.
   - "Two outcomes (parity holds / doesn't)" → Task 3 Step 5 decision gate.
   - Cleanup: "drop the parity sub-venv" → Task 5; "flip `--use-native-pipeline` to default" → Task 4; "update README + design spec" → Task 6.
   - "Failure-mode contingencies" (vendor `use_kv_cache=True` itself broken, chunk-0 only, seed-dependent) → Task 3 Step 5 decision-gate table.
   - "All 99 HY-WorldPlay CPU tests still pass" regression bar → Task 1 Step 5, Task 4 Step 6, Task 5 Step 4.
   - "Optional new CPU test: `test_parity_check_use_kv_cache_true_baseline_exists`" → Task 1 Steps 1-4 (three tests for the subclass factory; renamed to be CPU-only).

2. **Placeholder scan:** Tasks reference concrete files, exact commands, and complete code. Only `<FILL IN>` placeholders are intentional (the actual parity numbers from Task 3 Step 4 + the commit-time records in Tasks 3 Step 6, 4 Step 7, 5 Step 6, 6 Step 3 — these are filled in at execution time once Task 3 produces the diff).

3. **Type consistency:**
   - `make_use_kv_cache_true_subclass` is the same name in the helper (Task 1 Step 3), the test (Task 1 Step 1), and downstream references.
   - `USE_KV_CACHE_TRUE` env-var spelling is consistent across run.sh (Task 2 Step 1), parity_check README (Task 2 Step 3), and Task 3 invocation.
   - `use_native_pipeline` attribute spelling matches across Task 4's tests + the runner-config flip.

4. **Implicit decisions worth flagging:**
   - The plan assumes `--seed` is a valid `flashdreams-run` flag. If it isn't, Task 3 Step 3 should pass the seed via env-var or set `torch.manual_seed` from a wrapper. Cross-check via `flashdreams-run hy-worldplay-wan-i2v-5b --help` before Task 3.
   - The plan assumes `tmp/hy_parity_diff.py` exists and works. If it was a transient prototype, Task 3 Step 4 may need to either resurrect it from git history or replace it with a small inline imageio-based diff. Cross-check before running Task 3.
   - The plan assumes the existing GPU has ≥40 GB RAM. The prior 2b.5b-part2-followup run was on an RTX 6000 Pro at 256x448; production parity is 704x1280, requiring more memory. If OOM happens, fall back to lower resolution and re-baseline both sides at that resolution.
