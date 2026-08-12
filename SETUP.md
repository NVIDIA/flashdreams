# FlashDreams Interactive-Drive on Windows 11: Complete Setup & Fixes Guide

This is the comprehensive guide for running FlashDreams interactive-drive on Windows 11 with RTX 5090 (or similar NVIDIA GPU).

---

## Part 1: Requirements & Setup

### System Requirements

- **OS:** Windows 11 with CUDA 13.0
- **Python:** 3.11.15 (in `.venv`)
- **Compiler:** Visual Studio 2022 Community
- **GPU:** NVIDIA RTX 5090 or compatible (sm_120 architecture)
- **PyTorch:** 2.8.x (cu130 wheels) — **NOT 2.12.1+**
- **Disk:** 20+ GB free in HuggingFace cache directory (`C:\Users\<user>\.cache\huggingface\hub`)

### PyTorch Version Warning

**Use PyTorch 2.8.x, not 2.12.1+**

PyTorch 2.12.1+ has a broken functorch integration on Windows:
```
ImportError: cannot import name 'min_cut_rematerialization_partition' from 'functorch.compile'
```

This error occurs during `torch._dynamo` initialization (before environment variables like `TORCH_COMPILE_DISABLE` can take effect) and is not recoverable.

The setup script uses **narrow sync** to preserve your pinned torch version:
```powershell
uv sync --package flashdreams-omnidreams --extra dev --extra interactive-drive
```

This respects the project's dependency pins instead of upgrading to the latest (2.12.1+).

If you need to install a specific torch version:
```powershell
uv pip install "torch==2.8.1+cu130" --index https://download.pytorch.org/whl/cu130
```

---

## Part 2: Installation

### Step 1: Run Complete Setup

```powershell
cd C:\workspace\world\flashdream_public
.\setup_interactive_drive.bat
```

This script:
- Syncs dependencies via **narrow `uv sync --package flashdreams-omnidreams`** (preserves your torch version)
- Installs SageAttention (optional, pre-built wheel)
- Downloads models (Cosmos-Reason1, LightWave VAE/TAE, OmniDreams)
- Builds C++ extensions (Ludus renderer, PhysX)
- Optional: Precompiles torch.compile cache (skipped on Windows by default)

**Expected output:**
```
[SETUP] 1. Syncing dependencies...
[SETUP] 1b. Installing SageAttention...
[SETUP] 2. Syncing third-party sources...
[SETUP] 3. Preparing for perf (downloads models, builds extensions)...
✓ SETUP COMPLETE
```

### Step 2: Run Interactive-Drive

```powershell
.\run_interactive_drive_perf.bat --game-mode
```

**Expected output:**
```
===================================================================
LAUNCHING INTERACTIVE-DRIVE PERF WITH PHYSICS
===================================================================
Resolution: 1168x640 (perf tuned)
Denoising steps: [1000, 100]
Native acceleration: auto-fallback to PyTorch
===================================================================

[INIT] Starting event loop...
...
[config] Disabling torch.compile on Windows (CUDA graph deadlock)
[config] Disabling native DIT on Windows (nvcc compilation hang)
...
[chunk-pipeline] warmup done elapsed_ms=0.1
```

Then the HUD window opens and waits for scene selection.

---

## Part 3: Controls

### Driving
- **WASD** — Drive forward/back/left/right
- **Mouse** — Look around
- **C** — Spawn obstacle
- **R** — Restart session (clears KV cache)
- **Esc** — Quit

### Prompt Editing (in Scene Prompt text field)
- `/spawn car 30 5` — Spawn vehicle at position
- `/clear-actors` — Clear all actors

---

## Part 4: Performance & Timing

### Expected Performance

| Stage | Time | Notes |
|-------|------|-------|
| **App startup** | ~10 seconds | Includes CUDA init, model loading |
| **Scene selection** | <1 second | HUD ready |
| **First chunk generation** | ~30-45 seconds | Includes one-shot encoder precompute |
| **Subsequent chunks** | ~2-3 seconds @ 30fps | Real-time streaming |

### Configuration

**Resolution:** 1168x640 (perf tuned)  
**Denoising steps:** [1000, 100] (2-stage: coarse + refine)  
**Inference mode:** Eager mode (torch.compile disabled on Windows)  
**Attention backend:** cuDNN (fallback; SageAttention not used)

---

## Part 5: Windows-Specific Fixes & Architecture

### Issue 1: torch.compile Functorch Hang (FIXED)

**Problem:**
- PyTorch 2.12.1+ has broken functorch integration on Windows
- Error occurs in `torch._dynamo` during compiler infrastructure initialization
- Environment variable `TORCH_COMPILE_DISABLE` has no effect (error happens before the check)

**Solution:** Skip torch.compile entirely on Windows, use eager mode.

**File:** `flashdreams/flashdreams/infra/compile.py` (lines 148-149)
```python
def compile_module(module: M, *, mode: CompileMode = "max-autotune-no-cudagraphs") -> M:
    if sys.platform == "win32":
        return module  # Skip compilation on Windows
    _configure_inductor_cache()
    _patch_triton_bundle_collection()
    return cast(M, torch.compile(module, mode=mode))
```

**Trade-off:** ~2x slower inference (but still real-time)

---

### Issue 2: Native DIT Extension Compilation Hang (FIXED)

**Problem:**
- Native DIT (`omnidreams_singleview.select_backend()` with `mode=required`) tries to compile SageAttention + CUTLASS extensions via nvcc + Ninja
- On Windows: nvcc hangs finding CUDA toolkit, Ninja subprocess deadlocks, or compilation takes 45-90 minutes
- No timeout or fallback mechanism → silent hang

**Root cause:**
1. `torch.utils.cpp_extension.load()` invokes external tools (nvcc, Ninja, cl.exe)
2. Windows subprocess handling can deadlock when launching compilers from thread pools
3. CUDA toolkit detection on Windows PATH is fragile
4. No error handling, just hangs indefinitely

**Solution:** Disable native_dit_acceleration on Windows at config level.

**File:** `integrations/omnidreams/omnidreams/interactive_drive/world_model/flashdreams_adapter.py` (lines 151-161)
```python
if sys.platform == "win32":
    logger.info("[config] Disabling torch.compile on Windows (CUDA graph deadlock)")
    logger.info("[config] Disabling native DIT on Windows (nvcc compilation hang)")
    transformer_overrides = {
        **transformer_overrides,
        "compile_network": False,
        "native_dit_acceleration": "disabled",
    }
```

**Trade-off:** ~2-3x slower inference vs optimized native DIT (but still real-time at 2-3s/chunk)

---

### Issue 3: Disk Space Error During Scene Load (FIXED)

**Problem:**
- App loads for 30+ seconds, then crashes with `DiskSpaceError` during scene load
- Error happens in worker thread, crashes app with no recovery option
- User wastes time loading models before knowing disk is full

**Solutions implemented:**

A) **Preflight check at startup** (demo.py lines 706-713)
```python
try:
    ensure_free_disk(
        default_huggingface_cache_dir(),
        required_bytes=cache_min_free_bytes(),
        label="interactive-drive startup",
    )
except Exception as e:
    raise SystemExit(f"Disk space preflight failed: {e}") from e
```

B) **Graceful error handling in worker** (chunk_pipeline.py lines 339-345)
```python
except DiskSpaceError as exc:
    logger.error(
        f"[chunk-pipeline] DISK SPACE ERROR: {exc}\n"
        "Free up space or set HF_HOME to another drive and retry."
    )
    continue  # Don't crash, just wait for space
```

---

### Issue 4: No Timing Visibility on Model Loading (FIXED)

**Problem:**
- When app hangs, no logs to identify where (checkpoint load? state dict load? native DIT config?)
- Users have no way to diagnose if hang is in torch.load, load_state_dict, or extension compilation

**Solution:** Add timing logs around critical operations.

**File:** `flashdreams/flashdreams/core/checkpoint/load.py` (lines 744-748)
```python
logger.info(f"[CHECKPOINT-LOAD-START] torch.load({path})")
start = time.perf_counter()
result = torch.load(path, map_location=map_location, weights_only=False)
elapsed = time.perf_counter() - start
logger.info(f"[CHECKPOINT-LOAD-DONE] torch.load completed in {elapsed:.1f}s, {len(result)} tensors")
```

**File:** `integrations/omnidreams/omnidreams/transformer/__init__.py` (lines 364-377)
```python
logger.info(f"[STATE-DICT-TRANSFORM-START] Transforming {len(state_dict)} keys")
start = time.perf_counter()
state_dict = transform(state_dict)
elapsed = time.perf_counter() - start
logger.info(f"[STATE-DICT-TRANSFORM-DONE] Transform completed in {elapsed:.1f}s")

logger.info(f"[LOAD-STATE-DICT-START] Loading {len(state_dict)} tensors")
start = time.perf_counter()
self.network.load_state_dict(state_dict)
elapsed = time.perf_counter() - start
logger.info(f"[LOAD-STATE-DICT-DONE] load_state_dict completed in {elapsed:.1f}s")
```

**File:** `integrations/omnidreams/omnidreams/transformer/__init__.py` (lines 373-379)
```python
logger.info(f"[NATIVE-DIT-CONFIG-START] Loading native DIT (mode={config.native_dit_acceleration})")
start = time.perf_counter()
self._configure_optimized_dit_from_config()
elapsed = time.perf_counter() - start
logger.info(f"[NATIVE-DIT-CONFIG-DONE] Native DIT setup completed in {elapsed:.1f}s")
```

**Usage:** If no `[...-DONE]` log appears, the process is hanging at that stage.

---

### Issue 5: Excessive Debug Logging (FIXED)

**Problem:**
- Checkpoint loading had excessive `[DEBUG-*]` logs cluttering the output:
  ```
  [DEBUG-CACHE-CHECK] Checking if cached...
  [DEBUG-PREFLIGHT] Running preflight check...
  [DEBUG-HF-CACHE] Checking HF cache...
  [DEBUG-HF-DOWNLOAD-START] Starting HF hub download...
  [DEBUG-HF-DOWNLOAD-DONE] Download complete
  ```

**Solution:** Remove all `[DEBUG-*]` logs, keep only final success message.

**File:** `flashdreams/flashdreams/core/checkpoint/load.py` (lines 496-532)

**Result:** Cleaner logs, easier to read.

---

## Part 6: Dependencies & Wheels

### PyTorch Installation

The setup uses **narrow sync** to avoid upgrading torch:
```powershell
uv sync --package flashdreams-omnidreams --extra dev --extra interactive-drive
```

This installs torch 2.8.x from the project's pinned versions, not the latest.

### SageAttention (Optional)

Installed as a pre-built wheel (no compilation):
```powershell
uv pip install sageattention --no-deps
```

**Note:** SageAttention is not actively used on Windows (native DIT is disabled). It's installed for future use when native DIT can be enabled safely.

### Other Key Wheels

- **torch** — 2.8.x (cu130)
- **triton-windows** — Required for torch.compile on Windows (not used in eager mode)
- **flash-attn** — Pre-built wheels via mjun0812 (sm_120 verified)
- **transformers** — HuggingFace transformers library

---

## Part 7: Troubleshooting

### "ImportError: min_cut_rematerialization_partition"

**Cause:** PyTorch 2.12.1+ functorch broken on Windows

**Solution:**
```powershell
uv pip install "torch==2.8.1+cu130" --index https://download.pytorch.org/whl/cu130
Remove-Item -Recurse -Force flashdreams\flashdreams\infra\__pycache__
```

### "Not enough free disk for Hugging Face cache (18.5 GiB free, 20.0 GiB required)"

**Cause:** HuggingFace cache directory doesn't have 20 GB free

**Solutions:**
1. **Free up disk space** (~2 GB minimum)
2. **Move HF cache** to another drive:
   ```powershell
   $env:HF_HOME = "D:\huggingface"
   .\run_interactive_drive_perf.bat --game-mode
   ```
3. **Skip the check** (risky, but works if you monitor):
   ```powershell
   $env:FLASHDREAMS_MIN_CACHE_FREE_GB = "0"
   .\run_interactive_drive_perf.bat --game-mode
   ```

### "No module named pip"

**Cause:** uv-created venv doesn't include pip

**Solution:** Use `uv pip` instead of `python -m pip`
```powershell
uv pip install package-name
```

### Ludus build fails with "stdlib.h not found"

**Cause:** MSVC compiler not set up (missing vcvarsall.bat call)

**Solution:** Run manually:
```powershell
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
```

---

## Part 8: File Summary

### Modified Files

| File | Changes | Purpose |
|------|---------|---------|
| `flashdreams/infra/compile.py` | Skip torch.compile on Windows | Fix functorch hang |
| `flashdreams/core/checkpoint/load.py` | Add timing logs, remove debug logs | Visibility + cleaner output |
| `omnidreams/transformer/__init__.py` | Add logger import, timing logs | Visibility into model load |
| `omnidreams/interactive_drive/world_model/flashdreams_adapter.py` | Disable native DIT on Windows | Fix nvcc hang |
| `omnidreams/interactive_drive/video_model/chunk_pipeline.py` | Catch DiskSpaceError gracefully | Handle disk full gracefully |
| `omnidreams/interactive_drive/demo.py` | Add preflight disk check | Fail fast if disk full |
| `setup_interactive_drive.bat` | Narrow sync + SageAttention install | Preserve torch version, optional optimization |
| `example_world_model_perf.yaml` | Use sage3 attention backend | Prepare for future optimization |

---

## Part 9: Performance Summary

| Metric | With Fixes | Notes |
|--------|-----------|-------|
| **Startup** | ~10 seconds | CUDA init + model load |
| **First chunk** | ~30-45 seconds | One-shot encoder precompute |
| **Subsequent chunks** | ~2-3 seconds @ 30fps | Real-time streaming |
| **Inference mode** | Eager (PyTorch) | No torch.compile, no native DIT |
| **Stability** | Stable | No hangs, graceful error handling |

---

## Part 10: Architecture Diagram

```
App Startup
    ↓
Preflight disk space check (demo.py)
    ↓ (fails if <20 GB free)
Scene picker HUD
    ↓
User selects scene
    ↓
Load scene (flashdreams_adapter.py)
    ├─ Set config overrides (Windows)
    │  ├─ compile_network = False
    │  ├─ use_cuda_graph = False
    │  └─ native_dit_acceleration = "disabled"
    ├─ Download checkpoints (if not cached)
    │  └─ torch.load (1-2 seconds)
    ├─ Load state_dict (0.4 seconds)
    ├─ Skip native DIT config (Windows)
    └─ Initialize CUDA (30-60 seconds first time)
    ↓
Encoding (text + image)
    ├─ Text encoder (offloaded to CPU)
    └─ Image encoder (offloaded to CPU)
    ↓
Denoising loop (real-time)
    ├─ Stage 1: 1000 steps (coarse)
    └─ Stage 2: 100 steps (refine)
    ↓
Render & display @ 30fps
```

---

## Part 11: FAQ

**Q: Why is inference so slow on Windows?**  
A: Eager mode (no torch.compile, no native DIT) is ~2-3x slower than optimized, but still real-time (~2-3s/chunk). Trade-off favors stability over speed.

**Q: Can I enable native DIT on Windows?**  
A: Not recommended. It will hang during nvcc compilation. If you need the speedup, use WSL2 or a Linux machine.

**Q: Can I use PyTorch 2.12.1?**  
A: No. Use 2.8.x only. 2.12.1+ has broken functorch on Windows (not recoverable).

**Q: Where is the HuggingFace cache?**  
A: Default: `C:\Users\<username>\.cache\huggingface\hub`  
Override: `$env:HF_HOME = "D:\path"`

**Q: How much disk space do I need?**  
A: 20+ GB free in HuggingFace cache directory (for Cosmos-Reason1, LightWave, OmniDreams models).

**Q: What GPU do I need?**  
A: NVIDIA RTX 5090 (sm_120 architecture) with CUDA 13.0. Other recent NVIDIA GPUs may work with arch adjustments.

---

## Part 12: References

- **PyTorch functorch issue:** Windows torch._dynamo initialization fails with broken functorch import in 2.12.1+
- **CUDA graphs issue:** Windows WDDM2 driver interaction causes deadlocks with CUDA graph capture
- **Native DIT hang:** omnidreams_singleview.select_backend subprocess deadlock on Windows nvcc/Ninja launch
- **Disk space check:** Preflight HuggingFace cache validation before expensive model loading

---

## Questions?

See `WINDOWS_FIXES.md` for detailed technical breakdown of each fix, or check logs during app run for timing information.
