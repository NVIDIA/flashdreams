# OmniDreams cuDNN MHA + SlangPy/Vulkan Repro Notes

Date: 2026-05-27

Workspace: `/home/gtong/github/flashdreams`

## Summary

The cuDNN attention crash is reproducible without loading OmniDreams weights.
The trigger is creating a SlangPy Vulkan device before the first large cuDNN
MHA execution in the same Python thread.

Implemented workaround status:

- `InteractiveDriveApp` now starts the chunk-pipeline worker before constructing
  the SlangPy presenter.
- That worker runs a cuDNN MHA primer before SlangPy creates its Vulkan device,
  then waits while the presenter is created, then continues into normal
  world-model warmup and generation on the same thread.
- The primer must cover the filling KV-cache variants, not only the initial
  self-attention shape.

The small reproducer is:

```bash
integrations/omnidreams/scripts/repro_cudnn_slangpy.py
```

Failing command:

```bash
TORCH_COMPILE_DISABLE=1 PYTHONUNBUFFERED=1 \
  xvfb-run -a .venv/bin/python \
  integrations/omnidreams/scripts/repro_cudnn_slangpy.py --slang device
```

Failure:

```text
RuntimeError: Expected mha_graph.execute(handle, variant_pack, workspace_ptr.get()).is_good() to be true, but got false.
```

Passing baseline:

```bash
TORCH_COMPILE_DISABLE=1 PYTHONUNBUFFERED=1 \
  .venv/bin/python \
  integrations/omnidreams/scripts/repro_cudnn_slangpy.py --slang none
```

Passing same-thread primer:

```bash
TORCH_COMPILE_DISABLE=1 PYTHONUNBUFFERED=1 \
  xvfb-run -a .venv/bin/python \
  integrations/omnidreams/scripts/repro_cudnn_slangpy.py \
  --slang device --thread-mode worker-prime-before-slang
```

## Current Environment

Observed during the repro run:

```text
GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
Driver: 610.43.02
torch: 2.12.0+cu130
torch.version.cuda: 13.0
torch.backends.cudnn.version: 92000
nvidia-cudnn-cu13: 9.20.0.48
slangpy: 0.41.0
ludus-renderer: 0.9.0
CUDA_HOME: /home/gtong/cuda-13.2.1
LD_LIBRARY_PATH: /home/gtong/cuda-13.2.1/targets/x86_64-linux/lib:...
```

`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json` was used for the
main repro commands to remove non-NVIDIA ICD noise.

## Captured Failing Attention Call

The first failing OmniDreams attention call was captured by monkey-patching
`flashdreams.core.attention.cp.torch_sdpa_cudnn`.

```text
q shape=(1, 16, 7040, 128) dtype=torch.bfloat16 stride=(14417920, 128, 2048, 1)
k shape=(1, 16, 7040, 128) dtype=torch.bfloat16 stride=(43253760, 128, 2048, 1)
v shape=(1, 16, 7040, 128) dtype=torch.bfloat16 stride=(43253760, 128, 2048, 1)
return_lse=False
```

The repro preserves those shapes and the captured non-compact q/k/v strides by
default.

## Experiment Matrix

| Experiment | Result | Notes |
|---|---:|---|
| cuDNN MHA only, no SlangPy | Pass | Same shape/strides as failing call. |
| `import slangpy` only, then cuDNN | Pass | Importing SlangPy is not enough to trigger the bug. |
| `spy.Window(...)` only, then cuDNN | Pass | Creating the SDL/window object is not enough. |
| `spy.Device(type=vulkan)`, then cuDNN | Fail | Minimal trigger. Window is not required. |
| `torch import`, then `spy.Device`, then cuDNN | Fail | Importing torch first is not enough. |
| Clean `CUDA_HOME`, `CUDA_PATH`, `LD_LIBRARY_PATH` | Fail | Current shell CUDA env is not the primary trigger. |
| `CUDA_LAUNCH_BLOCKING=1` | Fail | Synchronization/debug mode does not avoid it. |
| Delete SlangPy device before cuDNN | Fail | Once the Vulkan device has been created, deleting the object does not reset the bad state. |
| `S=512` or `S=1024` after SlangPy device | Pass | Smaller attention shapes do not trigger it. |
| `S=2048`, `4096`, `7040` after SlangPy device | Fail | Threshold on this stack is between 1024 and 2048. |
| Compact contiguous `S=7040` after SlangPy device | Fail | The cache stride pattern is not required. |
| Main-thread `S=7040` primer before SlangPy, then main-thread cuDNN | Pass | cuDNN plan/handle state initialized before Vulkan appears to protect that same thread. |
| Main-thread `S=7040` primer before full demo | Fail | Full demo's first model chunk runs in a different worker thread. |
| Worker-thread `S=7040` primer before SlangPy, same worker after SlangPy | Pass | Strong evidence that the useful cuDNN initialization is thread-local. |
| Disable transformer/encoder/decoder compile and CUDA graph in full demo | Fail | The raw cuDNN op fails too; CUDA graph is not the root trigger. |
| Full demo with worker primer at `q=7040, k/v=7040` only | Fail later | Initial chunk passed, but first steady-state chunk failed at `q=7040, k/v=14080`. |
| Full demo with worker primer at `q=7040, k/v=[7040, 14080, 21120]` | Pass through chunk 56 | Manually terminated after the historical crash point; no cuDNN MHA failure observed. |

## What The Repro Suggests

The bug looks like a same-process interaction between:

- SlangPy creating an NVIDIA Vulkan device, and
- the first large cuDNN MHA execution on a thread that has not already
  initialized a matching cuDNN MHA plan/handle.

The useful state appears to be thread-local. A primer on the main thread does
not protect the interactive-drive chunk worker thread. A primer on the same
worker thread before Vulkan device creation does protect that worker in the
small repro.

## Possible Fixes And Consequences

### 1. Prime cuDNN MHA on the future model worker before SlangPy creates Vulkan

Implemented shape of the fix:

1. Create the model/chunk worker thread before constructing `SlangPyPresenter`.
2. On that exact worker thread, run representative cuDNN MHA primers for the
   expected self-attention KV-cache fill shapes:
   `B=1, H=16, Q=7040, K/V in {7040, 14080, 21120}, D=128`, bf16.
3. Only after that primer completes, create `spy.Device(type=vulkan)`.
4. Keep using that same worker thread for model inference.

Pros:

- Keeps the cuDNN backend.
- The tiny repro shows this can work.
- Does not require changing attention math or falling back to flash attention.

Consequences:

- Startup gets slower by three large cuDNN MHA calls. On the tested machine,
  the extended primer took about 64 ms.
- Temporary VRAM use increases. The default primer allocates a `Q=7040` tensor
  and `K/V=21120` cache tensors, plus cuDNN output/lse/workspace allocations,
  before releasing them and calling `torch.cuda.empty_cache()`.
- It is shape-sensitive. A `Q=7040, K/V=7040` primer protected the initial
  chunk but did not protect the first steady-state chunk; `K/V=14080` and
  `21120` had to be primed too.
- App startup architecture changes: the current app creates the SlangPy
  presenter before `ChunkPipeline` starts its worker. The workaround adds an
  explicit model-worker prestart phase and delays SlangPy/Vulkan presenter
  creation for `--no-hud` and autoload HUD runs.
- If future checkpoints use larger heads/sequence lengths, the primer shape
  must be updated or generalized.

Escape hatches:

- Set `OMNIDREAMS_CUDNN_SLANGPY_PRIMER=0` to disable the primer.
- Set `OMNIDREAMS_CUDNN_SLANGPY_PRIMER_SEQ_LEN=<S>` to force a single
  square-primer shape for quick experiments.

Verification command used after implementation:

```bash
timeout 480s env TORCH_COMPILE_DISABLE=1 PYTHONUNBUFFERED=1 \
  VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  xvfb-run -a .venv/bin/python -m omnidreams.interactive_drive.demo \
  --no-hud --no-wheel --autoload-scene
```

The run was manually terminated after `next_chunk index=56`; the old crash at
the first steady-state chunk did not recur.

### 2. Delay SlangPy Vulkan device creation until after model worker warmup/primer

Potential shape of the fix:

- Create the window or loading UI later, after the model worker has initialized
  cuDNN MHA.
- Or split `SlangPyPresenter` construction so `Window` can exist early, but
  `spy.Device(type=vulkan)` is delayed.

Pros:

- `spy.Window(...)` alone did not trigger the tiny repro; the Vulkan device did.
- Avoids process separation.

Consequences:

- The user may not see the current loading window while model warmup happens.
- Presenter lifecycle becomes more complex because the surface, display texture,
  and command buffers depend on the delayed Vulkan device.
- Still likely needs a representative same-thread cuDNN primer, not just normal
  torch import.

### 3. Process-separate rendering and model inference

Potential shape of the fix:

- Keep SlangPy/Vulkan in the presenter process.
- Run the world model in a separate Python process.
- Exchange frames/controls over shared memory, CUDA IPC, sockets, or an existing
  serving path.

Pros:

- Strong isolation from same-process Vulkan/cuDNN driver interactions.
- Likely the most robust workaround if the root cause is driver-global process
  state.
- Keeps cuDNN attention in the model process.

Consequences:

- More engineering work.
- More latency unless transport is carefully designed.
- More memory overhead if model/render resources cannot be shared.
- More failure modes: process lifecycle, IPC backpressure, synchronization, and
  teardown.

### 4. Reduce the cuDNN attention sequence length

Observation:

- `S=512` and `S=1024` pass after SlangPy device creation.
- `S=2048` and above fail in the tiny repro on this stack.

Pros:

- May avoid the driver/cuDNN failure while still using cuDNN.

Consequences:

- Likely changes model architecture or runtime chunking.
- May reduce temporal/spatial context, quality, or consistency.
- May increase overhead if the model needs more smaller attention calls.
- Not a clean fix unless the model already has a valid low-sequence operating
  point.

### 5. Disable CUDA graphs and torch compile

Tested result:

- Not a fix. The full demo still failed after forcing transformer,
  encoder, decoder, and image encoder `use_cuda_graph=False` and
  `use_compile=False`.
- The tiny repro fails with a raw eager cuDNN op after `spy.Device`.

Consequences if kept anyway:

- Lower steady-state performance.
- Longer per-chunk model time.
- Less complexity during debugging, but it does not address the root trigger.

### 6. Environment cleanup and synchronization knobs

Tested result:

- Unsetting `CUDA_HOME`, `CUDA_PATH`, and `LD_LIBRARY_PATH` did not fix the tiny
  repro.
- `CUDA_LAUNCH_BLOCKING=1` did not fix it.
- Forcing `VK_ICD_FILENAMES` keeps the process cleaner but did not fix the
  cuDNN failure.

Consequences:

- These are still good hygiene for reproducibility.
- They should not be treated as product fixes for this crash.

### 7. Driver / PyTorch / cuDNN matrix

Still worth testing:

- torch `2.11.0+cu130` with cuDNN `9.19`, matching the earlier known-good stack.
- Newer PyTorch/cuDNN nightly or NVIDIA container stack.
- Another NVIDIA driver branch.

Pros:

- The tiny repro is fast enough to run a version matrix cheaply.

Consequences:

- Version pinning may constrain other dependencies.
- A driver-only fix may be outside application control.

## Recommended Next Step

Prototype the least invasive product fix around option 1:

- Add a model-worker prestart path that runs before `SlangPyPresenter` creates
  `spy.Device`.
- In that worker thread, run a representative cuDNN MHA primer.
- Keep the worker alive and reuse it for actual generation.
- Verify with `ContextParallelAttention(... backend="cudnn")`,
  `TORCH_COMPILE_DISABLE=1`, and `VK_ICD_FILENAMES` forced.

If that is too awkward for the current app lifecycle, process separation is the
next most robust workaround.
