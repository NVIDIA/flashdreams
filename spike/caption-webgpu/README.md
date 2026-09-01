# Caption‑model WebGPU smoke test — task #1 de‑risk (throwaway)

Not product code. Proves the small **latent → caption‑bank classifier**
architecture exports to ONNX and **runs on onnxruntime‑web / WebGPU in the
browser — before any data collection or training**. Random weights, so the
predicted caption is meaningless; only the **op coverage, shapes, and latency**
matter. Delete before any product PR.

The architecture (`model.py`) is deliberately WebGPU‑op‑friendly: per‑frame 2D
convs over the latent grid → global average pool → temporal mean → small MLP →
caption‑bank logits. No 3D conv. Input window: 4 chunks × 2 frames = 8 latent
frames of `[16, 88, 160]` → input `[1, 8, 16, 88, 160]` → logits `[1, 32]`.

## Steps (RTX box)

1. **Export the random‑weights model:**
   ```bash
   uv run python spike/caption-webgpu/export_smoke.py
   ```
   → writes `caption_model.fp32.onnx` + `.spec.json` in this dir. Expect a line
   like `in (1, 8, 16, 88, 160) -> out (1, 32)` and `[ok] wrote …`.

2. **Serve this dir over a secure context** (WebGPU needs localhost/https):
   ```bash
   python -m http.server 8000 --directory spike/caption-webgpu
   ```
   Open `http://localhost:8000/smoke.html` — on the box directly, or via an SSH
   tunnel from your laptop: `ssh -L 8000:localhost:8000 <rtx-box>`.

3. **Click “Run smoke test.”** Success looks like:
   - `WebGPU adapter: available ✓`
   - `session created ✓`
   - `run ✓  logits dims=[1,32]  N ms`
   - `WebGPU op coverage OK — this architecture is browser‑runnable ✓`

## Reading the result

- **All green** → the backbone runs on WebGPU. Architecture de‑risked → proceed
  to task #2 (data generation + labeling) and train the real model with this
  same architecture.
- **`FAILED: … unsupported op …`** → onnxruntime‑web's WebGPU EP lacks an op
  (e.g. a pooling/reshape variant). Paste the op name; we factorize the backbone
  (swap the offending op for a WebGPU‑supported equivalent) and re‑export.
- **Latency** — note the `N ms`; it should be small (single forward on a tiny
  model). This is the per‑caption cost the client will pay on cadence.
