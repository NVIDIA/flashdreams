# 2b feasibility spike — TAEHV decoder ONNX + WebGPU parity (throwaway)

Not product code. Answers: **can the OmniDreams serving decoder (TAEHV /
LightTAE) run in the browser via onnxruntime-web WebGPU, does it match the
server (torch) decode, and is fp16 fast enough to ship?**

Scope: a single chunk with a fresh (zero) cache. Streaming continuity across
chunks is deferred to productionization. Delete this directory before any PR.

## Step 1 — export both precisions + the reference (GPU box, repo uv env)

```bash
uv run python spike/2b-vae-parity/export_vae_decoder.py
```
Writes into this dir: `vae_decoder.fp32.onnx`, `vae_decoder.fp16.onnx`,
`latent.f32.bin`, `latent.f16.bin`, `reference_rgb.f32.bin` (fp32 torch gold),
`meta.json`.

## Step 2 — run the A/B in a WebGPU browser (Edge on your Ubuntu box)

```bash
cd spike/2b-vae-parity && python3 -m http.server 8099
# if remote:  ssh -L 8099:localhost:8099 <gpu-box>
```
Open both and record the numbers each prints:
- **http://localhost:8099/?p=fp32**  → WARM DECODE (ms) + PSNR (dB)
- **http://localhost:8099/?p=fp16**  → WARM DECODE (ms) + PSNR (dB)

(Hard-refresh / Ctrl+F5 when switching so it re-fetches.)

## What we're comparing

| | fp32 | fp16 |
|---|---|---|
| **Decode time** | baseline (~880 ms last run) | expect ~2× faster — the question |
| **PSNR vs fp32 gold** | execution fidelity (~77 dB) | includes fp16 quantization — expect lower but should stay high |

**Decision:** if fp16 roughly halves the decode time (dropping it below the
server's ~648 ms/chunk generation) with PSNR still comfortably high (visually
lossless for a tiny AE), we productionize the client decoder in **fp16**. If
fp16 barely helps, we reconsider levers (WGSL, output res, fps) before building.

Paste back the two `WARM DECODE` / `PSNR` lines and any console errors.

## Likely iteration points (I can't run these here)

- fp16 export: building TAEHV in `float16` and exporting may warn or error on an
  op; paste it and I'll adjust.
- The fp16 page feeds a `float16` input tensor (Uint16Array) and converts the
  fp16 output back to float for PSNR; if ORT-web rejects the fp16 tensor, tell
  me the message.
- `onnxruntime-web@1.20.1` version / opset-18 model: bump if load fails.
