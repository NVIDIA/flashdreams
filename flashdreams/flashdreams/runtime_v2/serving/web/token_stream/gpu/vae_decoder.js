// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// WebGPU VAE decoder for the token path. Runs the exported cache-as-IO TAEHV
// decoder (onnxruntime-web, WebGPU EP) to turn a chunk of latent frames into
// displayable RGB, threading the temporal cache from one chunk to the next.
//
// The graph exposes the decoder's per-block temporal cache as explicit
// `cache_in_*` inputs and `cache_out_*` outputs (see the `.spec.json` the
// session header carries). The initial cache is all zeros, which is exactly the
// decoder's native cold-start state, so chunk 0 needs no special handling; each
// run's `cache_out_*` become the next run's `cache_in_*`.
//
// Precision: fp32 is the portable default. fp16 (half the model size) needs the
// WebGPU `shader-f16` feature and a native `Float16Array`; it is used only when
// the session advertises an fp16 artifact and the client supports both,
// otherwise the decoder transparently falls back to fp32.

import { applyDisplayTransform } from "./display_transform.js"
import { loadOrt, ORT_VERSION } from "./ort_loader.js"

export class VaeDecoder {
  constructor() {
    this._ort = null
    this._session = null
    this._cache = null // ort.Tensor[] threaded across chunks (cache_in/cache_out)
    this._precision = null
    // Serializes decode() calls: ORT-web run() is not reentrant, and the
    // temporal cache must thread strictly in submission order.
    this._chain = Promise.resolve()
    // GPU-resident temporal cache is the DEFAULT: all ORT outputs stay on the
    // GPU so cache_out threads straight into cache_in with no per-chunk GPU↔CPU
    // round-trip (that round-trip was ~half the decode; keeping the cache on the
    // GPU ~halves decode and makes the token path server-bound). Falls back to
    // the portable CPU-cache output at init if a client can't do it.
    this._gpuResident = true
    this._gpuDevice = null // the device ORT uses for GPU-direct render
    // GPU-direct-to-canvas: when the output is a bindable GPU storage buffer, the
    // render loop draws it straight to a WebGPU canvas (no getData readback, no
    // CPU display transform). Probed at warmup; enabled by the session.
    this._gpuBufferOk = false
    this._directRender = false
    this._log = () => {}
  }

  /** Selected decode precision ("fp16" | "fp32"); set during init(). */
  get precision() {
    return this._precision
  }

  /** Temporal-cache location after init ("gpu-resident" | "cpu" fallback). */
  get cacheMode() {
    return this._gpuResident ? "gpu-resident" : "cpu"
  }

  /** The WebGPU device ORT runs on (owns the output buffer), or null. */
  get device() {
    return this._gpuDevice
  }

  /** Decoder output width (set during init). */
  get outputWidth() {
    return this._out?.width
  }

  /** Decoder output height (set during init). */
  get outputHeight() {
    return this._out?.height
  }

  /** Whether GPU-direct-to-canvas is viable (gpu-resident + bindable buffer). */
  get directRenderCapable() {
    return this._gpuResident && this._gpuBufferOk && !!this._gpuDevice
  }

  /** Switch decode output to GPU-direct (held GPU buffer, no readback). */
  enableDirectRender() {
    this._directRender = true
  }

  /**
   * Prepare the decoder for a session from its `vae_model` descriptor.
   *
   * @param {object} sessionHeader parsed session-header JSON
   * @param {GPUDevice} device WebGPU device from the capability probe
   */
  async init(sessionHeader, device, log = () => {}) {
    const descriptor = sessionHeader?.vae_model
    if (!descriptor) {
      throw new Error("session header carries no vae_model descriptor")
    }
    this._device = device
    this._log = log
    this._ort = await loadOrt(sessionHeader?.ort_runtime)

    // The server only advertises a decoder whose identity matches the launched
    // model; log what we actually loaded so a mismatch (or stale artifact) is
    // visible client-side too.
    this._log(
      `[vae-decoder] descriptor version=${descriptor.version} ` +
        `identity=${JSON.stringify(descriptor.identity ?? null)}`,
      { source: "client" }
    )

    this._precision = await this._selectPrecision(descriptor)
    this._dtype = this._precision === "fp16" ? "float16" : "float32"
    this._ArrayCtor =
      this._precision === "fp16" ? globalThis.Float16Array : Float32Array

    // I/O names + shapes are session-independent. [0] is the latent / rgb; the
    // rest are the cache slots in order.
    this._latentName = descriptor.input_names[0]
    this._rgbName = descriptor.output_names[0]
    this._cacheInNames = descriptor.input_names.slice(1)
    this._cacheOutNames = descriptor.output_names.slice(1)
    // ONNX latent input is [B=1, V=1, T, Cl, Hl, Wl]; descriptor.latent_shape is
    // the [T, Cl, Hl, Wl] tail.
    this._latentShape = [1, 1, ...descriptor.latent_shape]
    this._latentLen = this._latentShape.reduce((a, b) => a * b, 1)
    const [frames, channels, height, width] = descriptor.output_shape
    this._out = { frames, channels, height, width }
    this._cacheSpec = descriptor.cache

    // Honor the spec rather than assuming the decoder's output layout: the
    // display transform renders RGB-3 only, so fail loud for any other channel
    // count instead of silently mis-rendering.
    if (channels !== 3) {
      throw new Error(
        `unsupported decoder output channels=${channels} (only RGB-3 is supported)`
      )
    }
    // Temporal-cache cold-start convention, carried in the spec. Only the
    // all-zeros cold start is implemented (the decoder's native state); a
    // different convention must be handled here, not silently zero-filled.
    this._coldStart = descriptor.cold_start ?? "zeros"
    if (this._coldStart !== "zeros") {
      throw new Error(
        `unsupported decoder cold_start=${this._coldStart} (only "zeros" is supported)`
      )
    }

    const modelUrl = new URL(descriptor.precisions[this._precision], location.href)
      .href
    const buildOpts = () => {
      const opts = { executionProviders: ["webgpu"], graphOptimizationLevel: "all" }
      if (this._gpuResident) {
        // All outputs on the GPU → cache_out stays GPU and threads into cache_in.
        opts.preferredOutputLocation = "gpu-buffer"
      }
      return opts
    }
    const setup = async () => {
      this._session = await this._ort.InferenceSession.create(modelUrl, buildOpts())
      // ORT-web creates and OWNS its WebGPU device (it calls requestDevice itself
      // and redefines env.webgpu.device to point at it), ignoring any device we
      // assigned earlier. Adopt ORT's actual device so GPU-direct binds the output
      // buffer on the SAME device -- otherwise CreateBindGroup fails cross-device.
      let ortDevice = this._ort?.env?.webgpu?.device
      if (ortDevice && typeof ortDevice.then === "function") {
        try {
          ortDevice = await ortDevice
        } catch {
          ortDevice = null
        }
      }
      if (ortDevice && typeof ortDevice.createBindGroup === "function") {
        this._gpuDevice = ortDevice
      }
      this._cache = this._zeroCache(this._cacheSpec)
      await this._warmup()
    }
    try {
      await setup()
    } catch (err) {
      if (!this._gpuResident) {
        throw err
      }
      // Production fallback: the GPU-resident path failed on this client — drop
      // to the portable CPU-cache output so the demo still renders. (This is the
      // default decode path, so graceful degradation is intended.)
      this._log(
        `[vae-decoder] GPU-resident path unavailable (${err?.message}); using CPU-cache fallback`,
        { source: "client" }
      )
      this._gpuResident = false
      try {
        this._ort.env.webgpu.profiling = { mode: "off" }
      } catch {
        // ignore
      }
      await setup()
    }
    this._log(
      `[vae-decoder] ready: precision=${this._precision} ` +
        `cache=${this._gpuResident ? "gpu-resident" : "cpu"} ortver=${ORT_VERSION}`,
      { source: "client" }
    )
  }

  /**
   * Decode one chunk of latent frames into per-frame display-ready RGB.
   *
   * @param {Array<Float16Array | Float32Array>} latentFrames chunk latents,
   *   each frame flattened to [Cl * Hl * Wl]
   * @returns {Promise<{frames: ImageData[], width: number, height: number}>}
   */
  async decode(latentFrames) {
    // Queue behind any in-flight decode so runs never overlap and the cache
    // threads in order. A rejected decode must not wedge the queue, so the
    // chain swallows errors while the returned promise still surfaces them.
    const result = this._chain.then(() => {
      return this._decodeOne(latentFrames)
    })
    this._chain = result.catch(() => {})
    return result
  }

  async _decodeOne(latentFrames) {
    if (this._gpuResident) {
      if (this._directRender) {
        // GPU-direct: keep the output GPU buffer, no readback / transform.
        return this._decodeOneGpuDirect(latentFrames)
      }
      // Default path: outputs (incl. RGB) are GPU buffers, read back to CPU.
      return this._decodeOneGpu(latentFrames)
    }
    return this._decodeOneCpu(latentFrames)
  }

  // Default decode: GPU-resident cache + GPU RGB output. The RGB output is a GPU
  // buffer (preferredOutputLocation), so read it back to CPU explicitly; the
  // cache stays on the GPU and threads forward in `_run`.
  async _decodeOneGpu(latentFrames) {
    const rgb = await this._run(this._assembleLatent(latentFrames))
    const data = await rgb.getData(true) // GPU→CPU readback of the RGB buffer
    const frames = applyDisplayTransform({
      data,
      frames: this._out.frames,
      channels: this._out.channels,
      height: this._out.height,
      width: this._out.width,
    })
    return {
      frames,
      width: this._out.width,
      height: this._out.height,
    }
  }

  // GPU-direct: run the decode and hand the render loop the output GPU buffer
  // (planar [T, 3, H, W] fp32, [-1, 1]) WITHOUT reading it back or running the CPU
  // display transform -- the render loop draws it straight to the canvas. The ORT
  // output tensor is held alive (so ORT does not recycle its buffer) and disposed
  // by the render loop after the chunk's last frame is presented. Per the no-copy
  // decision, a missing/unbindable buffer throws (-> session falls back to CPU).
  async _decodeOneGpuDirect(latentFrames) {
    const latent = this._assembleLatent(latentFrames)
    const rgb = await this._run(latent)
    // FUNCTIONAL wait (not profiling): the CPU path implicitly waited for the GPU
    // inside getData(), so "ack on buffer" meant the GPU was done and the 4-chunk
    // flow window bounded GPU work + held buffers. Without this, acks fire at
    // submit time, so on a slow client the GPU queue and the held 86 MB buffers
    // grow unbounded (latency creep, VRAM growth, device loss). Also keeps
    // profiled and unprofiled behavior identical.
    if (this._gpuDevice) {
      try {
        await this._gpuDevice.queue.onSubmittedWorkDone()
      } catch {
        // best-effort
      }
    }
    const buffer = rgb?.gpuBuffer
    if (!buffer) {
      try {
        rgb?.dispose?.()
      } catch {
        // best-effort
      }
      throw new Error("decoder output exposes no gpuBuffer; GPU-direct not usable")
    }
    return {
      gpu: true,
      buffer,
      frameCount: this._out.frames,
      width: this._out.width,
      height: this._out.height,
      // Held until the render loop presents the last frame, then disposed.
      dispose: () => {
        try {
          rgb.dispose?.()
        } catch {
          // best-effort
        }
      },
    }
  }

  // Portable fallback (client can't keep outputs on the GPU): the cache and RGB
  // are CPU tensors, so `session.run()` already resolves with CPU data — read it
  // directly, no readback call. `_run` awaits run(), which resolves only after
  // the GPU work completes and the output is read back, so this wall-clock span
  // is the true decode latency.
  async _decodeOneCpu(latentFrames) {
    const rgb = await this._run(this._assembleLatent(latentFrames))
    const frames = applyDisplayTransform({
      data: rgb.data,
      frames: this._out.frames,
      channels: this._out.channels,
      height: this._out.height,
      width: this._out.width,
    })
    return {
      frames,
      width: this._out.width,
      height: this._out.height,
    }
  }

  // Conservative estimate of the decode's largest WebGPU buffer: a full-resolution
  // activation ~[frames, C, H, W] with C≈64 for the TAEHV decoder (8*64*704*1280*4
  // ≈ 1.85 GB fp32 at 1280×704, matching the observed CreateBuffer size). Used to
  // refuse up front with an actionable message instead of hitting an uncaught
  // buffer-limit validation error that just blanks the screen.
  _peakBufferBytes(descriptor, precision) {
    const [frames, , height, width] = descriptor.output_shape
    const channels = 64
    const bytesPerElement = precision === "fp16" ? 2 : 4
    return frames * channels * height * width * bytesPerElement
  }

  async _selectPrecision(descriptor) {
    const preferred = descriptor.default_precision || "fp32"
    const wantsFp16 =
      !!descriptor.precisions.fp16 && typeof globalThis.Float16Array === "function"

    let adapter = null
    try {
      adapter = await navigator.gpu.requestAdapter()
    } catch {
      adapter = null
    }
    // fp16 only when the session offers it and the client can run the half-
    // precision shaders (Dawn/Linux does not expose shader-f16 today); else fp32.
    const useFp16 =
      !!adapter &&
      wantsFp16 &&
      adapter.features.has("shader-f16") &&
      !!descriptor.precisions.fp16
    const precision = useFp16
      ? "fp16"
      : descriptor.precisions.fp32
        ? "fp32"
        : preferred

    if (adapter) {
      // Fail loud BEFORE the cryptic CreateBuffer error: if the adapter can't even
      // grant a buffer big enough for the decode's largest activation (e.g. the
      // 1 GB default when the Vulkan backend isn't active), decoding is impossible
      // on this client — say exactly how to fix it, rather than blanking.
      const needed = this._peakBufferBytes(descriptor, precision)
      if (needed > adapter.limits.maxBufferSize) {
        const [, , height, width] = descriptor.output_shape
        throw new Error(
          `WebGPU maxBufferSize ${adapter.limits.maxBufferSize} < ~${needed} needed ` +
            `for ${precision} decode at ${width}×${height} — relaunch the browser with ` +
            `--use-angle=vulkan --enable-features=Vulkan to expose the full buffer limit`
        )
      }
      // Reuse the device ORT already runs on when it is big enough. ORT-web's
      // WebGPU backend is a singleton, so after a reset re-inits the decoder, a
      // second device we create is NOT the one ORT's output buffers live on --
      // binding them for GPU-direct would then fail cross-device. Only create a
      // device when ORT has none big enough yet (the first init).
      let existing = this._ort?.env?.webgpu?.device
      if (existing && typeof existing.then === "function") {
        try {
          existing = await existing
        } catch {
          existing = null
        }
      }
      if (
        existing &&
        typeof existing.createBindGroup === "function" &&
        (existing.limits?.maxBufferSize ?? 0) >= needed
      ) {
        this._gpuDevice = existing // functional: GPU-direct + GPU-completion wait
        this._log("[vae-decoder] reusing ORT's existing WebGPU device", {
          source: "client",
        })
      } else {
        // Create the device with the adapter's MAX buffer limits (both
        // precisions); ORT-web's default device (1 GB) is too small for the
        // ~1.8 GB fp32 activation. fp16 also needs shader-f16.
        try {
          const features = []
          if (useFp16) {
            features.push("shader-f16")
          }
          const device = await adapter.requestDevice({
            requiredFeatures: features,
            requiredLimits: {
              maxBufferSize: adapter.limits.maxBufferSize,
              maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            },
          })
          this._ort.env.webgpu.device = device
          this._gpuDevice = device // functional: GPU-direct + GPU-completion wait
        } catch (err) {
          // fall through: let ORT create its own default device (may hit the
          // buffer limit, but better than not starting the decoder at all)
          this._log(
            `[vae-decoder] custom WebGPU device setup failed (${err?.message}); using ORT default device`,
            { source: "client" }
          )
        }
      }
    }
    return precision
  }

  _zeroCache(cacheSpec) {
    return cacheSpec.map((slot) => {
      const count = slot.shape.reduce((a, b) => a * b, 1)
      return new this._ort.Tensor(this._dtype, new this._ArrayCtor(count), slot.shape)
    })
  }

  _assembleLatent(latentFrames) {
    const buffer = new this._ArrayCtor(this._latentLen)
    let offset = 0
    for (const frame of latentFrames) {
      buffer.set(frame, offset) // widens/narrows to the tensor dtype element-wise
      offset += frame.length
    }
    return new this._ort.Tensor(this._dtype, buffer, this._latentShape)
  }

  async _run(latentTensor) {
    const feeds = { [this._latentName]: latentTensor }
    this._cacheInNames.forEach((name, i) => {
      feeds[name] = this._cache[i]
    })
    const results = await this._session.run(feeds)
    const prev = this._cache
    // Thread the temporal cache forward for the next chunk.
    this._cache = this._cacheOutNames.map((name) => results[name])
    // GPU-resident cache: free the previous run's GPU cache tensors so they
    // don't accumulate across chunks. (CPU-fallback tensors are GC'd normally.)
    if (this._gpuResident && prev) {
      for (const t of prev) {
        try {
          t?.dispose?.()
        } catch {
          // best-effort
        }
      }
    }
    return results[this._rgbName]
  }

  async _warmup() {
    // Two decodes on a zero latent. The first compiles the WebGPU shaders; the
    // second is fed the first's cache_out, so warmup actually exercises the
    // GPU-resident cache_out → cache_in handoff — the exact step that first
    // fails on a client that can't keep the cache on the GPU. Running it here,
    // inside init()'s try, lets such a failure trigger the CPU-cache fallback,
    // instead of surfacing only on the 2nd real chunk after init() has already
    // reported success (which would reject every decode → a blank video).
    const zero = () =>
      new this._ort.Tensor(
        this._dtype,
        new this._ArrayCtor(this._latentLen),
        this._latentShape,
      )
    for (let i = 0; i < 2; i++) {
      const rgb = await this._run(zero())
      if (i === 1 && this._gpuResident) {
        // Probe whether the output is a GPU buffer we can bind read-only for
        // GPU-direct-to-canvas (needs STORAGE usage). Decided once, here. A probe
        // failure only disables GPU-direct; it must not break gpu-resident decode.
        try {
          const buf = rgb?.gpuBuffer
          const usage = buf?.usage ?? 0
          const hasStorage =
            typeof GPUBufferUsage !== "undefined" &&
            (usage & GPUBufferUsage.STORAGE) !== 0
          this._gpuBufferOk = !!(buf && hasStorage)
          this._log(
            `[vae-decoder] gpu-direct probe: gpuBuffer=${!!buf} ` +
              `usage=0x${usage.toString(16)} STORAGE=${hasStorage} ` +
              `size=${buf?.size ?? "n/a"}`,
            { source: "client" }
          )
        } catch (error) {
          this._gpuBufferOk = false
          this._log(`[vae-decoder] gpu-direct probe failed: ${error.message}`, {
            source: "client",
          })
        }
      }
      try {
        rgb?.dispose?.() // free the warmup RGB (a GPU buffer in resident mode)
      } catch {
        // best-effort
      }
    }
    // _run has advanced this._cache to the 2nd warmup's cache_out (and, in
    // resident mode, disposed the 1st). Dispose that final set too, then reset
    // to a FRESH zero cold-start cache so the first real chunk starts clean —
    // restoring a disposed reference would throw "invalid data location: none".
    if (this._gpuResident && this._cache) {
      for (const t of this._cache) {
        try {
          t?.dispose?.()
        } catch {
          // best-effort
        }
      }
    }
    this._cache = this._zeroCache(this._cacheSpec)
  }
}
