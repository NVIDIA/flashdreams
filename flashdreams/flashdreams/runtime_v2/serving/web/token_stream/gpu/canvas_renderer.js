// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// GPU-direct presentation for the token path. The exported decoder emits RGB as
// a GPU buffer (planar [T, 3, H, W] fp32 in [-1, 1]); instead of reading it back
// to the CPU (getData) and running the display transform on the CPU, this draws
// a frame straight from that buffer to a WebGPU-backed canvas, applying the
// [-1, 1] -> [0, 1] display remap in the fragment shader. That removes the
// GPU->CPU readback and the CPU transform -- the two phases that, measured,
// dominate the client decode on gpu-resident clients.
//
// The buffer lives on the device ORT runs on, so the canvas context and the
// pipeline MUST use that same device (WebGPU resources do not cross devices).
// Any setup failure throws so the caller falls back to the CPU-readback path.

// Fullscreen triangle; the fragment shader reads planar RGB and remaps to RGBA.
const _SHADER = /* wgsl */ `
struct Params {
  frameBase : u32,  // element offset of this frame's R plane
  width     : u32,
  height    : u32,
  hw        : u32,  // width * height (plane stride)
};
@group(0) @binding(0) var<storage, read> rgb : array<f32>;
@group(0) @binding(1) var<uniform> p : Params;

@vertex
fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4f {
  var corners = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(corners[i], 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos : vec4f) -> @location(0) vec4f {
  let x = u32(pos.x);
  let y = u32(pos.y);
  if (x >= p.width || y >= p.height) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }
  let pix = y * p.width + x;
  let r = rgb[p.frameBase + pix];
  let g = rgb[p.frameBase + p.hw + pix];
  let b = rgb[p.frameBase + 2u * p.hw + pix];
  // Display remap: [-1, 1] -> [0, 1], clamped (the canvas format quantizes to 8-bit).
  let c = clamp((vec3f(r, g, b) + 1.0) * 0.5, vec3f(0.0), vec3f(1.0));
  return vec4f(c, 1.0);
}
`

export class WebGpuCanvasRenderer {
  /**
   * Build a renderer, or throw (so the caller falls back to the CPU path).
   *
   * @param {GPUDevice} device the device ORT runs on (owns the output buffer)
   * @param {HTMLCanvasElement} canvas target canvas (becomes a WebGPU context)
   * @param {object} dims
   * @param {number} dims.width decoder output width
   * @param {number} dims.height decoder output height
   * @param {object} [opts]
   * @param {Function} [opts.log] logger for async (uncaptured) GPU errors
   * @returns {Promise<WebGpuCanvasRenderer>}
   */
  static async create(device, canvas, { width, height }, { log } = {}) {
    if (!device || !canvas || !width || !height) {
      throw new Error("canvas renderer needs a device, canvas, and dimensions")
    }
    const format = navigator.gpu.getPreferredCanvasFormat()

    // Validate the shader + pipeline BEFORE touching the canvas context. WebGPU
    // reports these errors asynchronously, and a canvas's context type is fixed
    // on the first getContext() -- so a pipeline failure must not lock the canvas
    // to WebGPU (that would block the CPU 2D fallback and leave a black screen).
    const module = device.createShaderModule({ code: _SHADER })
    const info = await module.getCompilationInfo()
    const shaderError = info.messages.find((m) => m.type === "error")
    if (shaderError) {
      throw new Error(`shader compile: ${shaderError.message}`)
    }
    device.pushErrorScope("validation")
    let pipeline
    try {
      pipeline = await device.createRenderPipelineAsync({
        layout: "auto",
        vertex: { module, entryPoint: "vs" },
        fragment: { module, entryPoint: "fs", targets: [{ format }] },
        primitive: { topology: "triangle-list" },
      })
    } catch (error) {
      await device.popErrorScope()
      throw new Error(`pipeline: ${error.message}`)
    }
    const pipelineError = await device.popErrorScope()
    if (pipelineError) {
      throw new Error(`pipeline validation: ${pipelineError.message}`)
    }

    // Pipeline is valid; now claim + configure the canvas (this locks it to WebGPU).
    const context = canvas.getContext("webgpu")
    if (!context) {
      throw new Error("canvas.getContext('webgpu') unavailable")
    }
    canvas.width = width // native decode resolution; CSS scales to the viewport
    canvas.height = height
    device.pushErrorScope("validation")
    context.configure({ device, format, alphaMode: "opaque" })
    const configError = await device.popErrorScope()
    if (configError) {
      throw new Error(`canvas configure: ${configError.message}`)
    }
    const params = device.createBuffer({
      size: 16, // 4 x u32
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    // Surface async (first-present) binding errors WebGPU reports out-of-band.
    if (typeof device.addEventListener === "function") {
      device.addEventListener("uncapturederror", (event) => {
        const message = `GPU-direct uncaptured error: ${event.error?.message ?? event.error}`
        if (log) {
          log(message, { source: "client", level: "error" })
        } else {
          console.error("[token]", message)
        }
      })
    }
    return new WebGpuCanvasRenderer(device, context, pipeline, params, width, height)
  }

  constructor(device, context, pipeline, paramsBuffer, width, height) {
    this._device = device
    this._context = context
    this._pipeline = pipeline
    this._params = paramsBuffer
    this._width = width
    this._height = height
    this._hw = width * height
    this._paramsScratch = new Uint32Array(4)
  }

  /**
   * Draw one frame of a chunk's decoder-output buffer to the canvas.
   *
   * @param {GPUBuffer} buffer the chunk's RGB output (planar [T, 3, H, W] fp32)
   * @param {number} frameIndex which frame T to present
   */
  present(buffer, frameIndex) {
    const frameBase = frameIndex * 3 * this._hw
    this._paramsScratch[0] = frameBase
    this._paramsScratch[1] = this._width
    this._paramsScratch[2] = this._height
    this._paramsScratch[3] = this._hw
    this._device.queue.writeBuffer(this._params, 0, this._paramsScratch)

    // The bind group references this chunk's buffer, so it is built per present.
    const bindGroup = this._device.createBindGroup({
      layout: this._pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer } },
        { binding: 1, resource: { buffer: this._params } },
      ],
    })
    const encoder = this._device.createCommandEncoder()
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this._context.getCurrentTexture().createView(),
          loadOp: "clear",
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          storeOp: "store",
        },
      ],
    })
    pass.setPipeline(this._pipeline)
    pass.setBindGroup(0, bindGroup)
    pass.draw(3)
    pass.end()
    this._device.queue.submit([encoder.finish()])
  }

  destroy() {
    try {
      this._params?.destroy?.()
    } catch {
      // best-effort
    }
    this._context = null
    this._pipeline = null
  }
}
