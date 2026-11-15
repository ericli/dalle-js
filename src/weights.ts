import { type Device, getWebGPUDevice, numpy as np } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";

import type { Weights } from "./model.ts";

export type LoadOptions = {
  device?: Device;
  /** dtype for floating-point tensors (default: as stored). */
  dtype?: np.DType;
};

/**
 * Parse a safetensors file into jax-js arrays.
 *
 * - float16 data headed for a float32 array is widened in JS before upload,
 *   so it works on devices without float16 support.
 * - int8 tensors (from python/quantize.py) come with a `<name>.scale` tensor
 *   holding one float32 scale per row. They're dequantized here, in JS, to
 *   `dtype` (float32 if unset), exactly as quantize.py does.
 */
export function loadWeights(bytes: Uint8Array<ArrayBuffer>, opts: LoadOptions = {}): Weights {
  const file = safetensors.parse(bytes);
  const weights: Weights = {};
  for (const [name, t] of Object.entries(file.tensors)) {
    if (name.endsWith(".scale") && file.tensors[name.slice(0, -".scale".length)]?.dtype === "I8") continue;
    let data: Float16Array<ArrayBuffer> | Float32Array<ArrayBuffer> | Int32Array<ArrayBuffer>;
    if (t.dtype === "I8") {
      const scale = file.tensors[`${name}.scale`];
      if (!scale || scale.dtype !== "F32") throw new Error(`${name}: int8 tensor without a float32 scale`);
      data = dequantizeRows(t.data as Int8Array, scale.data as Float32Array, opts.dtype === np.float16);
    } else if (t.dtype === "F16" && opts.dtype === np.float32) {
      data = new Float32Array(t.data as Float16Array);
    } else if (t.dtype === "F16" || t.dtype === "F32" || t.dtype === "I32") {
      data = t.data as typeof data;
    } else {
      throw new Error(`${name}: unsupported dtype ${t.dtype}`);
    }
    let x = np.array(data, { shape: t.shape, device: opts.device });
    if (opts.dtype && t.dtype !== "I32" && x.dtype !== opts.dtype) x = x.astype(opts.dtype);
    weights[name] = x;
  }
  return weights;
}

/** q[r, :] * scale[r] for a row-major int8 matrix, into float32 or float16. */
function dequantizeRows(
  q: Int8Array,
  scale: Float32Array,
  half: boolean,
): Float32Array<ArrayBuffer> | Float16Array<ArrayBuffer> {
  const out = half ? new Float16Array(q.length) : new Float32Array(q.length);
  const cols = q.length / scale.length;
  for (let r = 0; r < scale.length; r++) {
    const s = scale[r];
    const base = r * cols;
    for (let c = 0; c < cols; c++) out[base + c] = q[base + c] * s;
  }
  return out;
}

/** Whether `device` can compute in float16 (WebGPU needs the shader-f16 feature). */
export function supportsFloat16(device: Device): boolean {
  if (device === "webgpu") return getWebGPUDevice().features.has("shader-f16");
  return device === "cpu";
}
