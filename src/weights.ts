import { type Device, getWebGPUDevice, numpy as np } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";

import type { Weights } from "./model.ts";

export type LoadOptions = {
  device?: Device;
  /** dtype for floating-point tensors (default: as stored). */
  dtype?: np.DType;
};

/**
 * Parse a safetensors file into jax-js arrays. float16 data headed for a
 * float32 array is widened in JS before upload, so it works on devices
 * without float16 support.
 */
export function loadWeights(bytes: Uint8Array<ArrayBuffer>, opts: LoadOptions = {}): Weights {
  const file = safetensors.parse(bytes);
  const weights: Weights = {};
  for (const [name, t] of Object.entries(file.tensors)) {
    let data: Float16Array<ArrayBuffer> | Float32Array<ArrayBuffer> | Int32Array<ArrayBuffer>;
    if (t.dtype === "F16" && opts.dtype === np.float32) data = new Float32Array(t.data as Float16Array);
    else if (t.dtype === "F16" || t.dtype === "F32" || t.dtype === "I32") data = t.data as typeof data;
    else throw new Error(`${name}: unsupported dtype ${t.dtype}`);
    let x = np.array(data, { shape: t.shape, device: opts.device });
    if (opts.dtype && t.dtype !== "I32" && x.dtype !== opts.dtype) x = x.astype(opts.dtype);
    weights[name] = x;
  }
  return weights;
}

/** Whether `device` can compute in float16 (WebGPU needs the shader-f16 feature). */
export function supportsFloat16(device: Device): boolean {
  if (device === "webgpu") return getWebGPUDevice().features.has("shader-f16");
  return device === "cpu";
}
