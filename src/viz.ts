/** Helpers that turn arrays into pictures for the UI. */
import { jit, nn, numpy as np, random } from "@jax-js/jax";

export type FeatureMap = { width: number; height: number; rgba: Uint8ClampedArray<ArrayBuffer> };

const projections = new Map<number, np.Array>();

/** A fixed random projection from `channels` to 3 colors. */
function projection(channels: number): np.Array {
  let p = projections.get(channels);
  if (!p) {
    p = random.normal(random.key(channels), [3, channels]);
    projections.set(channels, p);
  }
  return p.ref;
}

const colorize = jit((x: np.Array, proj: np.Array) => {
  // [1, C, H, W] -> [H, W, 3]: project channels to colors, then normalize
  // each color over the image so structure shows at every scale.
  const [, , h, w] = x.shape;
  const y = nn.standardize(np.einsum("nchw,dc->nhwd", x, proj), [0, 1, 2]);
  return nn.sigmoid(y.mul(1.5)).mul(255).reshape([h, w, 3]);
});

/** False-color picture of activations [1, C, H, W] (consumed). */
export async function featureMap(x: np.Array): Promise<FeatureMap> {
  const [, c, h, w] = x.shape;
  const rgb = (await colorize(x, projection(c)).data()) as Float32Array;
  return { width: w, height: h, rgba: toRGBA(rgb) };
}

/** RGB floats in 0..255 -> RGBA bytes, truncating like torch's .to(uint8). */
export function toRGBA(pixels: Float32Array): Uint8ClampedArray<ArrayBuffer> {
  const count = pixels.length / 3;
  const rgba = new Uint8ClampedArray(count * 4);
  for (let p = 0; p < count; p++) {
    rgba[4 * p] = pixels[3 * p] | 0;
    rgba[4 * p + 1] = pixels[3 * p + 1] | 0;
    rgba[4 * p + 2] = pixels[3 * p + 2] | 0;
    rgba[4 * p + 3] = 255;
  }
  return rgba;
}
