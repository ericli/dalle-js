/**
 * DALL·E Mini/Mega in jax-js, ported from min-dalle's PyTorch models
 * (dalle_bart_encoder.py, dalle_bart_decoder.py, vqgan_detokenizer.py).
 *
 * Ownership follows jax-js rules: every function consumes its Array
 * arguments, except `weights`, which is borrowed: each use takes a new ref
 * through `w(name)`. Weights keep PyTorch layout (Linear [out, in], conv OIHW).
 *
 * Two departures from min-dalle that don't change the numbers:
 * - Cross-attention keys/values of the (fixed) encoder state are projected
 *   once per prompt instead of at every decoder step.
 * - The self-attention KV cache is a fixed [rows, 256, embed] array per layer,
 *   written with `where(position == i, ...)` so the step jits once.
 */
import { jit, lax, nn, numpy as np, random, tree } from "@jax-js/jax";

export type Weights = Record<string, np.Array>;

export type Config = {
  layer_count: number;
  attention_head_count: number;
  embed_count: number;
  glu_embed_count: number;
  text_vocab_count: number;
  image_vocab_count: number;
  text_token_count: number;
  image_token_count: number;
};

export const IMAGE_TOKEN_COUNT = 256;
/** min-dalle starts decoding from 2**14 - 1 (not the 16384 BOS of the original). */
export const START_TOKEN = 2 ** 14 - 1;

function getter(weights: Weights, prefix = "") {
  return (name: string): np.Array => {
    const x = weights[prefix + name];
    if (!x) throw new Error(`missing weight ${prefix + name}`);
    return x.ref;
  };
}

/** x @ w.T for a PyTorch Linear weight w of shape [out, in]. */
function linear(x: np.Array, w: np.Array): np.Array {
  return np.matmul(x, w.transpose());
}

/**
 * LayerNorm over the last axis (eps 1e-5, like nn.LayerNorm). In float16 the
 * statistics are computed in float32, as torch autocast does.
 */
function layerNorm(x: np.Array, scale: np.Array, bias: np.Array): np.Array {
  const dtype = x.dtype;
  const y = dtype === np.float16 ? nn.standardize(x.astype(np.float32), -1).astype(dtype) : nn.standardize(x, -1);
  return y.mul(scale).add(bias);
}

/** Multi-head attention on [B, L, E] queries and [B, S, E] keys/values. */
function attention(q: np.Array, k: np.Array, v: np.Array, mask: np.Array, heads: number): np.Array {
  const [b, l, e] = q.shape;
  const s = k.shape[1];
  const out = nn.dotProductAttention(
    q.reshape([b, l, heads, e / heads]),
    k.reshape([b, s, heads, e / heads]),
    v.reshape([b, s, heads, e / heads]),
    { mask },
  );
  return out.reshape([b, l, e]);
}

/**
 * Attention that also returns its weights averaged over heads, [B, L, S]
 * (for visualization). Softmax runs in float32, as under torch autocast.
 */
function attentionWithWeights(
  q: np.Array,
  k: np.Array,
  v: np.Array,
  mask: np.Array,
  heads: number,
): [np.Array, np.Array] {
  const [b, l, e] = q.shape;
  const s = k.shape[1];
  const d = e / heads;
  const dtype = q.dtype;
  const scores = np.einsum(
    "bqhc,bkhc->bhqk",
    q.reshape([b, l, heads, d]).mul(1 / Math.sqrt(d)),
    k.reshape([b, s, heads, d]),
  );
  const weights = nn.softmax(np.where(mask, scores.astype(np.float32), -Infinity), -1);
  const out = np.einsum("bhqk,bkhc->bqhc", weights.ref.astype(dtype), v.reshape([b, s, heads, d]));
  return [out.reshape([b, l, e]), weights.mean(1)];
}

function glu(x: np.Array, w: (name: string) => np.Array): np.Array {
  x = layerNorm(x, w("ln0.weight"), w("ln0.bias"));
  const gate = nn.gelu(linear(x.ref, w("fc0.weight")), { approximate: false });
  const value = linear(x, w("fc1.weight"));
  x = layerNorm(gate.mul(value), w("ln1.weight"), w("ln1.bias"));
  return linear(x, w("fc2.weight"));
}

// ---------------------------------------------------------------------------
// Encoder

/** Padding mask for attention over text tokens: [rows, 1, 1, 64], true = attend. */
export function textMask(textTokens: np.Array): np.Array {
  const [rows, count] = textTokens.shape;
  return textTokens.notEqual(1).reshape([rows, 1, 1, count]);
}

/** text tokens [rows, 64] int32 -> encoder state [rows, 64, embed]. */
export function encode(weights: Weights, textTokens: np.Array, cfg: Config): np.Array {
  const w = getter(weights);
  const mask = textMask(textTokens.ref);
  let x = np.take(w("embed_tokens.weight"), textTokens, 0).add(w("embed_positions.weight"));
  x = layerNorm(x, w("layernorm_embedding.weight"), w("layernorm_embedding.bias"));
  for (let i = 0; i < cfg.layer_count; i++) {
    const l = getter(weights, `layers.${i}.`);
    let h = layerNorm(x.ref, l("pre_self_attn_layer_norm.weight"), l("pre_self_attn_layer_norm.bias"));
    h = attention(
      linear(h.ref, l("self_attn.q_proj.weight")),
      linear(h.ref, l("self_attn.k_proj.weight")),
      linear(h, l("self_attn.v_proj.weight")),
      mask.ref,
      cfg.attention_head_count,
    );
    h = linear(h, l("self_attn.out_proj.weight"));
    h = layerNorm(h, l("self_attn_layer_norm.weight"), l("self_attn_layer_norm.bias"));
    x = x.add(h);
    x = x.ref.add(glu(x, getter(weights, `layers.${i}.glu.`)));
  }
  mask.dispose();
  return layerNorm(x, w("final_ln.weight"), w("final_ln.bias"));
}

// ---------------------------------------------------------------------------
// Decoder

/** Per-prompt cross-attention inputs, computed once after the encoder. */
export type CrossAttention = {
  /** Keys and values per layer, each [rows, 64, embed]. */
  k: np.Array[];
  v: np.Array[];
  /** Encoder padding mask [rows, 1, 1, 64]. */
  mask: np.Array;
};

/** Self-attention KV cache per layer, each [rows, 256, embed]. */
export type KVCache = { k: np.Array[]; v: np.Array[] };

/**
 * Project the encoder state for cross-attention and allocate an empty cache.
 * Rows are [unconditioned x imageCount, conditioned x imageCount].
 */
export function initDecoder(
  weights: Weights,
  encoderState: np.Array,
  textTokens: np.Array,
  imageCount: number,
  cfg: Config,
): { cross: CrossAttention; cache: KVCache } {
  const rowIndex = np.array(
    Array.from({ length: 2 * imageCount }, (_, i) => (i < imageCount ? 0 : 1)),
    { dtype: np.int32 },
  );
  const enc = np.take(encoderState, rowIndex.ref, 0);
  const cross: CrossAttention = { k: [], v: [], mask: textMask(np.take(textTokens, rowIndex, 0)) };
  for (let i = 0; i < cfg.layer_count; i++) {
    const l = getter(weights, `layers.${i}.encoder_attn.`);
    cross.k.push(linear(enc.ref, l("k_proj.weight")));
    cross.v.push(linear(enc.ref, l("v_proj.weight")));
  }
  enc.dispose();
  const dtype = cross.k[0].dtype;
  const shape = [2 * imageCount, IMAGE_TOKEN_COUNT, cfg.embed_count];
  const zeros = () => Array.from({ length: cfg.layer_count }, () => np.zeros(shape, { dtype }));
  return { cross, cache: { k: zeros(), v: zeros() } };
}

/**
 * One decoder step: previous image tokens [imageCount] int32 at position
 * `index` (int32 scalar) -> logits [2 * imageCount, image_vocab + 1] and the
 * updated cache. Borrows `weights`; consumes everything else.
 *
 * Also returns `crossAttention` [imageCount, 64]: how much the conditioned
 * rows attend to each text token, averaged over heads and layers.
 */
export function decodeStep(
  weights: Weights,
  cross: CrossAttention,
  cache: KVCache,
  prevTokens: np.Array,
  index: np.Array,
  cfg: Config,
): { logits: np.Array; cache: KVCache; crossAttention: np.Array } {
  const w = getter(weights);
  const imageCount = prevTokens.shape[0];
  const rows = 2 * imageCount;
  const heads = cfg.attention_head_count;

  // prev_tokens.repeat(2, 1): unconditioned and conditioned rows share tokens.
  let x = np.take(w("embed_tokens.weight"), np.concatenate([prevTokens.ref, prevTokens]), 0);
  x = x.add(np.take(w("embed_positions.weight"), index.ref.reshape([1]), 0));
  x = layerNorm(x.reshape([rows, 1, cfg.embed_count]), w("layernorm_embedding.weight"), w("layernorm_embedding.bias"));

  const positions = np.arange(IMAGE_TOKEN_COUNT).astype(np.int32);
  const selfMask = positions.ref.lessEqual(index.ref).reshape([1, 1, 1, IMAGE_TOKEN_COUNT]);
  const writeHere = positions.equal(index).reshape([1, IMAGE_TOKEN_COUNT, 1]);

  const next: KVCache = { k: [], v: [] };
  let attentionSum: np.Array | null = null;
  for (let i = 0; i < cfg.layer_count; i++) {
    const l = getter(weights, `layers.${i}.`);

    let h = layerNorm(x.ref, l("pre_self_attn_layer_norm.weight"), l("pre_self_attn_layer_norm.bias"));
    const k = np.where(writeHere.ref, linear(h.ref, l("self_attn.k_proj.weight")), cache.k[i]);
    const v = np.where(writeHere.ref, linear(h.ref, l("self_attn.v_proj.weight")), cache.v[i]);
    next.k.push(k.ref);
    next.v.push(v.ref);
    h = attention(linear(h, l("self_attn.q_proj.weight")), k, v, selfMask.ref, heads);
    h = linear(h, l("self_attn.out_proj.weight"));
    h = layerNorm(h, l("self_attn_layer_norm.weight"), l("self_attn_layer_norm.bias"));
    x = x.add(h);

    h = layerNorm(x.ref, l("pre_encoder_attn_layer_norm.weight"), l("pre_encoder_attn_layer_norm.bias"));
    let attn: np.Array;
    [h, attn] = attentionWithWeights(linear(h, l("encoder_attn.q_proj.weight")), cross.k[i], cross.v[i], cross.mask.ref, heads);
    attentionSum = attentionSum ? attentionSum.add(attn) : attn;
    h = linear(h, l("encoder_attn.out_proj.weight"));
    h = layerNorm(h, l("encoder_attn_layer_norm.weight"), l("encoder_attn_layer_norm.bias"));
    x = x.add(h);

    x = x.ref.add(glu(x, getter(weights, `layers.${i}.glu.`)));
  }
  cross.mask.dispose();
  selfMask.dispose();
  writeHere.dispose();

  x = layerNorm(x, w("final_ln.weight"), w("final_ln.bias"));
  const logits = linear(x, w("lm_head.weight")).reshape([rows, cfg.image_vocab_count + 1]);
  const crossAttention = attentionSum!
    .mul(1 / cfg.layer_count)
    .reshape([rows, -1])
    .slice([imageCount, rows]);
  return { logits, cache: next, crossAttention };
}

export type SampleSettings = { temperature: number; topK: number; superconditionFactor: number };

/** How many of the most likely candidates `sampleTokens` reports. */
export const CANDIDATE_COUNT = 8;

/**
 * min-dalle's sample_tokens: superconditioned logits, top-k filter,
 * temperature, then a categorical draw. logits [2n, vocab + 1] ->
 * tokens [n], plus the top candidates [n, 8] and their probabilities.
 */
export function sampleTokens(
  logits: np.Array,
  key: np.Array,
  settings: SampleSettings,
): { tokens: np.Array; candidates: np.Array; probabilities: np.Array } {
  const n = logits.shape[0] / 2;
  const vocab = 2 ** 14;
  logits = logits.astype(np.float32);
  const uncond = logits.ref.slice([0, n], [0, vocab]);
  const cond = logits.slice([n, 2 * n], [0, vocab]);
  const f = settings.superconditionFactor;
  const combined = uncond.mul(1 - f).add(cond.mul(f));
  const [top, topIndex] = lax.topK(combined.ref, settings.topK, -1);
  const kth = top.ref.slice([], [settings.topK - 1, settings.topK]); // [n, 1]
  const max = top.slice([], [0, 1]);
  const scaled = combined.ref.sub(max).div(settings.temperature);
  const filtered = np.where(combined.greaterEqual(kth), scaled, -Infinity);
  const tokens = random.categorical(key, filtered.ref, { axis: -1 });
  const candidates = topIndex.slice([], [0, CANDIDATE_COUNT]);
  const probabilities = np.takeAlongAxis(nn.softmax(filtered, -1), candidates.ref, -1);
  return { tokens, candidates, probabilities };
}

// ---------------------------------------------------------------------------
// VQGAN detokenizer

function groupNorm(x: np.Array, scale: np.Array, bias: np.Array, groups = 32): np.Array {
  const [n, c, h, wd] = x.shape;
  const y = nn.standardize(x.reshape([n, groups, (c / groups) * h * wd]), -1).reshape([n, c, h, wd]);
  return y.mul(scale.reshape([1, c, 1, 1])).add(bias.reshape([1, c, 1, 1]));
}

function conv2d(x: np.Array, weight: np.Array, bias: np.Array): np.Array {
  const pad = (weight.shape[2] - 1) / 2;
  const y = lax.conv(x, weight, [1, 1], [[pad, pad], [pad, pad]]);
  return y.add(bias.reshape([1, -1, 1, 1]));
}

function swish(x: np.Array): np.Array {
  return x.ref.mul(nn.sigmoid(x));
}

function resnetBlock(x: np.Array, w: (name: string) => np.Array, hasShortcut: boolean): np.Array {
  let h = swish(groupNorm(x.ref, w("norm1.weight"), w("norm1.bias")));
  h = conv2d(h, w("conv1.weight"), w("conv1.bias"));
  h = swish(groupNorm(h, w("norm2.weight"), w("norm2.bias")));
  h = conv2d(h, w("conv2.weight"), w("conv2.bias"));
  if (hasShortcut) x = conv2d(x, w("nin_shortcut.weight"), w("nin_shortcut.bias"));
  return x.add(h);
}

function attentionBlock(x: np.Array, w: (name: string) => np.Array): np.Array {
  const [m, c, hh, ww] = x.shape;
  const h = groupNorm(x.ref, w("norm.weight"), w("norm.bias"));
  // [m, c, h, w] -> [m, h*w, 1 head, c]
  const tokens = (y: np.Array) => y.reshape([m, c, hh * ww]).transpose([0, 2, 1]).reshape([m, hh * ww, 1, c]);
  const q = tokens(conv2d(h.ref, w("q.weight"), w("q.bias")));
  const k = tokens(conv2d(h.ref, w("k.weight"), w("k.bias")));
  const v = tokens(conv2d(h, w("v.weight"), w("v.bias")));
  const out = nn.dotProductAttention(q, k, v).reshape([m, hh * ww, c]).transpose([0, 2, 1]).reshape([m, c, hh, ww]);
  return x.add(conv2d(out, w("proj_out.weight"), w("proj_out.bias")));
}

function upsample(x: np.Array, w: (name: string) => np.Array): np.Array {
  x = np.repeat(np.repeat(x, 2, 2), 2, 3); // nearest-neighbor 2x
  return conv2d(x, w("conv.weight"), w("conv.bias"));
}

/** (log2 in channels, log2 out channels, attention, upsample) for decoder.up[i]. */
const UP_BLOCKS: [number, number, boolean, boolean][] = [
  [7, 7, false, false],
  [8, 7, false, true],
  [8, 8, false, true],
  [9, 8, false, true],
  [9, 9, true, true],
];

/**
 * The weights whose names start with one of `prefixes`, with `strip` removed
 * from the front of each name. Borrowed: no refs are taken.
 */
function pick(weights: Weights, prefixes: string[], strip = ""): Weights {
  const out: Weights = {};
  for (const [k, v] of Object.entries(weights)) {
    if (prefixes.some((p) => k.startsWith(p))) out[k.slice(strip.length)] = v;
  }
  return out;
}

/** Wrap a block that borrows its weights so that it can be jitted (see below). */
function block(f: (w: (name: string) => np.Array, x: np.Array, weights: Weights) => np.Array) {
  return (weights: Weights, x: np.Array) => {
    const out = f(getter(weights), x, weights);
    tree.dispose(weights);
    return out;
  };
}

const vqgan = {
  input: block((w, tokens) => {
    const n = tokens.shape[0];
    let z = np.take(w("embedding.weight"), tokens, 0); // [n, 256 tokens, 256 channels]
    z = z.reshape([n, 16, 16, 256]).transpose([0, 3, 1, 2]);
    z = conv2d(z, w("post_quant_conv.weight"), w("post_quant_conv.bias"));
    return conv2d(z, w("decoder.conv_in.weight"), w("decoder.conv_in.bias"));
  }),
  resnet: block((w, x, weights) => resnetBlock(x, w, "nin_shortcut.weight" in weights)),
  attention: block((w, x) => attentionBlock(x, w)),
  upsample: block((w, x) => upsample(x, w)),
  output: block((w, z) => {
    z = swish(groupNorm(z, w("norm_out.weight"), w("norm_out.bias")));
    z = conv2d(z, w("conv_out.weight"), w("conv_out.bias"));
    return np.clip(z.transpose([0, 2, 3, 1]), 0, 1).mul(255);
  }),
};

let jittedBlocks: typeof vqgan | null = null;

/** The VQGAN blocks, jitted once and shared (compiled per input shape). */
function jittedVqgan(): typeof vqgan {
  jittedBlocks ??= {
    input: jit(vqgan.input),
    resnet: jit(vqgan.resnet),
    attention: jit(vqgan.attention),
    upsample: jit(vqgan.upsample),
    output: jit(vqgan.output),
  };
  return jittedBlocks;
}

/** Named VQGAN stages reported to `onStage`, from 16x16 up to 256x256. */
export const VQGAN_STAGES = ["conv_in", "mid", "up4", "up3", "up2", "up1", "up0"] as const;
export type VqganStage = (typeof VQGAN_STAGES)[number];

export type Detokenizer = (
  weights: Weights,
  imageTokens: np.Array,
  onStage?: (stage: VqganStage, activations: np.Array) => void | Promise<void>,
) => Promise<np.Array>;

/**
 * VQGAN: image tokens [n, 256] int32 -> images [n, 256, 256, 3] float32 in
 * 0..255. Borrows `weights`, consumes `imageTokens`.
 *
 * Runs block by block with a sync after each, because jax-js is lazy: queued
 * ops keep their inputs alive until dispatch, and the whole VQGAN graph at
 * once doesn't fit (wasm hits its 4 GiB limit). Must run in float32: the
 * residual stream in up[0] exceeds the float16 range.
 *
 * `onStage` receives the activations [n, C, H, W] after each stage in
 * VQGAN_STAGES; it owns that reference and must dispose it.
 * `useJit = false` runs the same blocks eagerly, for debugging.
 */
export function makeDetokenizer(useJit = true): Detokenizer {
  const f = useJit ? jittedVqgan() : vqgan;
  return async (weights, imageTokens, onStage) => {
    const run = (fn: (w: Weights, x: np.Array) => np.Array, w: Weights, x: np.Array) =>
      fn(tree.ref(w), x).blockUntilReady();
    const emit = async (stage: VqganStage, x: np.Array) => {
      if (onStage) await onStage(stage, x.ref);
    };
    const sub = (prefix: string) => pick(weights, [prefix], prefix);

    let z = await run(f.input, pick(weights, ["embedding.", "post_quant_conv.", "decoder.conv_in."]), imageTokens);
    await emit("conv_in", z);
    z = await run(f.resnet, sub("decoder.mid.block_1."), z);
    z = await run(f.attention, sub("decoder.mid.attn_1."), z);
    z = await run(f.resnet, sub("decoder.mid.block_2."), z);
    await emit("mid", z);
    for (let i = UP_BLOCKS.length - 1; i >= 0; i--) {
      const [, , hasAttention, hasUpsample] = UP_BLOCKS[i];
      for (let j = 0; j < 3; j++) {
        z = await run(f.resnet, sub(`decoder.up.${i}.block.${j}.`), z);
        if (hasAttention) z = await run(f.attention, sub(`decoder.up.${i}.attn.${j}.`), z);
      }
      if (hasUpsample) z = await run(f.upsample, sub(`decoder.up.${i}.upsample.`), z);
      await emit(`up${i}` as VqganStage, z);
    }
    return run(f.output, pick(weights, ["decoder.norm_out.", "decoder.conv_out."], "decoder."), z);
  };
}

// ---------------------------------------------------------------------------
// Jitted entry points
//
// The model functions borrow `weights` (taking a ref per use), while jit
// consumes every argument. So callers pass tree.ref(weights), and each wrapper
// disposes its traced copy once the body is done.

export function makeJitted(cfg: Config) {
  return {
    encode: jit((weights: Weights, textTokens: np.Array) => {
      const out = encode(weights, textTokens, cfg);
      tree.dispose(weights);
      return out;
    }),
    decodeStep: jit((weights: Weights, cross: CrossAttention, cache: KVCache, prevTokens: np.Array, index: np.Array) => {
      const out = decodeStep(weights, cross, cache, prevTokens, index, cfg);
      tree.dispose(weights);
      return out;
    }),
    sampleTokens: (settings: SampleSettings) => jit((logits: np.Array, key: np.Array) => sampleTokens(logits, key, settings)),
  };
}

// ---------------------------------------------------------------------------
// Live preview

const applyPreviewHead = jit((head: Weights, mid: np.Array) => {
  // [n, 512, 16, 16] -> [n, 16, 16, 3] patch colors in 0..255
  const rgb = np.einsum("nchw,dc->nhwd", mid, head.weight).add(head.bias);
  return np.clip(rgb, 0, 255);
});

/**
 * Low-res preview of a (partial) token grid [n, 256] int32, with unfilled
 * positions set to START_TOKEN: the VQGAN's first blocks at 16x16 (~5 GFLOP,
 * vs ~250 for the full decoder), then a linear head fit by
 * python/preview_head.py. Returns [n, 16, 16, 3] float32 patch colors.
 * Borrows `weights` and `head`, consumes `tokens`.
 */
export async function previewTokens(weights: Weights, head: Weights, tokens: np.Array): Promise<np.Array> {
  const f = jittedVqgan();
  const sub = (prefix: string) => tree.ref(pick(weights, [prefix], prefix));
  let z = f.input(tree.ref(pick(weights, ["embedding.", "post_quant_conv.", "decoder.conv_in."])), tokens);
  z = f.resnet(sub("decoder.mid.block_1."), z);
  z = f.attention(sub("decoder.mid.attn_1."), z);
  z = f.resnet(sub("decoder.mid.block_2."), z);
  return applyPreviewHead(tree.ref(head), z).blockUntilReady();
}
