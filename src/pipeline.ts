/**
 * End-to-end text-to-image generation: tokenize, encode, 256 decoder steps
 * with sampling, then the VQGAN. Shared by the web page and Node scripts.
 */
import { type Device, defaultDevice, jit, numpy as np, random, tree } from "@jax-js/jax";

import {
  type Config,
  type SampleSettings,
  type VqganStage,
  type Weights,
  CANDIDATE_COUNT,
  IMAGE_TOKEN_COUNT,
  START_TOKEN,
  initDecoder,
  makeDetokenizer,
  makeJitted,
  previewTokens,
  sampleTokens,
} from "./model.ts";
import { TextTokenizer, textTokenRows } from "./tokenizer.ts";
import { type FeatureMap, featureMap, toRGBA } from "./viz.ts";
import { loadWeights, supportsFloat16 } from "./weights.ts";

export { CANDIDATE_COUNT, IMAGE_TOKEN_COUNT };
export type { FeatureMap, VqganStage };

/** Fetches one model file by name, e.g. "decoder.f16.safetensors". */
export type FileLoader = (
  name: string,
  onProgress?: (loadedBytes: number, totalBytes?: number) => void,
) => Promise<Uint8Array<ArrayBuffer>>;

export type LoadOptions = {
  device?: Device;
  /** Which weight files to fetch: "q8" (int8, ~465 MB), "f16" (~925 MB) or "f32". */
  precision: "f16" | "f32" | "q8";
  /**
   * dtype the transformer computes in. The VQGAN always uses float32.
   * Default: float16 if the device supports it, else float32.
   */
  computeDtype?: np.DType;
  onProgress?: (file: string, loadedBytes: number, totalBytes?: number) => void;
};

/** What happened in one decoder step. Arrays are per image. */
export type StepInfo = {
  step: number;
  elapsedMs: number;
  /** The token sampled for each image, [n]. */
  tokens: Int32Array;
  /** With `insights`: the most likely candidates [n, 8] and their probabilities. */
  candidates?: Int32Array;
  probabilities?: Float32Array;
  /** With `insights`: attention over the 64 text positions, [n, 64]. */
  crossAttention?: Float32Array;
};

export type GenerateEvents = {
  onText?: (tokens: number[], subwords: string[]) => void;
  /** The conditioned encoder output, [64, embed], after `elapsedMs`. */
  onEncoded?: (state: Float32Array, elapsedMs: number) => void;
  onStep?: (info: StepInfo) => void;
  /** Low-res (16x16 RGBA) previews of each image so far, every `previewEvery` steps; patches not generated yet are transparent. */
  onPreview?: (step: number, previews: Uint8ClampedArray<ArrayBuffer>[]) => void;
  /** A false-color view of the VQGAN activations for one image. */
  onStage?: (image: number, stage: VqganStage, map: FeatureMap) => void;
  /** One finished 256x256 RGBA image. */
  onImage?: (image: number, rgba: Uint8ClampedArray<ArrayBuffer>) => void;
};

export type GenerateOptions = Partial<SampleSettings> & {
  seed: number;
  /** Images per prompt (default 1). The decoder runs them as one batch. */
  imageCount?: number;
  /** Read back candidates and attention each step (for visualization). Checked every step. */
  insights?: boolean | (() => boolean);
  /** Steps between `onPreview` events (default 8). */
  previewEvery?: number;
  events?: GenerateEvents;
  signal?: AbortSignal;
};

export type GenerateResult = {
  /** 256x256 RGBA pixels per image. */
  images: Uint8ClampedArray<ArrayBuffer>[];
  textTokens: number[];
  /** [n, 256] row-major. */
  imageTokens: Int32Array;
  timings: { encodeMs: number; decodeMs: number; detokenizeMs: number };
};

type Sampler = (logits: np.Array, key: np.Array) => ReturnType<typeof sampleTokens>;

/** Flatten one step's insights into a float32 vector: tokens, candidates, probabilities, attention. */
const packInsights = jit((tokens: np.Array, candidates: np.Array, probabilities: np.Array, attention: np.Array) =>
  np.concatenate([
    tokens.astype(np.float32),
    candidates.astype(np.float32).reshape([-1]),
    probabilities.reshape([-1]),
    attention.astype(np.float32).reshape([-1]),
  ]),
);

const utf8 = new TextDecoder();

export class DalleModel {
  private readonly jitted;
  private readonly detokenize = makeDetokenizer();
  private readonly samplers = new Map<string, Sampler>();

  private constructor(
    readonly cfg: Config,
    readonly tokenizer: TextTokenizer,
    private readonly encoder: Weights,
    private readonly decoder: Weights,
    private readonly detoker: Weights,
    private readonly previewHead: Weights,
    /** Average color of each image token, 16384 x RGB. */
    readonly tokenColors: Uint8Array,
    readonly computeDtype: np.DType,
  ) {
    this.jitted = makeJitted(cfg);
  }

  static async load(load: FileLoader, opts: LoadOptions): Promise<DalleModel> {
    const device = opts.device ?? defaultDevice();
    const computeDtype = opts.computeDtype ?? (supportsFloat16(device) ? np.float16 : np.float32);
    const text = async (name: string) => utf8.decode(await load(name));
    const weights = async (name: string, dtype: np.DType) => {
      const file = `${name}.${opts.precision}.safetensors`;
      const bytes = await load(file, (loaded, total) => opts.onProgress?.(file, loaded, total));
      return loadWeights(bytes, { device, dtype });
    };
    const cfg: Config = JSON.parse(await text("config.json"));
    const tokenizer = new TextTokenizer(JSON.parse(await text("vocab.json")), await text("merges.txt"));
    const encoder = await weights("encoder", computeDtype);
    const decoder = await weights("decoder", computeDtype);
    // The VQGAN always computes in float32 (its activations overflow float16).
    const detoker = await weights("detoker", np.float32);
    const previewHead = loadWeights(await load("preview_head.safetensors"), { device, dtype: np.float32 });
    const tokenColors = new Uint8Array(await load("token_colors.bin"));
    return new DalleModel(cfg, tokenizer, encoder, decoder, detoker, previewHead, tokenColors, computeDtype);
  }

  private sampler(settings: SampleSettings): Sampler {
    const key = JSON.stringify(settings);
    let f = this.samplers.get(key);
    if (!f) {
      f = jit((logits: np.Array, rngKey: np.Array) => sampleTokens(logits, rngKey, settings)) as Sampler;
      this.samplers.set(key, f);
    }
    return f;
  }

  async generate(prompt: string, opts: GenerateOptions): Promise<GenerateResult> {
    const n = opts.imageCount ?? 1;
    const ev = opts.events ?? {};
    const sample = this.sampler({
      temperature: opts.temperature ?? 1,
      topK: opts.topK ?? 256,
      superconditionFactor: opts.superconditionFactor ?? 16,
    });

    const textTokens = this.tokenizer.tokenize(prompt).slice(0, this.cfg.text_token_count);
    ev.onText?.(textTokens, textTokens.map((t) => this.tokenizer.subword(t)));

    let t0 = performance.now();
    const rows = textTokenRows(textTokens, this.cfg.text_token_count);
    const tokenArray = np.array(rows, { shape: [2, this.cfg.text_token_count], dtype: np.int32 });
    const encoderState = await this.jitted.encode(tree.ref(this.encoder), tokenArray.ref).blockUntilReady();
    const encodeMs = performance.now() - t0;
    if (ev.onEncoded) {
      const state = (await encoderState.ref.astype(np.float32).data()) as Float32Array;
      ev.onEncoded(state.slice(state.length / 2), encodeMs); // the conditioned row
    }

    t0 = performance.now();
    let { cross, cache } = initDecoder(this.decoder, encoderState, tokenArray, n, this.cfg);
    const keys = random.split(random.key(opts.seed), IMAGE_TOKEN_COUNT);
    let prev = np.full([n], START_TOKEN, { dtype: np.int32 });
    const sampled: np.Array[] = [];
    const grid = new Int32Array(n * IMAGE_TOKEN_COUNT).fill(START_TOKEN);
    const previewEvery = opts.previewEvery ?? 8;
    try {
      for (let i = 0; i < IMAGE_TOKEN_COUNT; i++) {
        opts.signal?.throwIfAborted();
        const step = this.jitted.decodeStep(
          tree.ref(this.decoder),
          tree.ref(cross),
          cache,
          prev,
          np.array(i, { dtype: np.int32 }),
        );
        cache = step.cache;
        const { tokens, candidates, probabilities } = sample(step.logits, keys.ref.slice(i));
        prev = tokens;
        sampled.push(prev.ref);
        // Reading the tokens also syncs: jax-js is lazy, and without a sync
        // every step's buffers would stay queued. With insights, pack
        // everything into one readback, since each costs a GPU round trip.
        let info: StepInfo;
        if (typeof opts.insights === "function" ? opts.insights() : opts.insights) {
          const packed = (await packInsights(prev.ref, candidates, probabilities, step.crossAttention).data()) as Float32Array;
          const c = n * CANDIDATE_COUNT;
          info = {
            step: i,
            elapsedMs: 0,
            tokens: Int32Array.from(packed.subarray(0, n)),
            candidates: Int32Array.from(packed.subarray(n, n + c)),
            probabilities: packed.slice(n + c, n + 2 * c),
            crossAttention: packed.slice(n + 2 * c),
          };
        } else {
          tree.dispose([candidates, probabilities, step.crossAttention]);
          info = { step: i, elapsedMs: 0, tokens: (await prev.ref.data()) as Int32Array };
        }
        info.elapsedMs = performance.now() - t0;
        ev.onStep?.(info);
        for (let k = 0; k < n; k++) grid[k * IMAGE_TOKEN_COUNT + i] = info.tokens[k];
        if (ev.onPreview && ((i + 1) % previewEvery === 0 || i === IMAGE_TOKEN_COUNT - 1)) {
          const previews = await previewTokens(this.detoker, this.previewHead, np.array(grid, { shape: [n, IMAGE_TOKEN_COUNT], dtype: np.int32 }));
          const rgba = toRGBA((await previews.data()) as Float32Array);
          const size = 16 * 16 * 4;
          // Patches not generated yet are transparent.
          for (let k = 0; k < n; k++) for (let p = i + 1; p < IMAGE_TOKEN_COUNT; p++) rgba[k * size + p * 4 + 3] = 0;
          ev.onPreview(i, Array.from({ length: n }, (_, k) => rgba.slice(k * size, (k + 1) * size)));
        }
      }
    } catch (e) {
      tree.dispose(sampled);
      throw e;
    } finally {
      tree.dispose([cross, cache, keys, prev]);
    }
    const imageTokens = np.stack(sampled, 1); // [n, 256]
    const decodeMs = performance.now() - t0;

    t0 = performance.now();
    const tokenData = (await imageTokens.ref.data()) as Int32Array;
    const images: Uint8ClampedArray<ArrayBuffer>[] = [];
    try {
      // One image at a time: the float32 VQGAN needs ~0.5 GB of activations each.
      for (let k = 0; k < n; k++) {
        opts.signal?.throwIfAborted();
        const onStage = ev.onStage
          ? async (stage: VqganStage, x: np.Array) => ev.onStage!(k, stage, await featureMap(x))
          : undefined;
        const image = await this.detokenize(this.detoker, imageTokens.ref.slice([k, k + 1]), onStage);
        const rgba = toRGBA((await image.data()) as Float32Array);
        images.push(rgba);
        ev.onImage?.(k, rgba);
      }
    } finally {
      imageTokens.dispose();
    }
    const detokenizeMs = performance.now() - t0;
    return { images, textTokens, imageTokens: tokenData, timings: { encodeMs, decodeMs, detokenizeMs } };
  }

  dispose() {
    tree.dispose([this.encoder, this.decoder, this.detoker, this.previewHead]);
  }
}
