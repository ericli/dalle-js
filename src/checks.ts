/**
 * Checks the jax-js port against min-dalle reference tensors (from
 * python/dump_refs.py). Shared by scripts/verify-node.ts (wasm, f32) and
 * web/verify.html (WebGPU, f16 or f32).
 *
 * The decoder is teacher-forced on the reference image tokens: PyTorch's RNG
 * can't be reproduced, so we compare logits at every step instead of samples.
 */
import { numpy as np, tree } from "@jax-js/jax";
import { safetensors } from "@jax-js/loaders";

import { type Config, START_TOKEN, type VqganStage, initDecoder, makeDetokenizer, makeJitted } from "./model.ts";
import type { FileLoader } from "./pipeline.ts";
import { TextTokenizer, textTokenRows } from "./tokenizer.ts";
import { loadWeights } from "./weights.ts";

export type Stage = "tokenizer" | "encoder" | "decoder" | "vqgan";

export type CheckOptions = {
  /** Loads files from the converted weights directory. */
  loadWeightsFile: FileLoader;
  /** Loads files from the refs directory. */
  loadRefFile: FileLoader;
  /** Reference name, e.g. "mini" for mini.json and mini.safetensors. */
  refName: string;
  /** Which weight files to load. */
  precision: "f16" | "f32" | "q8";
  /** dtype the transformer computes in (the VQGAN always uses float32). */
  computeDtype: np.DType;
  stages: Set<Stage>;
  /** Decoder steps to teacher-force (up to 256). */
  steps: number;
  log: (line: string) => void;
};

export type CheckResult = { check: string; value: string; ok: boolean };

/**
 * Tolerances. float16 compute ones come from min-dalle's own fp16-vs-fp32
 * drift (refs/*.json "fp16_vs_fp32": encoder 0.021, raw logits <= 0.04), with
 * headroom for kernel differences. The transformer's f16 files are lossless
 * (the checkpoints are fp16), but the VQGAN's are rounded from fp32, which
 * moves the image by <= 1.3 in PyTorch.
 */
const COMPUTE_TOLERANCE = {
  float32: { encoder: 2e-3, logits: 2e-2, topKFraction: 1 },
  float16: { encoder: 0.1, logits: 0.2, topKFraction: 0.98 },
};
const VQGAN_TOLERANCE = {
  f32: { vqRelative: 1e-3, image: 0.5 },
  f16: { vqRelative: 1e-2, image: 3 },
  // Checked against refs computed from the same dequantized weights (refs/*_q8).
  q8: { vqRelative: 1e-3, image: 0.5 },
};

const EMOJI = /\p{Extended_Pictographic}/u;

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) throw new Error(`length mismatch ${a.length} vs ${b.length}`);
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

function maxAbs(a: ArrayLike<number>): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i]));
  return m;
}

export async function runChecks(opts: CheckOptions): Promise<CheckResult[]> {
  const { log, stages, steps, precision } = opts;
  if (opts.computeDtype !== np.float16 && opts.computeDtype !== np.float32) throw new Error(`unsupported compute dtype ${opts.computeDtype}`);
  const tol = { ...COMPUTE_TOLERANCE[opts.computeDtype], ...VQGAN_TOLERANCE[precision] };
  log(`weights ${precision}, compute ${opts.computeDtype}`);
  const results: CheckResult[] = [];
  const report = (check: string, value: string, ok: boolean) => {
    results.push({ check, value, ok });
    log(`${ok ? "PASS" : "FAIL"}  ${check}: ${value}`);
  };
  const compare = async (check: string, got: np.Array, want: Float32Array, absTol: number) => {
    const data = (await got.astype(np.float32).data()) as Float32Array;
    const diff = maxAbsDiff(data, want);
    report(check, `max |diff| ${diff.toExponential(2)} (ref max |x| ${maxAbs(want).toFixed(1)}, tol ${absTol.toPrecision(2)})`, diff <= absTol);
  };
  const utf8 = new TextDecoder();
  const json = async (load: FileLoader, name: string) => JSON.parse(utf8.decode(await load(name)));
  const timed = async <T>(label: string, f: () => Promise<T>): Promise<T> => {
    const t0 = performance.now();
    const out = await f();
    log(`${label}: ${(performance.now() - t0).toFixed(0)} ms`);
    return out;
  };
  const weights = (name: string, dtype: np.DType) =>
    timed(`load ${name}`, async () => loadWeights(await opts.loadWeightsFile(`${name}.${precision}.safetensors`), { dtype }));

  const cfg: Config = await json(opts.loadWeightsFile, "config.json");
  const refMeta = await json(opts.loadRefFile, `${opts.refName}.json`);
  const refs = safetensors.parse(await opts.loadRefFile(`${opts.refName}.safetensors`)).tensors;
  const ref = (name: string) => refs[name].data as Float32Array<ArrayBuffer>;
  const refImageTokens = refs.image_tokens.data as Int32Array<ArrayBuffer>;
  const tokenizer = new TextTokenizer(
    await json(opts.loadWeightsFile, "vocab.json"),
    utf8.decode(await opts.loadWeightsFile("merges.txt")),
  );
  const rows = textTokenRows(tokenizer.tokenize(refMeta.prompt));

  if (stages.has("tokenizer")) {
    const cases: { prompt: string; tokens: number[] }[] = await json(opts.loadRefFile, "tokenizer_cases.json");
    let matched = 0;
    for (const c of cases) {
      const got = tokenizer.tokenize(c.prompt);
      if (JSON.stringify(got) === JSON.stringify(c.tokens)) matched++;
      else if (EMOJI.test(c.prompt)) log(`  (known gap: no demojize) ${JSON.stringify(c.prompt)}`);
      else log(`  mismatch ${JSON.stringify(c.prompt)}\n    got  ${got}\n    want ${c.tokens}`);
    }
    const emoji = cases.filter((c) => EMOJI.test(c.prompt)).length;
    report("tokenizer cases", `${matched}/${cases.length} match (${emoji} emoji case(s) expected to differ)`, matched >= cases.length - emoji);
    report("text token rows", refMeta.prompt, maxAbsDiff(rows, refs.text_tokens.data as Int32Array) === 0);
  }

  const jitted = makeJitted(cfg);
  if (stages.has("encoder") || stages.has("decoder")) {
    const encoderWeights = await weights("encoder", opts.computeDtype);
    const textTokens = np.array(rows, { shape: [2, cfg.text_token_count], dtype: np.int32 });
    const encoderState = await timed("encoder", () => jitted.encode(tree.ref(encoderWeights), textTokens.ref).blockUntilReady());
    await compare("encoder state", encoderState.ref, ref("encoder_state"), tol.encoder);
    tree.dispose(encoderWeights);

    if (stages.has("decoder")) {
      const decoderWeights = await weights("decoder", opts.computeDtype);
      let { cross, cache } = initDecoder(decoderWeights, encoderState, textTokens, 1, cfg);
      const refLogits = ref("logits");
      const vocab1 = cfg.image_vocab_count + 1;
      const factor = 16;
      let worst = 0;
      let worstStep = 0;
      let inTopK = 0;
      const t0 = performance.now();
      for (let i = 0; i < steps; i++) {
        const prev = np.array([i === 0 ? START_TOKEN : refImageTokens[i - 1]], { dtype: np.int32 });
        const out = jitted.decodeStep(tree.ref(decoderWeights), tree.ref(cross), cache, prev, np.array(i, { dtype: np.int32 }));
        cache = out.cache;
        out.crossAttention.dispose();
        const got = (await out.logits.astype(np.float32).data()) as Float32Array;
        const want = refLogits.subarray(i * 2 * vocab1, (i + 1) * 2 * vocab1);
        const diff = maxAbsDiff(got, want);
        if (diff > worst) [worst, worstStep] = [diff, i];
        if (i === 0) report("decoder step-0 logits", `max |diff| ${diff.toExponential(2)} (ref max |x| ${maxAbs(want).toFixed(1)}, tol ${tol.logits})`, diff <= tol.logits);
        // Is the reference sample inside our superconditioned top-256?
        const combined = new Float32Array(2 ** 14);
        for (let j = 0; j < 2 ** 14; j++) combined[j] = got[j] * (1 - factor) + got[vocab1 + j] * factor;
        const kth = Float32Array.from(combined).sort().reverse()[255];
        if (combined[refImageTokens[i]] >= kth) inTopK++;
        if ((i + 1) % 32 === 0) log(`  step ${i + 1}: ${((performance.now() - t0) / (i + 1)).toFixed(0)} ms/step`);
      }
      log(`decoder: ${((performance.now() - t0) / steps).toFixed(1)} ms/step over ${steps} steps (includes readback)`);
      report(`decoder logits, ${steps} steps`, `worst max |diff| ${worst.toExponential(2)} at step ${worstStep}`, worst <= tol.logits);
      report("reference token in our top-256", `${inTopK}/${steps}`, inTopK >= Math.floor(tol.topKFraction * steps));
      tree.dispose([cross, cache, decoderWeights]);
    } else {
      tree.dispose([encoderState, textTokens]);
    }
  }

  if (stages.has("vqgan")) {
    const detokerWeights = await weights("detoker", np.float32);
    const taps: Partial<Record<VqganStage, np.Array>> = {};
    const image = await timed("vqgan", () =>
      makeDetokenizer()(detokerWeights, np.array(refImageTokens, { shape: [1, 256], dtype: np.int32 }), (stage, x) => {
        if (`vq_${stage}` in refs) taps[stage] = x;
        else x.dispose();
      }),
    );
    for (const [stage, x] of Object.entries(taps)) {
      const want = ref(`vq_${stage}`);
      await compare(`vq_${stage}`, x, want, tol.vqRelative * Math.max(1, maxAbs(want)));
    }
    await compare("image (0..255)", image, ref("image"), tol.image);
    tree.dispose(detokerWeights);
  }

  const failed = results.filter((r) => !r.ok).length;
  log(failed ? `\n${failed} check(s) failed` : "\nall checks passed");
  return results;
}
