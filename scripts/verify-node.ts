/**
 * Check the jax-js port against min-dalle reference tensors on wasm (float32).
 *
 *   npx tsx scripts/verify-node.ts [--weights weights/mini] [--refs refs] [--precision f32|q8] [--ref mini]
 *       [--steps 256] [--stages tokenizer,encoder,decoder,vqgan]
 *
 * The VQGAN stage is slow here (single-threaded wasm convolutions).
 */
import { readFile } from "node:fs/promises";
import { parseArgs } from "node:util";

import { defaultDevice, init, numpy as np } from "@jax-js/jax";

import { type Stage, runChecks } from "../src/checks.ts";

const { values: args } = parseArgs({
  options: {
    weights: { type: "string", default: "weights/mini" },
    refs: { type: "string", default: "refs" },
    ref: { type: "string" },
    precision: { type: "string", default: "f32" },
    steps: { type: "string", default: "256" },
    stages: { type: "string", default: "tokenizer,encoder,decoder,vqgan" },
  },
});

await init("wasm");
defaultDevice("wasm");

const fileLoader = (dir: string) => async (name: string) => new Uint8Array(await readFile(`${dir}/${name}`));
const results = await runChecks({
  loadWeightsFile: fileLoader(args.weights!),
  loadRefFile: fileLoader(args.refs!),
  refName: args.ref ?? (args.precision === "q8" ? "mini_q8" : "mini"),
  precision: args.precision as "f32" | "q8",
  computeDtype: np.float32,
  stages: new Set(args.stages!.split(",") as Stage[]),
  steps: Number(args.steps),
  log: (line) => console.log(line),
});
process.exit(results.every((r) => r.ok) ? 0 : 1);
