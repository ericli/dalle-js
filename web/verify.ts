import { defaultDevice, init, numpy as np } from "@jax-js/jax";

import { type CheckResult, type Stage, runChecks } from "../src/checks.ts";
import { supportsFloat16 } from "../src/weights.ts";

const form = document.getElementById("form") as HTMLFormElement;
const logEl = document.getElementById("log")!;
const log = (line: string) => {
  logEl.textContent += line + "\n";
  console.log(line);
};

declare global {
  interface Window {
    checkResults?: CheckResult[] | { error: string };
  }
}

const devices = await init();
const adapter = await navigator.gpu?.requestAdapter();
log(`devices: ${devices.join(", ")}`);
if (adapter) {
  log(`adapter: ${adapter.info.vendor} ${adapter.info.architecture} ${adapter.info.description}`.trim());
  log(`shader-f16: ${adapter.features.has("shader-f16")}`);
}
if (!devices.includes("webgpu")) log("WebGPU is not available in this browser.");
else defaultDevice("webgpu");

const fetchLoader = (base: string) => async (name: string) => {
  const resp = await fetch(base + name);
  if (!resp.ok) throw new Error(`${base + name}: HTTP ${resp.status}`);
  return new Uint8Array(await resp.arrayBuffer());
};

async function run(data: FormData) {
  window.checkResults = undefined;
  const precision = data.get("precision") as "f16" | "f32";
  const compute = data.get("compute") as string;
  const computeDtype = compute === "auto" ? (supportsFloat16("webgpu") ? np.float16 : np.float32) : (compute as np.DType);
  const stages = new Set(data.getAll("stage") as Stage[]);
  log(`\n--- ${precision}, ${data.get("steps")} steps, ${[...stages].join(",")} ---`);
  try {
    window.checkResults = await runChecks({
      loadWeightsFile: fetchLoader("/weights/mini/"),
      loadRefFile: fetchLoader("/refs/"),
      refName: "mini",
      precision,
      computeDtype,
      stages,
      steps: Number(data.get("steps")),
      log,
    });
  } catch (e) {
    log(`ERROR: ${(e as Error).stack ?? e}`);
    window.checkResults = { error: String(e) };
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  run(new FormData(form));
});

// ?precision=f16&steps=8&stages=encoder,decoder&autorun
const params = new URLSearchParams(location.search);
for (const name of ["precision", "compute", "steps"]) {
  const v = params.get(name);
  if (v) (form.elements.namedItem(name) as HTMLInputElement).value = v;
}
if (params.has("stages")) {
  const wanted = new Set(params.get("stages")!.split(","));
  for (const box of form.querySelectorAll<HTMLInputElement>("input[name=stage]")) box.checked = wanted.has(box.value);
}
if (params.has("autorun")) run(new FormData(form));
