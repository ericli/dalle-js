import { defaultDevice, init } from "@jax-js/jax";
import { cachedFetch } from "@jax-js/loaders";

import { DalleModel, type FileLoader, IMAGE_TOKEN_COUNT } from "../src/pipeline.ts";
import { InsightsPanel, type Trace } from "./insights.ts";

/** Where the converted weights live; set VITE_WEIGHTS_URL for production. */
const WEIGHTS_URL: string = import.meta.env.VITE_WEIGHTS_URL ?? "/weights/mini/";
const MAX_HISTORY = 12;
const CACHED_FLAG = "dalle-js:weights-cached";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const form = $<HTMLFormElement>("prompt-form");
const promptInput = $<HTMLInputElement>("prompt");
const generateButton = $<HTMLButtonElement>("generate");
const statusEl = $("status");
const statusText = $("status-text");
const track = statusEl.querySelector<HTMLElement>(".track")!;
const fill = statusEl.querySelector<HTMLElement>(".fill")!;
const results = $("results");
const toggle = $<HTMLButtonElement>("toggle-insights");
const insightsEl = $("insights");

let model: DalleModel | null = null;
let loading: Promise<DalleModel> | null = null;
let running: AbortController | null = null;
let current: Trace | null = null;
let selectedImage = 0;
let nextId = 1;
const history: Trace[] = [];
const panel = new InsightsPanel(() => model?.tokenColors ?? null);

function setStatus(text: string, progress?: number, error = false) {
  statusText.textContent = text;
  statusEl.classList.toggle("error", error);
  track.hidden = progress === undefined;
  if (progress !== undefined) fill.style.width = `${Math.round(progress * 100)}%`;
}

// ---- Model loading ---------------------------------------------------------

async function fetchWithProgress(url: string, onProgress?: (loaded: number, total?: number) => void) {
  const resp = await fetch(url);
  if (!resp.ok || !resp.body) throw new Error(`${url}: HTTP ${resp.status}`);
  const total = Number(resp.headers.get("Content-Length")) || undefined;
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  const reader = resp.body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress?.(loaded, total);
  }
  const out = new Uint8Array(new ArrayBuffer(loaded));
  let offset = 0;
  for (const c of chunks) {
    out.set(c, offset);
    offset += c.length;
  }
  return out;
}

// In production, keep the weights in the Origin Private File System so later
// visits skip the download. In dev, read them fresh from the local server.
const loadFile: FileLoader = (name, onProgress) =>
  import.meta.env.PROD
    ? cachedFetch(WEIGHTS_URL + name, undefined, (p) => onProgress?.(p.loadedBytes, p.totalBytes))
    : fetchWithProgress(WEIGHTS_URL + name, onProgress);

const FILE_LABELS: Record<string, string> = {
  encoder: "text encoder",
  decoder: "image decoder",
  detoker: "VQGAN",
};

function loadModel(): Promise<DalleModel> {
  loading ??= (async () => {
    const t0 = performance.now();
    const m = await DalleModel.load(loadFile, {
      precision: "f16",
      onProgress: (file, loaded, total) => {
        const label = FILE_LABELS[file.split(".")[0]] ?? file;
        const mb = (x: number) => (x / 2 ** 20).toFixed(0);
        setStatus(
          total ? `Downloading ${label} · ${mb(loaded)} / ${mb(total)} MB` : `Downloading ${label} · ${mb(loaded)} MB`,
          total ? loaded / total : undefined,
        );
      },
    });
    try {
      localStorage.setItem(CACHED_FLAG, "1");
    } catch {
      // Storage may be unavailable (private mode); only costs an auto-load.
    }
    $("device-info").textContent = `${m.computeDtype} · WebGPU`;
    setStatus(`Model ready · ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    return m;
  })();
  loading.catch(() => (loading = null));
  return loading;
}

// ---- Results ---------------------------------------------------------------

type Card = { root: HTMLElement; preview: HTMLCanvasElement; final: HTMLCanvasElement; badge: HTMLElement };
let cards: Card[] = [];

function buildCards(trace: Trace) {
  const cols = trace.count === 1 ? 1 : trace.count === 4 ? 2 : 3;
  results.style.setProperty("--cols", String(cols));
  results.dataset.count = String(trace.count);
  cards = Array.from({ length: trace.count }, (_, k) => {
    const root = document.createElement("figure");
    root.className = "card";
    root.tabIndex = 0;
    root.setAttribute("aria-label", `Image ${k + 1} of ${trace.count}: ${trace.prompt}`);
    const preview = Object.assign(document.createElement("canvas"), { width: 16, height: 16, className: "preview" });
    const final = Object.assign(document.createElement("canvas"), { width: 256, height: 256, className: "final" });
    const badge = Object.assign(document.createElement("span"), { className: "badge" });
    const save = Object.assign(document.createElement("button"), { className: "save", type: "button", textContent: "Save" });
    save.addEventListener("click", (e) => {
      e.stopPropagation();
      final.toBlob((blob) => {
        if (!blob) return;
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `${trace.prompt.slice(0, 60).replace(/[^\w -]+/g, "").trim() || "dalle"}-${trace.seed}-${k + 1}.png`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 1000);
      });
    });
    const select = () => {
      selectedImage = k;
      renderSelection();
      panel.show(current, selectedImage);
    };
    root.addEventListener("click", select);
    root.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        select();
      }
    });
    root.append(preview, final, badge, save);
    return { root, preview, final, badge };
  });
  results.replaceChildren(...cards.map((c) => c.root));
  renderCaption(trace);
  renderSelection();
}

function renderCaption(trace: Trace) {
  const caption = $("current-caption");
  caption.hidden = false;
  const meta = document.createElement("span");
  meta.className = "label";
  meta.textContent = `seed ${trace.seed}${trace.totalMs ? ` · ${(trace.totalMs / 1000).toFixed(1)} s` : ""}`;
  caption.replaceChildren(trace.prompt, meta);
}

function renderSelection() {
  cards.forEach((c, k) => c.root.classList.toggle("selected", insightsOpen() && k === selectedImage && cards.length > 1));
}

function renderResults(trace: Trace) {
  cards.forEach((card, k) => {
    const preview = trace.previews[k];
    if (preview) card.preview.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(preview), 16, 16), 0, 0);
    const image = trace.images[k];
    if (image && !card.root.classList.contains("done")) {
      card.final.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(image), 256, 256), 0, 0);
      card.root.classList.add("done");
    }
    const steps = trace.steps.length;
    card.badge.hidden = !!image;
    card.badge.textContent =
      trace.phase === "decoding" || trace.phase === "encoding"
        ? `${steps} / ${IMAGE_TOKEN_COUNT}`
        : trace.phase === "vqgan"
          ? "PIXELS"
          : trace.phase === "stopped"
            ? "STOPPED"
            : "";
  });
}

let frame = 0;
function scheduleRender() {
  if (frame) return;
  frame = requestAnimationFrame(() => {
    frame = 0;
    if (!current) return;
    renderResults(current);
    if (insightsOpen()) panel.show(current, selectedImage);
  });
}

// ---- History ---------------------------------------------------------------

function renderHistory() {
  const earlier = history.filter((t) => t !== current);
  $("history").hidden = earlier.length === 0;
  $("history-count").textContent = `· ${earlier.length}`;
  $("history-list").replaceChildren(
    ...earlier.map((t) => {
      const li = document.createElement("li");
      const button = document.createElement("button");
      button.type = "button";
      const thumb = document.createElement("span");
      thumb.className = "thumb";
      const image = t.images.find(Boolean);
      if (image) {
        const c = Object.assign(document.createElement("canvas"), { width: 256, height: 256 });
        c.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(image), 256, 256), 0, 0);
        thumb.append(c);
      }
      const title = document.createElement("span");
      title.className = "title";
      title.textContent = t.prompt;
      const meta = document.createElement("span");
      meta.className = "meta";
      const n = t.images.filter(Boolean).length;
      meta.textContent = `${n} image${n === 1 ? "" : "s"}${t.totalMs ? ` · ${(t.totalMs / 1000).toFixed(1)} s` : ""}`;
      button.append(thumb, title, meta);
      button.addEventListener("click", () => {
        if (running) return;
        current = t;
        selectedImage = 0;
        promptInput.value = t.prompt;
        buildCards(t);
        renderResults(t);
        renderHistory();
        panel.show(t, 0);
        window.scrollTo({ top: 0, behavior: "smooth" });
      });
      li.append(button);
      return li;
    }),
  );
}

// ---- Generation ------------------------------------------------------------

function numberInput(id: string, fallback: number): number {
  const v = Number(($(id) as HTMLInputElement).value);
  return Number.isFinite(v) && ($(id) as HTMLInputElement).value !== "" ? v : fallback;
}

async function generate() {
  const prompt = promptInput.value.trim() || promptInput.placeholder;
  const count = Number((document.querySelector("input[name=count]:checked") as HTMLInputElement).value);
  const seed = Math.floor(numberInput("seed", Math.floor(Math.random() * 2 ** 31)));
  const abort = new AbortController();
  running = abort;
  generateButton.textContent = "Stop";
  generateButton.classList.add("stop");

  const trace: Trace = {
    id: nextId++,
    prompt,
    seed,
    count,
    textTokens: [],
    subwords: [],
    steps: [],
    previews: [],
    previewHistory: [],
    stages: Array.from({ length: count }, () => new Map()),
    images: [],
    phase: "encoding",
  };
  current = trace;
  selectedImage = Math.min(selectedImage, count - 1);
  buildCards(trace);
  panel.show(trace, selectedImage);

  const t0 = performance.now();
  try {
    const m = model ?? (model = await loadModel());
    abort.signal.throwIfAborted();
    setStatus("Encoding prompt");
    const result = await m.generate(prompt, {
      seed,
      imageCount: count,
      temperature: numberInput("temperature", 1),
      topK: Math.round(numberInput("top-k", 256)),
      superconditionFactor: numberInput("supercondition", 16),
      insights: insightsOpen,
      previewEvery: 4,
      signal: abort.signal,
      events: {
        onText: (tokens, subwords) => {
          trace.textTokens = tokens;
          trace.subwords = subwords;
          scheduleRender();
        },
        onEncoded: (state, ms) => {
          trace.encoderState = state;
          trace.encodeMs = ms;
          trace.phase = "decoding";
          scheduleRender();
        },
        onStep: (info) => {
          trace.steps.push({
            tokens: info.tokens,
            candidates: info.candidates,
            probabilities: info.probabilities,
            attention: info.crossAttention,
            elapsedMs: info.elapsedMs,
          });
          const msPerStep = info.elapsedMs / (info.step + 1);
          setStatus(
            `Step ${info.step + 1} / ${IMAGE_TOKEN_COUNT} · ${msPerStep.toFixed(0)} ms per step`,
            (info.step + 1) / IMAGE_TOKEN_COUNT,
          );
          scheduleRender();
        },
        onPreview: (step, previews) => {
          trace.previews = previews;
          trace.previewHistory.push({ step, images: previews });
          scheduleRender();
        },
        onStage: (image, stage, map) => {
          trace.phase = "vqgan";
          trace.stages[image].set(stage, map);
          setStatus(`Tokens to pixels · image ${image + 1} / ${count}`, (image + 0.5) / count);
          scheduleRender();
        },
        onImage: (image, rgba) => {
          trace.images[image] = rgba;
          scheduleRender();
        },
      },
    });
    trace.phase = "done";
    trace.totalMs = performance.now() - t0;
    const { encodeMs, decodeMs, detokenizeMs } = result.timings;
    setStatus(
      `Done · ${(trace.totalMs / 1000).toFixed(1)} s · encoder ${encodeMs.toFixed(0)} ms · ` +
        `decoder ${(decodeMs / 1000).toFixed(1)} s · VQGAN ${(detokenizeMs / 1000).toFixed(1)} s`,
    );
    renderCaption(trace);
  } catch (e) {
    if (abort.signal.aborted) {
      trace.phase = "stopped";
      setStatus("Stopped");
    } else {
      trace.phase = "error";
      console.error(e);
      setStatus(`Error · ${(e as Error).message}`, undefined, true);
    }
  } finally {
    running = null;
    generateButton.textContent = "Generate";
    generateButton.classList.remove("stop");
    if (trace.images.some(Boolean)) {
      history.unshift(trace);
      history.length = Math.min(history.length, MAX_HISTORY);
    }
    renderResults(trace);
    renderHistory();
    panel.show(trace, selectedImage);
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  if (running) running.abort();
  else generate();
});

// ---- Insights toggle -------------------------------------------------------

function insightsOpen(): boolean {
  return !insightsEl.hidden;
}

function setInsights(open: boolean) {
  insightsEl.hidden = !open;
  toggle.setAttribute("aria-pressed", String(open));
  document.body.classList.toggle("insights-open", open);
  renderSelection();
  if (open) panel.show(current, selectedImage);
  try {
    localStorage.setItem("dalle-js:insights", open ? "1" : "0");
  } catch {
    // ignore
  }
}

toggle.addEventListener("click", () => {
  setInsights(!insightsOpen());
  // Below the two-column breakpoint the panel sits under the images.
  if (insightsOpen() && !matchMedia("(min-width: 1100px)").matches) {
    insightsEl.scrollIntoView({ behavior: "smooth", block: "start" });
  }
});

// ---- Startup ---------------------------------------------------------------

try {
  if (localStorage.getItem("dalle-js:insights") === "1" || new URLSearchParams(location.search).has("insights")) {
    setInsights(true);
  }
} catch {
  // ignore
}

const devices = await init();
if (!devices.includes("webgpu")) {
  generateButton.disabled = true;
  setStatus(
    "This demo needs WebGPU, which this browser doesn't offer. Try a recent Chrome or Edge, or Safari 26.",
    undefined,
    true,
  );
} else {
  defaultDevice("webgpu");
  let cached = false;
  try {
    cached = localStorage.getItem(CACHED_FLAG) === "1";
  } catch {
    // ignore
  }
  // Load right away when it won't trigger a big download.
  if (cached || import.meta.env.DEV) {
    $("hint").hidden = true;
    loadModel().then((m) => (model = m), (e) => setStatus(`Couldn't load the model: ${e.message}`, undefined, true));
  }
  const prompt = new URLSearchParams(location.search).get("prompt");
  if (prompt) promptInput.value = prompt;
}
