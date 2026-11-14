/**
 * The "How it works" panel: renders each pipeline stage from a recorded
 * generation trace at a given decoder step, so live runs, scrubbing and
 * replays share one code path.
 */
import { CANDIDATE_COUNT, type FeatureMap, IMAGE_TOKEN_COUNT, type VqganStage } from "../src/pipeline.ts";

export type StepRecord = {
  tokens: Int32Array;
  candidates?: Int32Array;
  probabilities?: Float32Array;
  attention?: Float32Array;
  elapsedMs: number;
};

export type Phase = "encoding" | "decoding" | "vqgan" | "done" | "stopped" | "error";

export type Trace = {
  id: number;
  prompt: string;
  seed: number;
  count: number;
  textTokens: number[];
  subwords: string[];
  encoderState?: Float32Array;
  encodeMs?: number;
  steps: StepRecord[];
  /** Latest 16x16 RGBA preview per image. */
  previews: Uint8ClampedArray[];
  /** Every preview, in step order, for scrubbing. */
  previewHistory: { step: number; images: Uint8ClampedArray[] }[];
  stages: Map<VqganStage, FeatureMap>[];
  images: (Uint8ClampedArray | undefined)[];
  phase: Phase;
  totalMs?: number;
};

const STAGE_LABELS: Record<VqganStage, [string, string]> = {
  conv_in: ["Codebook lookup", "16² · 512 ch"],
  mid: ["Attention block", "16² · 512 ch"],
  up4: ["Upsample", "32² · 512 ch"],
  up3: ["Upsample", "64² · 256 ch"],
  up2: ["Upsample", "128² · 256 ch"],
  up1: ["Upsample", "256² · 128 ch"],
  up0: ["Refine", "256² · 128 ch"],
};

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

type RGB = [number, number, number];

function cssColor(name: string): RGB {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const hex = v.startsWith("#") ? v.slice(1) : "888888";
  return [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16)) as RGB;
}

const mix = (a: RGB, b: RGB, t: number): RGB => [0, 1, 2].map((i) => Math.round(a[i] + (b[i] - a[i]) * t)) as RGB;
const rgb = (c: RGB) => `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
const luminance = (c: RGB) => (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) / 255;

function drawRGBA(canvas: HTMLCanvasElement, width: number, height: number, rgba: Uint8ClampedArray) {
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  canvas.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(rgba), width, height), 0, 0);
}

export class InsightsPanel {
  private trace: Trace | null = null;
  private image = 0;
  /** null follows the latest step. */
  private pinnedStep: number | null = null;
  private replayFrame = 0;
  private drawnEncoderFor = -1;
  private encoderAnimatedFor = -1;
  private readonly tooltip = $<HTMLDivElement>("tooltip");
  private readonly slider = $<HTMLInputElement>("step-slider");

  constructor(private readonly tokenColors: () => Uint8Array | null) {
    for (const id of ["enc-layers", "dec-layers"]) {
      const el = $(id);
      for (let i = 0; i < 12; i++) {
        const span = document.createElement("span");
        span.style.setProperty("--i", String(i));
        el.append(span);
      }
    }
    this.slider.addEventListener("input", () => {
      this.stopReplay();
      const step = Number(this.slider.value);
      const last = (this.trace?.steps.length ?? 0) - 1;
      this.pinnedStep = step >= last && this.trace?.phase !== "done" ? null : Math.min(step, last);
      this.render();
    });
    $("replay").addEventListener("click", () => this.replay());
    this.hoverHeatmap();
    this.hoverTokenGrid();
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
      this.drawnEncoderFor = -1;
      this.render();
    });
  }

  show(trace: Trace | null, image = this.image) {
    if (trace !== this.trace) {
      this.stopReplay();
      this.pinnedStep = null;
      this.drawnEncoderFor = -1;
    }
    this.trace = trace;
    this.image = trace ? Math.min(image, trace.count - 1) : 0;
    this.render();
  }

  private get step(): number {
    const last = (this.trace?.steps.length ?? 0) - 1;
    return this.pinnedStep === null ? last : Math.min(this.pinnedStep, last);
  }

  render() {
    const t = this.trace;
    const stage = (name: string, state: "pending" | "active" | "done") =>
      ((document.querySelector(`.stage[data-stage="${name}"]`) as HTMLElement).dataset.state = state);
    if (!t) {
      for (const s of ["tokenize", "encode", "decode", "vqgan"]) stage(s, "pending");
      return;
    }
    const replaying = this.replayFrame !== 0;
    const decoded = t.phase === "vqgan" || t.phase === "done";
    stage("tokenize", "done");
    stage("encode", t.phase === "encoding" ? "active" : t.encodeMs !== undefined ? "done" : "pending");
    stage("decode", t.phase === "decoding" || replaying ? "active" : t.steps.length ? "done" : "pending");
    stage("vqgan", t.phase === "vqgan" ? "active" : t.phase === "done" && !replaying ? "done" : "pending");

    this.renderTokens(t);
    this.renderEncoder(t);
    this.renderDecoder(t, decoded && !replaying);
    this.renderVqgan(t, replaying);

    const step = this.step;
    this.slider.max = String(Math.max(0, IMAGE_TOKEN_COUNT - 1));
    this.slider.value = String(Math.max(0, step));
    this.slider.disabled = t.steps.length === 0;
    $("step-output").textContent = t.steps.length ? `step ${step + 1}` : "—";
    ($("replay") as HTMLButtonElement).disabled = t.phase !== "done" && t.phase !== "stopped";
  }

  private renderTokens(t: Trace) {
    const box = $("tok-chips");
    const key = `${t.id}:${t.textTokens.length}`;
    if (box.dataset.key === key) return;
    box.dataset.key = key;
    box.replaceChildren(
      ...t.subwords.map((word, i) => {
        const chip = document.createElement("span");
        const special = i === 0 || (i === t.subwords.length - 1 && word === "</s>");
        chip.className = special ? "chip special" : "chip";
        chip.textContent = word.trim() || word;
        const id = document.createElement("small");
        id.textContent = String(t.textTokens[i]);
        chip.append(id);
        return chip;
      }),
    );
    $("tok-meta").textContent = `${t.textTokens.length} tokens`;
  }

  private renderEncoder(t: Trace) {
    const layers = $("enc-layers");
    if (t.encodeMs === undefined) {
      layers.className = "layers";
    } else if (this.encoderAnimatedFor !== t.id) {
      this.encoderAnimatedFor = t.id;
      layers.className = "layers";
      void layers.offsetWidth; // restart the sweep
      layers.className = "layers run";
    }
    $("enc-meta").textContent = t.encodeMs === undefined ? "BART · 12 layers" : `BART · 12 layers · ${t.encodeMs.toFixed(0)} ms`;

    const canvas = $<HTMLCanvasElement>("enc-heat");
    const labels = $("enc-labels");
    if (!t.encoderState) {
      canvas.hidden = true;
      labels.replaceChildren();
      return;
    }
    if (this.drawnEncoderFor === t.id) return;
    this.drawnEncoderFor = t.id;
    canvas.hidden = false;
    const rows = t.textTokens.length;
    const cols = 128;
    const embed = t.encoderState.length / 64;
    // Clip the color scale at the 98th percentile of |x| so outliers don't wash it out.
    const sample: number[] = [];
    for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) sample.push(Math.abs(t.encoderState[r * embed + c]));
    sample.sort((a, b) => a - b);
    const scale = sample[Math.floor(sample.length * 0.98)] || 1;
    const [neg, mid, pos] = [cssColor("--div-neg"), cssColor("--div-mid"), cssColor("--div-pos")];
    const rgba = new Uint8ClampedArray(rows * cols * 4);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = Math.max(-1, Math.min(1, t.encoderState[r * embed + c] / scale));
        const color = mix(mid, v < 0 ? neg : pos, Math.abs(v));
        rgba.set([...color, 255], (r * cols + c) * 4);
      }
    }
    drawRGBA(canvas, cols, rows, rgba);
    const rowPx = rows > 24 ? 6 : 12;
    canvas.style.height = `${rows * rowPx}px`;
    labels.style.setProperty("--row", `${rowPx}px`);
    labels.replaceChildren(
      ...t.subwords.map((w) => {
        const span = document.createElement("span");
        span.textContent = rowPx >= 12 ? w.trim() || w : "";
        return span;
      }),
    );
  }

  private renderDecoder(t: Trace, finished: boolean) {
    const k = this.image;
    const step = this.step;
    const layers = $("dec-layers");
    layers.className = t.phase === "decoding" || this.replayFrame ? "layers loop" : t.steps.length ? "layers done" : "layers";

    // The image so far: the latest preview at or before this step, with any
    // newer patches in their token's typical color.
    const colors = this.tokenColors();
    const canvas = $<HTMLCanvasElement>("token-grid");
    const rgba = new Uint8ClampedArray(IMAGE_TOKEN_COUNT * 4);
    let preview: { step: number; images: Uint8ClampedArray[] } | undefined;
    for (let j = t.previewHistory.length - 1; j >= 0; j--) {
      if (t.previewHistory[j].step <= step) {
        preview = t.previewHistory[j];
        break;
      }
    }
    for (let i = 0; i <= step; i++) {
      const tok = t.steps[i].tokens[k];
      if (preview && i <= preview.step) rgba.set(preview.images[k].subarray(i * 4, i * 4 + 4), i * 4);
      else if (colors) rgba.set([colors[3 * tok], colors[3 * tok + 1], colors[3 * tok + 2], 255], i * 4);
      else rgba.set([128, 128, 128, 255], i * 4);
    }
    drawRGBA(canvas, 16, 16, rgba);
    // Ring the patch whose candidates are shown below.
    const cursor = $("grid-cursor");
    cursor.hidden = step < 0 || (finished && this.pinnedStep === null);
    cursor.style.left = `${(step % 16) * 8}px`;
    cursor.style.top = `${Math.floor(step / 16) * 8}px`;

    const shown = Math.max(0, step + 1);
    $("big-step").innerHTML = `${shown}<small> / 256</small>`;
    const rec = t.steps[step];
    const meta = $("step-meta");
    if (rec) {
      const msPerStep = rec.elapsedMs / (step + 1);
      meta.textContent = `${msPerStep.toFixed(0)} ms per step · ${(rec.elapsedMs / 1000).toFixed(1)} s`;
    } else {
      meta.innerHTML = "&nbsp;";
    }

    // Candidates for this patch.
    const list = $("candidates");
    if (!rec?.candidates || !rec.probabilities) {
      list.replaceChildren(
        Object.assign(document.createElement("li"), {
          textContent: rec ? "Open this panel before generating to record candidates." : "",
          style: "display:block",
        }),
      );
    } else {
      const cands = rec.candidates.subarray(k * CANDIDATE_COUNT, (k + 1) * CANDIDATE_COUNT);
      const probs = rec.probabilities.subarray(k * CANDIDATE_COUNT, (k + 1) * CANDIDATE_COUNT);
      const sampled = rec.tokens[k];
      const top = probs[0] || 1;
      const rows = Array.from(cands, (tok, j) => {
        const li = document.createElement("li");
        if (tok === sampled) li.className = "sampled";
        const sw = document.createElement("span");
        sw.className = "swatch";
        if (colors) sw.style.background = rgb([colors[3 * tok], colors[3 * tok + 1], colors[3 * tok + 2]]);
        const id = document.createElement("span");
        id.textContent = `#${tok}`;
        const track = document.createElement("span");
        track.className = "track";
        const bar = document.createElement("div");
        bar.className = "bar";
        bar.style.width = `${(probs[j] / top) * 100}%`;
        track.append(bar);
        const value = document.createElement("span");
        value.className = "value";
        value.textContent = `${(probs[j] * 100).toFixed(1)}%`;
        li.append(sw, id, track, value);
        li.title = tok === sampled ? `Sampled token #${tok}` : `Token #${tok}`;
        return li;
      });
      if (!cands.includes(sampled)) {
        const li = document.createElement("li");
        li.className = "sampled";
        li.style.display = "block";
        li.textContent = `Sampled #${sampled}, outside the top ${CANDIDATE_COUNT}`;
        rows.push(li);
      }
      list.replaceChildren(...rows);
    }

    // Which text tokens the decoder attends to.
    const chips = $("attn-chips");
    if (!rec?.attention) {
      chips.replaceChildren();
    } else {
      const n = t.textTokens.length;
      const att = rec.attention.subarray(k * 64, k * 64 + n);
      const max = Math.max(...att) || 1;
      const order = Array.from(att.keys()).sort((a, b) => att[b] - att[a]);
      const [lo, hi] = [cssColor("--seq-lo"), cssColor("--seq-hi")];
      chips.replaceChildren(
        ...t.subwords.map((w, i) => {
          const chip = document.createElement("span");
          chip.className = "chip";
          const color = mix(lo, hi, att[i] / max);
          chip.style.background = rgb(color);
          chip.style.color = luminance(color) < 0.5 ? "#ffffff" : "#0b0b0b";
          chip.textContent = w.trim() || w;
          if (order.indexOf(i) < 3) {
            const pct = document.createElement("small");
            pct.textContent = `${(att[i] * 100).toFixed(0)}%`;
            chip.append(pct);
          }
          chip.title = `${(att[i] * 100).toFixed(1)}% of attention`;
          return chip;
        }),
      );
    }
  }

  private renderVqgan(t: Trace, replaying: boolean) {
    const strip = $("vq-strip");
    const key = `${t.id}:${this.image}:${replaying}`;
    const stages = replaying ? new Map<VqganStage, FeatureMap>() : (t.stages[this.image] ?? new Map());
    const image = replaying ? undefined : t.images[this.image];
    const count = stages.size + (image ? 1 : 0);
    if (strip.dataset.key === key && Number(strip.dataset.count) === count) return;
    strip.dataset.key = key;
    strip.dataset.count = String(count);
    const figures: HTMLElement[] = [];
    for (const [stage, [label, dims]] of Object.entries(STAGE_LABELS) as [VqganStage, [string, string]][]) {
      const map = stages.get(stage);
      const fig = document.createElement("figure");
      const canvas = document.createElement("canvas");
      if (map) drawRGBA(canvas, map.width, map.height, map.rgba);
      const cap = document.createElement("figcaption");
      cap.innerHTML = `<b>${label}</b><br>${dims}`;
      fig.append(canvas, cap);
      figures.push(fig);
    }
    const fig = document.createElement("figure");
    fig.className = "final";
    const canvas = document.createElement("canvas");
    if (image) drawRGBA(canvas, 256, 256, image);
    const cap = document.createElement("figcaption");
    cap.innerHTML = "<b>Image</b><br>256² · RGB";
    fig.append(canvas, cap);
    figures.push(fig);
    strip.replaceChildren(...figures);
  }

  /** Animate the recorded steps from the start (about 4 s), then the VQGAN. */
  replay() {
    const t = this.trace;
    if (!t || !t.steps.length) return;
    this.stopReplay();
    this.encoderAnimatedFor = -1;
    const start = performance.now();
    const total = t.steps.length;
    const tick = () => {
      const step = Math.min(total - 1, Math.floor(((performance.now() - start) / 4000) * total));
      this.pinnedStep = step;
      if (step >= total - 1) {
        this.replayFrame = 0;
        this.pinnedStep = null;
        this.render();
        return;
      }
      this.replayFrame = requestAnimationFrame(tick);
      this.render();
    };
    this.replayFrame = requestAnimationFrame(tick);
  }

  private stopReplay() {
    if (this.replayFrame) cancelAnimationFrame(this.replayFrame);
    this.replayFrame = 0;
  }

  private showTooltip(text: string | null, x = 0, y = 0) {
    this.tooltip.hidden = text === null;
    if (text !== null) {
      this.tooltip.textContent = text;
      this.tooltip.style.left = `${x}px`;
      this.tooltip.style.top = `${y}px`;
    }
  }

  private hoverHeatmap() {
    const canvas = $<HTMLCanvasElement>("enc-heat");
    canvas.addEventListener("mousemove", (e) => {
      const t = this.trace;
      if (!t?.encoderState) return;
      const box = canvas.getBoundingClientRect();
      const col = Math.floor(((e.clientX - box.left) / box.width) * canvas.width);
      const row = Math.floor(((e.clientY - box.top) / box.height) * canvas.height);
      if (row < 0 || row >= t.textTokens.length || col < 0 || col >= canvas.width) return this.showTooltip(null);
      const v = t.encoderState[row * (t.encoderState.length / 64) + col];
      const word = (t.subwords[row] ?? "").trim();
      this.showTooltip(`“${word}” · feature ${col + 1} · ${v.toFixed(2)}`, e.clientX, e.clientY);
    });
    canvas.addEventListener("mouseleave", () => this.showTooltip(null));
  }

  private hoverTokenGrid() {
    const canvas = $<HTMLCanvasElement>("token-grid");
    canvas.addEventListener("mousemove", (e) => {
      const t = this.trace;
      const box = canvas.getBoundingClientRect();
      const col = Math.floor(((e.clientX - box.left) / box.width) * 16);
      const row = Math.floor(((e.clientY - box.top) / box.height) * 16);
      const i = row * 16 + col;
      if (!t || i < 0 || i > this.step || col < 0 || col > 15) return this.showTooltip(null);
      this.showTooltip(`Step ${i + 1} · row ${row + 1}, col ${col + 1} · token #${t.steps[i].tokens[this.image]}`, e.clientX, e.clientY);
    });
    canvas.addEventListener("mouseleave", () => this.showTooltip(null));
    canvas.addEventListener("click", (e) => {
      // Jump the scrubber to the clicked patch.
      const t = this.trace;
      if (!t?.steps.length) return;
      const box = canvas.getBoundingClientRect();
      const i = Math.floor(((e.clientY - box.top) / box.height) * 16) * 16 + Math.floor(((e.clientX - box.left) / box.width) * 16);
      if (i >= 0 && i < t.steps.length) {
        this.stopReplay();
        this.pinnedStep = i;
        this.render();
      }
    });
  }
}
