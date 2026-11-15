/**
 * Full-screen image viewer, like Are.na's block view: the image large on the
 * left, details and actions on the right. A native modal <dialog> handles
 * focus and Escape.
 */
import type { Trace } from "./insights.ts";

export type ViewerActions = {
  /** Put this generation's prompt and settings back in the form. */
  onReuse: (trace: Trace) => void;
  /** Show this image in the How it works panel. */
  onInspect: (trace: Trace, image: number) => void;
};

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;

/** A filename for a generated image, from its prompt, seed and index. */
export function fileName(trace: Trace, image: number): string {
  const slug = trace.prompt.slice(0, 60).replace(/[^\w -]+/g, "").trim().replace(/\s+/g, "-") || "dalle";
  return `${slug}-${trace.seed}-${image + 1}.png`;
}

/** Download 256x256 RGBA pixels as a PNG. */
export function savePNG(rgba: Uint8ClampedArray, name: string) {
  const canvas = Object.assign(document.createElement("canvas"), { width: 256, height: 256 });
  canvas.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(rgba), 256, 256), 0, 0);
  canvas.toBlob((blob) => {
    if (!blob) return;
    const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: name });
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
}

export class ImageViewer {
  private readonly dialog = $<HTMLDialogElement>("viewer");
  private readonly canvas = $<HTMLCanvasElement>("viewer-canvas");
  private trace: Trace | null = null;
  private index = 0;

  constructor(private readonly actions: ViewerActions) {
    $("viewer-close").addEventListener("click", () => this.dialog.close());
    $("viewer-prev").addEventListener("click", () => this.step(-1));
    $("viewer-next").addEventListener("click", () => this.step(1));
    $("viewer-save").addEventListener("click", () => {
      const image = this.trace?.images[this.index];
      if (this.trace && image) savePNG(image, fileName(this.trace, this.index));
    });
    $("viewer-reuse").addEventListener("click", () => {
      if (!this.trace) return;
      this.dialog.close();
      this.actions.onReuse(this.trace);
    });
    $("viewer-inspect").addEventListener("click", () => {
      if (!this.trace) return;
      this.dialog.close();
      this.actions.onInspect(this.trace, this.index);
    });
    this.dialog.addEventListener("keydown", (e) => {
      if (e.key === "ArrowLeft") this.step(-1);
      else if (e.key === "ArrowRight") this.step(1);
    });
    // Clicking the empty area around the image closes the viewer.
    $("viewer-stage").addEventListener("click", (e) => {
      if (e.target === e.currentTarget) this.dialog.close();
    });
  }

  open(trace: Trace, index: number) {
    this.trace = trace;
    this.index = index;
    this.render();
    if (!this.dialog.open) this.dialog.showModal();
  }

  /** Move to the previous or next finished image, wrapping around. */
  private step(delta: number) {
    const t = this.trace;
    if (!t) return;
    for (let i = 1; i <= t.count; i++) {
      const k = (this.index + delta * i + t.count * i) % t.count;
      if (t.images[k]) {
        this.index = k;
        this.render();
        return;
      }
    }
  }

  private render() {
    const t = this.trace;
    if (!t) return;
    const image = t.images[this.index];
    if (image) {
      this.canvas.getContext("2d")!.putImageData(new ImageData(new Uint8ClampedArray(image), 256, 256), 0, 0);
    }
    this.canvas.setAttribute("aria-label", `Image ${this.index + 1} of ${t.count}: ${t.prompt}`);
    $("viewer-index").textContent = `Image ${this.index + 1} / ${t.count}`;
    $("viewer-prompt").textContent = t.prompt;
    const s = t.settings;
    const rows: [string, string][] = [
      ["Seed", String(t.seed)],
      ["Time", t.totalMs ? `${(t.totalMs / 1000).toFixed(1)} s for ${t.count}` : "—"],
      ["Size", "256 × 256"],
    ];
    if (s) {
      rows.push(["Temperature", String(s.temperature)], ["Top-k", String(s.topK)], ["Supercondition", String(s.superconditionFactor)]);
    }
    $("viewer-meta").replaceChildren(
      ...rows.flatMap(([k, v]) => [
        Object.assign(document.createElement("dt"), { className: "label", textContent: k }),
        Object.assign(document.createElement("dd"), { textContent: v }),
      ]),
    );
    const many = t.images.filter(Boolean).length > 1;
    $("viewer-prev").hidden = !many;
    $("viewer-next").hidden = !many;
  }
}
