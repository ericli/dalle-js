# dalle-js

DALL·E Mini running entirely in the browser on WebGPU, using [jax-js](https://github.com/ekzhang/jax-js).
It's a port of [min-dalle](https://github.com/kuprel/min-dalle)'s PyTorch models (BART encoder, BART
decoder with KV cache, VQGAN), checked tensor by tensor against the original.

The page has a prompt box with a grid of 1, 4 or 9 images, which develop live while they generate. A
"How it works" panel animates each stage from real data: the tokenizer, the encoder output, every
decoder step (sampling candidates, cross-attention over the prompt, the image so far), and the VQGAN's
feature maps at each resolution. A scrubber and a Replay button re-run any finished generation.

## Status

Measured on an RTX 2080 Super in Chromium (Linux, Vulkan, no `shader-f16`, so the compute is float32):

| | 1 image | 4 images |
|---|---|---|
| Text encoder | ~60 ms | ~60 ms |
| Image decoder (256 steps) | ~6.3 s (25 ms/step) | ~8.2 s (32 ms/step) |
| VQGAN | ~0.2 s | ~0.9 s |
| **Total** | **~6.7 s** | **~9.2 s** |

The first generation after loading the page also spends about 1.5 s compiling shaders. Downloading the
weights the first time is about 925 MB. Later visits load them from the browser's Origin Private File
System, which takes about 3 s.

Verification against min-dalle (fp32 PyTorch on CUDA, prompt "an armchair in the shape of an avocado",
seed 7). The decoder is teacher-forced on the reference tokens, because PyTorch's RNG can't be
reproduced:

| Check | wasm / f32 (Node) | WebGPU / f16 weights, f32 compute |
|---|---|---|
| Tokenizer, 11 cases | 10/11 (emoji, see below) | 10/11 |
| Encoder output, max abs diff | 4.3e-5 | 2.1e-5 |
| Decoder logits, worst of 256 steps | ≤ 1.1e-4 | 7.6e-5 |
| Reference token in our top-256 | 256/256 | 256/256 |
| VQGAN image (0–255), max abs diff | 0.008 | 1.29 (fp16 weight rounding; PyTorch gets 1.30) |

## Layout

```
python/                  offline steps (PyTorch + min-dalle)
  download_mini.py       fetch Mini weights into ~/.cache/min-dalle (checks HTTP status and size)
  convert.py             .pt -> fp16 and fp32 safetensors (fills the LayerNorm scales the checkpoints omit)
  dump_refs.py           reference tensors and fp16 drift stats from min-dalle
  dump_tokenizer_cases.py
  preview_head.py        learns the live-preview head and the per-token colors
src/                     the port (shared by the page and Node)
  tokenizer.ts           min-dalle's BPE tokenizer
  model.ts               encoder, KV-cached decoder step, sampler, VQGAN, preview
  pipeline.ts            DalleModel: load + generate, with events for the UI
  checks.ts              comparisons against the reference tensors
  weights.ts, viz.ts
web/                     Vite app: index.html (the page), verify.html (dev-only checks on WebGPU)
scripts/verify-node.ts   the checks on wasm/f32
weights/, refs/          generated, not committed
```

## Setup

The offline steps need Python with PyTorch (a CUDA GPU for the reference run), `min-dalle==0.4.11`
and `safetensors`.

```bash
pip install min-dalle==0.4.11 safetensors
python -I python/download_mini.py
python -I python/convert.py mini weights/mini
python -I python/dump_refs.py weights/mini refs/mini
python -I python/dump_tokenizer_cases.py weights/mini refs/tokenizer_cases.json
python -I python/preview_head.py weights/mini     # about 3 minutes on the GPU
npm install
```

## Develop

```bash
npm run dev
```

Then open http://localhost:5173 for the app, or http://localhost:5173/verify.html for the reference
checks on WebGPU (`?precision=f16&compute=auto&steps=256&autorun`). The dev server serves `weights/`
and `refs/` directly; they are never copied into the build.

```bash
npm run verify:node -- --stages tokenizer,encoder,decoder --steps 16
```

This runs the same checks in Node on wasm. The VQGAN stage takes about 20 minutes there, because wasm
convolutions run on a single thread.

## Deploy

1. Upload these files from `weights/mini/` to a host that allows CORS and large files (about 925 MB
   total; the largest file is 449 MB): `config.json`, `vocab.json`, `merges.txt`,
   `encoder.f16.safetensors`, `decoder.f16.safetensors`, `detoker.f16.safetensors`,
   `preview_head.safetensors` and `token_colors.bin`. A Hugging Face model repo works, as does any
   bucket with CORS enabled. GitHub Pages, Netlify and Cloudflare Pages all have per-file size limits
   that are too small.
2. Build with the weights URL, which must end in `/`:
   ```bash
   VITE_WEIGHTS_URL=https://huggingface.co/<user>/<repo>/resolve/main/ npm run build
   ```
3. Host `dist/` (about 110 KB gzipped) anywhere, for example under a path on your site. `npm run preview`
   serves the production build locally, including the OPFS caching.

The page downloads nothing until the first Generate click, and a note under the prompt warns about the
download size first. After that it remembers that the weights are cached and loads them on startup.

## Notes from the port

- **min-dalle quirks reproduced:** the decoder starts from token 16383, not 16384. Prompts are cut to 64
  tokens, which drops `</s>` from long prompts. GELU is the exact erf version, while jax-js defaults to
  the tanh approximation. The checkpoints leave out several LayerNorm scales (Flax `use_scale=False`),
  and `convert.py` writes them as ones.
- **VQGAN must run in float32.** Its residual stream reaches about 6.8e4 in the last block, past the
  float16 maximum of 65504. Its weights download as fp16 and are widened on load.
- **jax-js is lazy.** Each op allocates its output when it's created, and queued ops keep their inputs
  alive until something reads or syncs. So the decode loop reads the sampled token every step, and the
  VQGAN runs block by block with a sync after each. Without that, wasm runs out of its 4 GiB.
- **Float16 needs `shader-f16`,** which this Chromium/Vulkan build doesn't expose. `DalleModel` computes
  in float16 where the adapter supports it, and otherwise widens the fp16 weights to float32 in JS
  before upload (about 2 GB of VRAM instead of 1).
- **Speedups that keep the numbers the same:** the cross-attention keys and values of the encoder
  output are projected once per prompt (min-dalle recomputes them at all 256 steps). The KV cache is a
  fixed `[rows, 256, embed]` array written with `where(position == i)`, so the decoder step compiles
  once. A step's insights (candidates, probabilities, attention) come back in one packed readback.
- **The live preview** runs the VQGAN's first blocks (`conv_in` and the mid block with global
  attention, about 5 GFLOP against about 250 for the full decoder) on the partial token grid. A ridge
  regression then maps those features to colors at 16×16. It was trained on partial grids of 360 Mini
  generations; held-out R² is 0.76 with 32 tokens filled and 0.95 with all of them.

## Known gaps and next steps

- No `demojize`: emoji are dropped, while min-dalle turns them into words (🐶 → "dog_face").
- The float16 compute path hasn't run here yet. It needs a browser with `shader-f16`, such as Chrome on
  Windows or macOS. `verify.html?compute=float16` checks it.
- Mega (about 5.4 GB fp16): the code is size-agnostic (`convert.py mega`), but loading needs sharded
  downloads and realistically float16 compute.

## License

MIT, see [LICENSE](LICENSE). The model code is ported from
[min-dalle](https://github.com/kuprel/min-dalle) (MIT, © 2022 Brett Kuprel). The weights aren't in this
repository; they come from DALL·E Mini and its VQGAN, under their own licenses.
