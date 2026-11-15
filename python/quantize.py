"""Quantize the converted weights to int8 for a smaller download (*.q8.safetensors).

Weight-only, dequantized on load: the browser computes exactly as with fp16
weights, so this only trades a little accuracy for half the download.

- Transformer (encoder, decoder): 2-D tensors with >= 1M elements (attention,
  GLU, embeddings, lm_head) become int8 with one fp32 scale per row
  (absmax / 127), stored as `<name>` (I8) and `<name>.scale` (F32).
- VQGAN: kernels with >= 65,536 elements, one scale per output channel.
- Everything else stays fp16.

Also writes refs/<ref>_q8.* : the reference tensors recomputed with the
dequantized weights, teacher-forced on the fp32 reference tokens, so the JS
checks can confirm dequantization matches. And prints how far int8 drifts
from fp32.

Usage: python -I python/quantize.py WEIGHTS_DIR REFS_PREFIX   (e.g. weights/mini refs/mini)
"""
import json
import os
import sys

import numpy
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_refs import DEVICE, IMAGE_TOKEN_COUNT, build, capture, decode_step  # noqa: E402

torch.set_grad_enabled(False)


def quantize_rows(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows = w.reshape(w.shape[0], -1).float()
    scale = rows.abs().amax(dim=1).clamp_min(1e-12) / 127
    q = (rows / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return q.reshape(w.shape), scale


def quantize_file(src: str, dst: str, min_numel: int) -> dict[str, torch.Tensor]:
    """Write the q8 file and return the dequantized fp32 tensors."""
    tensors = load_file(src)
    out, dequant = {}, {}
    for name, w in tensors.items():
        if w.ndim >= 2 and w.numel() >= min_numel:
            q, scale = quantize_rows(w)
            out[name], out[f"{name}.scale"] = q.contiguous(), scale.contiguous()
            dequant[name] = (q.reshape(w.shape[0], -1).float() * scale[:, None]).reshape(w.shape)
        else:
            out[name] = w.half().contiguous()
            dequant[name] = w.half().float()
    save_file(out, dst)
    print(f"wrote {dst} ({os.path.getsize(dst) / 2**20:.1f} MiB)")
    return dequant


def sampling_dist(logits: torch.Tensor) -> torch.Tensor:
    x = logits[0, : 2**14] * -15 + logits[1, : 2**14] * 16
    kth = x.topk(256).values[-1]
    return torch.softmax(torch.where(x >= kth, x, torch.full_like(x, -torch.inf)), -1)


def main():
    weights_dir, ref_prefix = sys.argv[1], sys.argv[2]
    deq = {
        "encoder": quantize_file(f"{weights_dir}/encoder.f32.safetensors", f"{weights_dir}/encoder.q8.safetensors", 1_000_000),
        "decoder": quantize_file(f"{weights_dir}/decoder.f32.safetensors", f"{weights_dir}/decoder.q8.safetensors", 1_000_000),
        "detoker": quantize_file(f"{weights_dir}/detoker.f32.safetensors", f"{weights_dir}/detoker.q8.safetensors", 65_536),
    }

    cfg, encoder, decoder, detoker = build(weights_dir, "f32", torch.float32)
    for module, name in [(encoder, "encoder"), (decoder, "decoder"), (detoker, "detoker")]:
        module.load_state_dict({k: v.to(DEVICE) for k, v in deq[name].items()}, strict=True)

    fp32 = load_file(f"{ref_prefix}.safetensors")
    refs: dict[str, torch.Tensor] = {"text_tokens": fp32["text_tokens"], "image_tokens": fp32["image_tokens"]}
    text = fp32["text_tokens"].long().to(DEVICE)
    encoder_state = encoder.forward(text)
    refs["encoder_state"] = encoder_state.float().cpu()

    mask = text.not_equal(1)[:, None, None, :]
    state = torch.zeros((cfg["layer_count"], 4, IMAGE_TOKEN_COUNT, cfg["embed_count"]), device=DEVICE)
    prev = torch.cat([torch.tensor([2**14 - 1]), fp32["image_tokens"][:-1].long()]).to(DEVICE)
    logits_all, tv, top1, raw = [], [], 0, []
    for i in range(IMAGE_TOKEN_COUNT):
        logits, state = decode_step(decoder, cfg, mask, encoder_state, state, prev[i], i)
        got = logits[:, -1].float().cpu()
        logits_all.append(got)
        want = fp32["logits"][i]
        raw.append((got - want).abs().max().item())
        p, q = sampling_dist(got), sampling_dist(want)
        tv.append(0.5 * (p - q).abs().sum().item())
        top1 += int(p.argmax() == q.argmax())
    refs["logits"] = torch.stack(logits_all)

    hooks = [
        capture(detoker.decoder.conv_in, refs, "vq_conv_in"),
        capture(detoker.decoder.mid, refs, "vq_mid"),
        capture(detoker.decoder.up[4], refs, "vq_up4"),
        capture(detoker.decoder.up[3], refs, "vq_up3"),
        capture(detoker.decoder.up[2], refs, "vq_up2"),
    ]
    image = detoker.forward(False, fp32["image_tokens"].long().to(DEVICE)[None])
    for h in hooks:
        h()
    refs["image"] = image.float().cpu().contiguous()
    save_file({k: v.contiguous() for k, v in refs.items()}, f"{ref_prefix}_q8.safetensors")

    err = (refs["image"] - fp32["image"]).abs()
    psnr = 10 * numpy.log10(255**2 / ((refs["image"] - fp32["image"]) ** 2).mean().item())
    drift = {
        "logits_raw_max_abs": max(raw),
        "sampling_tv_mean": float(numpy.mean(tv)),
        "sampling_tv_max": float(numpy.max(tv)),
        "top1_agreement": f"{top1}/{IMAGE_TOKEN_COUNT}",
        "vqgan_image_abs": {"max": err.max().item(), "mean": err.mean().item(), "psnr_db": psnr},
    }
    meta = json.load(open(f"{ref_prefix}.json"))
    meta.update({"weights": "q8 (dequantized), teacher-forced on the fp32 reference tokens", "q8_vs_fp32": drift})
    meta.pop("fp16_vs_fp32", None)
    json.dump(meta, open(f"{ref_prefix}_q8.json", "w"), indent=2)
    print(json.dumps(drift, indent=2))


if __name__ == "__main__":
    main()
