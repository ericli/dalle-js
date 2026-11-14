"""Convert min-dalle .pt checkpoints to fp16 and fp32 safetensors.

The checkpoints omit some LayerNorm scales (the Flax originals use
use_scale=False), and min-dalle loads them with strict=False so those stay
at PyTorch's default of ones. We load into the real min-dalle modules and
export the full state dict, so every LayerNorm has a weight and a bias and
the JS side can load strictly.

Tensors keep PyTorch layout: Linear weights are [out, in], conv kernels OIHW.

Usage: python -I python/convert.py [mini|mega] OUT_DIR
"""
import json
import os
import shutil
import sys

import torch
from safetensors.torch import save_file
from min_dalle.models import DalleBartDecoder, DalleBartEncoder, VQGanDetokenizer

ROOT = os.path.expanduser("~/.cache/min-dalle")

CONFIGS = {
    "mini": dict(
        layer_count=12,
        attention_head_count=16,
        embed_count=1024,
        glu_embed_count=2730,
        text_vocab_count=50264,
        image_vocab_count=16384,
    ),
    "mega": dict(
        layer_count=24,
        attention_head_count=32,
        embed_count=2048,
        glu_embed_count=4096,
        text_vocab_count=50272,
        image_vocab_count=16415,
    ),
}


def load_full(module: torch.nn.Module, path: str) -> dict:
    params = torch.load(path, map_location="cpu")
    result = module.load_state_dict(params, strict=False)
    if result.unexpected_keys:
        sys.exit(f"{path}: unexpected keys {result.unexpected_keys}")
    for key in result.missing_keys:
        default = module.state_dict()[key]
        # Only LayerNorm scales should be missing, and they must still be ones.
        if not (key.endswith(".weight") and default.ndim == 1 and torch.all(default == 1)):
            sys.exit(f"{path}: missing key {key} is not a default LayerNorm scale")
    print(f"{os.path.basename(path)}: {len(params)} stored, "
          f"{len(result.missing_keys)} filled with ones")
    return {k: v.detach().contiguous() for k, v in module.state_dict().items()}


def save_both(state: dict, out_dir: str, name: str) -> None:
    for dtype, suffix in [(torch.float16, "f16"), (torch.float32, "f32")]:
        tensors = {k: v.to(dtype).contiguous() for k, v in state.items()}
        if dtype == torch.float16:
            for k, v in tensors.items():
                if not torch.isfinite(v).all():
                    sys.exit(f"{name}.{k} overflows float16")
        path = os.path.join(out_dir, f"{name}.{suffix}.safetensors")
        save_file(tensors, path)
        print(f"  wrote {path} ({os.path.getsize(path) / 2**20:.1f} MiB)")


def main() -> None:
    variant, out_dir = sys.argv[1], sys.argv[2]
    cfg = CONFIGS[variant]
    src = os.path.join(ROOT, f"dalle_bart_{variant}")
    os.makedirs(out_dir, exist_ok=True)

    # Constructing on "meta" would skip the ones we need, so build on CPU.
    encoder = DalleBartEncoder(
        layer_count=cfg["layer_count"],
        embed_count=cfg["embed_count"],
        attention_head_count=cfg["attention_head_count"],
        text_vocab_count=cfg["text_vocab_count"],
        text_token_count=64,
        glu_embed_count=cfg["glu_embed_count"],
        device="cpu",
    )
    save_both(load_full(encoder, os.path.join(src, "encoder.pt")), out_dir, "encoder")
    del encoder

    decoder = DalleBartDecoder(
        image_vocab_count=cfg["image_vocab_count"],
        embed_count=cfg["embed_count"],
        attention_head_count=cfg["attention_head_count"],
        glu_embed_count=cfg["glu_embed_count"],
        layer_count=cfg["layer_count"],
        device="cpu",
    )
    save_both(load_full(decoder, os.path.join(src, "decoder.pt")), out_dir, "decoder")
    del decoder

    detoker = VQGanDetokenizer()
    save_both(load_full(detoker, os.path.join(ROOT, "vqgan", "detoker.pt")), out_dir, "detoker")

    for name in ["vocab.json", "merges.txt"]:
        shutil.copy(os.path.join(src, name), os.path.join(out_dir, name))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({"variant": variant, "text_token_count": 64, "image_token_count": 256, **cfg},
                  f, indent=2)
    print(f"wrote tokenizer and config.json to {out_dir}")


if __name__ == "__main__":
    main()
