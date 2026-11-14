"""Dump reference tensors from min-dalle for checking the jax-js port.

Mirrors MinDalle.generate_raw_image_stream for a single image (grid_size=1),
but builds the modules directly from our converted safetensors (strict load),
so it also validates the conversion and never touches the network.

The fp32 run is the reference. An fp16 run with min-dalle's autocast setup is
then teacher-forced on the reference tokens to measure how far fp16 drifts,
which sets the tolerances for WebGPU/fp16 later.

Usage: python -I python/dump_refs.py WEIGHTS_DIR OUT_PREFIX [prompt] [seed]
Writes OUT_PREFIX.safetensors, OUT_PREFIX.json and OUT_PREFIX.png.
"""
import json
import sys

import numpy
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from min_dalle.models import DalleBartDecoder, DalleBartEncoder, VQGanDetokenizer
from min_dalle.text_tokenizer import TextTokenizer

torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEVICE = "cuda"
IMAGE_TOKEN_COUNT = 256
TOP_K, TEMPERATURE, SUPERCONDITION = 256, 1.0, 16


def build(weights_dir: str, suffix: str, dtype: torch.dtype):
    cfg = json.load(open(f"{weights_dir}/config.json"))
    encoder = DalleBartEncoder(
        layer_count=cfg["layer_count"],
        embed_count=cfg["embed_count"],
        attention_head_count=cfg["attention_head_count"],
        text_vocab_count=cfg["text_vocab_count"],
        text_token_count=64,
        glu_embed_count=cfg["glu_embed_count"],
        device=DEVICE,
    )
    decoder = DalleBartDecoder(
        image_vocab_count=cfg["image_vocab_count"],
        embed_count=cfg["embed_count"],
        attention_head_count=cfg["attention_head_count"],
        glu_embed_count=cfg["glu_embed_count"],
        layer_count=cfg["layer_count"],
        device=DEVICE,
    )
    detoker = VQGanDetokenizer()
    for module, name in [(encoder, "encoder"), (decoder, "decoder"), (detoker, "detoker")]:
        module.load_state_dict(load_file(f"{weights_dir}/{name}.{suffix}.safetensors"), strict=True)
    encoder = encoder.to(dtype).eval().to(DEVICE)
    decoder = decoder.to(dtype).eval().to(DEVICE)
    detoker = detoker.eval().to(DEVICE)  # min-dalle always runs the VQGAN in fp32
    return cfg, encoder, decoder, detoker


def text_tokens_for(weights_dir: str, prompt: str) -> tuple[list[int], torch.Tensor]:
    vocab = json.load(open(f"{weights_dir}/vocab.json", encoding="utf8"))
    merges = open(f"{weights_dir}/merges.txt", encoding="utf8").read().split("\n")[1:-1]
    tokens = TextTokenizer(vocab, merges).tokenize(prompt)[:64]
    text_tokens = numpy.ones((2, 64), dtype=numpy.int32)
    text_tokens[0, :2] = [tokens[0], tokens[-1]]
    text_tokens[1, : len(tokens)] = tokens
    return tokens, torch.tensor(text_tokens, dtype=torch.long, device=DEVICE)


def capture(module: torch.nn.Module, store: dict, name: str):
    """Record a module's first output. min-dalle calls .forward() directly,
    which bypasses forward hooks, so wrap the bound method instead."""
    original = module.forward

    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        if name not in store:
            t = out[0] if isinstance(out, tuple) else out
            store[name] = t.detach().float().cpu().contiguous()
        return out

    module.forward = wrapped
    return lambda: delattr(module, "forward")


def decode_step(decoder, cfg, attention_mask, encoder_state, attention_state, token, i):
    token_index = torch.tensor([i], device=DEVICE)
    return decoder.forward(
        attention_mask=attention_mask,
        encoder_state=encoder_state,
        attention_state=attention_state,
        prev_tokens=token.view(1, 1),
        token_index=token_index,
    )


def run_fp32(weights_dir, prompt, seed):
    cfg, encoder, decoder, detoker = build(weights_dir, "f32", torch.float32)
    tokens, text_tokens = text_tokens_for(weights_dir, prompt)
    refs: dict[str, torch.Tensor] = {"text_tokens": text_tokens.int().cpu()}

    hooks = [
        capture(encoder.layernorm_embedding, refs, "enc_embed"),
        capture(encoder.layers[0], refs, "enc_layer0"),
        capture(decoder.layernorm_embedding, refs, "dec0_embed"),
        capture(decoder.layers[0], refs, "dec0_layer0"),
    ]
    encoder_state = encoder.forward(text_tokens)
    refs["encoder_state"] = encoder_state.float().cpu()

    attention_mask = text_tokens.not_equal(1)[:, None, None, :]
    attention_state = torch.zeros(
        (cfg["layer_count"], 4, IMAGE_TOKEN_COUNT, cfg["embed_count"]), device=DEVICE
    )
    image_tokens = torch.full((1, IMAGE_TOKEN_COUNT + 1), 2**14 - 1, dtype=torch.long, device=DEVICE)
    settings = torch.tensor([TEMPERATURE, TOP_K, SUPERCONDITION], dtype=torch.float32, device=DEVICE)

    # Record the decoder logits that sample_tokens sees at every step.
    all_logits = []
    original_forward = decoder.forward

    def recording_forward(**kwargs):
        logits, state = original_forward(**kwargs)
        all_logits.append(logits[:, -1, :].float().cpu())
        return logits, state

    decoder.forward = recording_forward
    if seed > 0:
        torch.manual_seed(seed)
    token_indices = torch.arange(IMAGE_TOKEN_COUNT, device=DEVICE)
    for i in range(IMAGE_TOKEN_COUNT):
        image_tokens[:, i + 1], attention_state = decoder.sample_tokens(
            settings=settings,
            attention_mask=attention_mask,
            encoder_state=encoder_state,
            attention_state=attention_state,
            prev_tokens=image_tokens[:, [i]],
            token_index=token_indices[[i]],
        )
    decoder.forward = original_forward
    for h in hooks:
        h()

    refs["logits"] = torch.stack(all_logits)  # [256, 2, image_vocab + 1]
    refs["image_tokens"] = image_tokens[0, 1:].int().cpu()

    vq_hooks = [
        capture(detoker.post_quant_conv, refs, "vq_post_quant"),
        capture(detoker.decoder.conv_in, refs, "vq_conv_in"),
        capture(detoker.decoder.mid, refs, "vq_mid"),
        capture(detoker.decoder.up[4], refs, "vq_up4"),
        capture(detoker.decoder.up[3], refs, "vq_up3"),
        capture(detoker.decoder.up[2], refs, "vq_up2"),
    ]
    image = detoker.forward(False, image_tokens[:, 1:])
    for h in vq_hooks:
        h()
    refs["image"] = image.float().cpu().contiguous()  # [256, 256, 3], 0..255 float
    return cfg, tokens, refs


def fp16_drift(weights_dir, refs, cfg):
    """Teacher-force min-dalle's fp16 path on the fp32 reference tokens."""
    _, encoder, decoder, detoker = build(weights_dir, "f16", torch.float16)
    text_tokens = refs["text_tokens"].long().to(DEVICE)
    with torch.autocast("cuda", dtype=torch.float16):
        encoder_state = encoder.forward(text_tokens)
    enc_diff = (encoder_state.float().cpu() - refs["encoder_state"]).abs().max().item()

    attention_mask = text_tokens.not_equal(1)[:, None, None, :]
    attention_state = torch.zeros(
        (cfg["layer_count"], 4, IMAGE_TOKEN_COUNT, cfg["embed_count"]), device=DEVICE
    )
    prev = torch.cat([torch.tensor([2**14 - 1]), refs["image_tokens"][:-1].long()]).to(DEVICE)
    raw_diffs, combined_diffs, in_topk = [], [], 0
    for i in range(IMAGE_TOKEN_COUNT):
        with torch.autocast("cuda", dtype=torch.float16):
            logits, attention_state = decode_step(
                decoder, cfg, attention_mask, encoder_state, attention_state, prev[i], i
            )
        got = logits[:, -1, :].float().cpu()
        want = refs["logits"][i]
        raw_diffs.append((got - want).abs().max().item())
        combine = lambda x: x[0, : 2**14] * (1 - SUPERCONDITION) + x[1, : 2**14] * SUPERCONDITION
        got_c, want_c = combine(got), combine(want)
        combined_diffs.append((got_c - want_c).abs().max().item())
        threshold = got_c.topk(TOP_K).values[-1]
        in_topk += int(got_c[refs["image_tokens"][i]] >= threshold)

    # A pure-fp16 VQGAN overflows: its residual stream in decoder.up[0] reaches
    # ~6.8e4 > 65504. So measure what we can ship: fp16 weights, fp32 compute.
    image16 = detoker.forward(False, refs["image_tokens"].long().to(DEVICE)[None])
    vq_diff = (image16.float().cpu() - refs["image"]).abs()
    return {
        "encoder_state_max_abs": enc_diff,
        "logits_raw_max_abs": {"step0": raw_diffs[0], "max": max(raw_diffs),
                               "median": float(numpy.median(raw_diffs))},
        "logits_combined_max_abs": {"step0": combined_diffs[0], "max": max(combined_diffs),
                                    "median": float(numpy.median(combined_diffs))},
        "ref_token_in_fp16_topk": f"{in_topk}/{IMAGE_TOKEN_COUNT}",
        "vqgan_f16_weights_fp32_compute_image_abs": {"max": vq_diff.max().item(), "mean": vq_diff.mean().item()},
    }


def main():
    weights_dir, out = sys.argv[1], sys.argv[2]
    prompt = sys.argv[3] if len(sys.argv) > 3 else "an armchair in the shape of an avocado"
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 7

    cfg, tokens, refs = run_fp32(weights_dir, prompt, seed)
    save_file(refs, f"{out}.safetensors")
    Image.fromarray(refs["image"].to(torch.uint8).numpy()).save(f"{out}.png")
    torch.cuda.empty_cache()

    drift = fp16_drift(weights_dir, refs, cfg)
    meta = {
        "prompt": prompt,
        "seed": seed,
        "tokens": tokens,
        "image_tokens_head": refs["image_tokens"][:16].tolist(),
        "shapes": {k: list(v.shape) for k, v in refs.items()},
        "fp16_vs_fp32": drift,
    }
    json.dump(meta, open(f"{out}.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k != "shapes"}, indent=2))
    print("shapes:", meta["shapes"])


if __name__ == "__main__":
    main()
