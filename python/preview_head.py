"""Cheap live previews of a partially generated token grid.

Learned from real generations: run Mini on a few hundred prompts and record
the mean color of the 16x16 pixel patch at each token position. Two outputs:

- token_colors.bin (16384 x 3 uint8): per-token average color, for swatches.
  Decoding a grid of one repeated token is out of distribution (saturated
  noise), so this blends observed means with a ridge fit from the token
  embedding. Token colors depend a lot on context, so this is only a palette.
- preview_head.safetensors: a linear map from the VQGAN's 16x16 "mid"
  activations (after conv_in, a ResNet block, global attention and another
  block: ~5 GFLOP, vs ~250 for the full decoder) to patch colors. Context
  aware, so it works as a low-res preview while decoding.

The VQGAN is shared by Mini and Mega, so these serve both.

Usage: python -I python/preview_head.py WEIGHTS_DIR [batches]
Caches generations in refs/preview_data.npz.
"""
import itertools
import os
import random
import sys
import time

import numpy
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_refs import DEVICE, IMAGE_TOKEN_COUNT, SUPERCONDITION, TOP_K, build, text_tokens_for  # noqa: E402

SUBJECTS = [
    "a red apple", "a golden retriever", "a city skyline at night", "a bowl of ramen", "a snowy mountain",
    "a sailboat on the ocean", "a portrait of an old man", "a field of sunflowers", "a cat wearing a hat",
    "a vintage car", "a forest in autumn", "a cup of coffee", "an astronaut", "a castle on a hill",
    "a tropical beach", "a bicycle", "a bouquet of roses", "a robot", "a lighthouse in a storm",
    "a plate of sushi", "a desert at sunset", "a green frog", "a wooden cabin", "a jellyfish",
    "a pizza", "a waterfall", "a blue butterfly", "a chess board", "a hot air balloon", "a dragon",
    "a living room", "a pair of sneakers", "an owl", "a bowl of fruit", "a tiger", "a subway station",
]
STYLES = ["", "oil painting of ", "pencil sketch of ", "photo of ", "pixel art of ", "watercolor of "]


def generate(encoder, decoder, detoker, cfg, weights_dir, prompt, image_count, seed):
    """min-dalle's generate_raw_image_stream for a grid of images, fp16."""
    _, text_tokens = text_tokens_for(weights_dir, prompt)
    with torch.autocast("cuda", dtype=torch.float16):
        encoder_state = encoder.forward(text_tokens)
        index = [0] * image_count + [1] * image_count
        text_tokens, encoder_state = text_tokens[index], encoder_state[index]
        attention_mask = text_tokens.not_equal(1)[:, None, None, :]
        attention_state = torch.zeros(
            (cfg["layer_count"], image_count * 4, IMAGE_TOKEN_COUNT, cfg["embed_count"]), device=DEVICE
        )
        image_tokens = torch.full((image_count, IMAGE_TOKEN_COUNT + 1), 2**14 - 1, dtype=torch.long, device=DEVICE)
    torch.manual_seed(seed)
    settings = torch.tensor([1.0, TOP_K, SUPERCONDITION], dtype=torch.float32, device=DEVICE)
    positions = torch.arange(IMAGE_TOKEN_COUNT, device=DEVICE)
    for i in range(IMAGE_TOKEN_COUNT):
        with torch.autocast("cuda", dtype=torch.float16):
            image_tokens[:, i + 1], attention_state = decoder.sample_tokens(
                settings=settings,
                attention_mask=attention_mask,
                encoder_state=encoder_state,
                attention_state=attention_state,
                prev_tokens=image_tokens[:, [i]],
                token_index=positions[[i]],
            )
    tokens = image_tokens[:, 1:]
    images = []
    for t in tokens:  # one at a time: the VQGAN runs in fp32
        images.append(detoker.forward(False, t[None]).clip(0, 255))
    return tokens.cpu().numpy(), torch.stack(images).cpu().numpy()  # [n, 256], [n, 256, 256, 3]


def ridge(x, y, lam):
    x1 = numpy.hstack([x, numpy.ones((len(x), 1))])
    return numpy.linalg.solve(x1.T @ x1 + lam * numpy.eye(x1.shape[1]), x1.T @ y)


def r2(x, y, w):
    fit = numpy.hstack([x, numpy.ones((len(x), 1))]) @ w
    return 1 - ((y - fit) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()


def vq_features(detoker, tokens):
    """[n, 256] tokens -> conv_in and mid activations, each [n * 256, 512]."""
    z = detoker.embedding(torch.from_numpy(tokens).long().to(DEVICE))
    z = z.view(-1, 16, 16, 256).permute(0, 3, 1, 2).contiguous()
    conv_in = detoker.decoder.conv_in.forward(detoker.post_quant_conv.forward(z))
    mid = detoker.decoder.mid.forward(conv_in)
    flat = lambda f: f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]).float().cpu().numpy()
    return flat(conv_in), flat(mid)


def main():
    weights_dir = sys.argv[1]
    batches = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    cache = "refs/preview_data.npz"
    cfg, encoder, decoder, detoker = build(weights_dir, "f16", torch.float16)
    if os.path.exists(cache):
        data = numpy.load(cache)
        all_tokens, all_patches = data["tokens"], data["patches"]
    else:
        prompts = [style + subject for subject, style in itertools.product(SUBJECTS, STYLES)]
        random.Random(0).shuffle(prompts)
        tokens_list, patches_list = [], []
        t0 = time.time()
        for b in range(batches):
            prompt = prompts[b % len(prompts)]
            tokens, images = generate(encoder, decoder, detoker, cfg, weights_dir, prompt, 9, b + 1)
            tokens_list.append(tokens)
            patches_list.append(images.reshape(-1, 16, 16, 16, 16, 3).mean(axis=(2, 4)).reshape(-1, 256, 3))
            print(f"batch {b + 1}/{batches}: {prompt!r} ({time.time() - t0:.0f} s)", flush=True)
        all_tokens, all_patches = numpy.concatenate(tokens_list), numpy.concatenate(patches_list)
        numpy.savez(cache, tokens=all_tokens, patches=all_patches)
    del encoder, decoder
    torch.cuda.empty_cache()

    # Per-token palette.
    embedding = detoker.embedding.weight.detach().float().cpu().numpy()
    flat_tokens, flat_colors = all_tokens.reshape(-1), all_patches.reshape(-1, 3)
    counts = numpy.bincount(flat_tokens, minlength=2**14)
    sums = numpy.stack([numpy.bincount(flat_tokens, flat_colors[:, c], minlength=2**14) for c in range(3)], 1)
    w = ridge(embedding[flat_tokens], flat_colors, 1.0)
    predicted = numpy.hstack([embedding, numpy.ones((2**14, 1))]) @ w
    weight = (counts / (counts + 4))[:, None]
    colors = weight * sums / numpy.maximum(counts, 1)[:, None] + (1 - weight) * predicted
    colors = numpy.clip(colors, 0, 255).round().astype(numpy.uint8)
    colors.tofile(f"{weights_dir}/token_colors.bin")
    print(f"token_colors.bin: {int((counts > 0).sum())}/{2**14} tokens observed")

    # Preview head: a ridge fit from mid activations to patch colors. While
    # decoding, positions >= p still hold the start token 16383, which skews
    # the VQGAN's global attention, so train on partial grids too (counting
    # only filled cells). Hold out 20% of images to measure fit quality.
    n = len(all_tokens)
    split = int(n * 0.8)
    prefixes = [32, 64, 96, 128, 160, 192, 224, 256]

    def partial_features(tokens, p):
        tokens = tokens.copy()
        tokens[:, p:] = 2**14 - 1
        with torch.no_grad():
            _, mid = vq_features(detoker, tokens)
        filled = numpy.tile(numpy.arange(256) < p, len(tokens))
        return mid[filled], filled

    xtx = numpy.zeros((513, 513))
    xty = numpy.zeros((513, 3))
    for start in range(0, split, 32):
        chunk = slice(start, min(start + 32, split))
        y_chunk = all_patches[chunk].reshape(-1, 3)
        for p in prefixes:
            x, filled = partial_features(all_tokens[chunk], p)
            x1 = numpy.hstack([x, numpy.ones((len(x), 1))])
            xtx += x1.T @ x1
            xty += x1.T @ y_chunk[filled]
    w = numpy.linalg.solve(xtx + 10.0 * numpy.eye(513), xty)
    y_test = all_patches[split:].reshape(-1, 3)
    for p in prefixes:
        x, filled = partial_features(all_tokens[split:], p)
        print(f"preview head, first {p:3d} tokens filled: held-out R^2 {r2(x, y_test[filled], w):.3f}")

    w = w.astype(numpy.float32)
    save_file({"weight": torch.from_numpy(w[:-1].T.copy()), "bias": torch.from_numpy(w[-1].copy())},
              f"{weights_dir}/preview_head.safetensors", metadata={"stage": "mid"})
    print(f"wrote {weights_dir}/preview_head.safetensors")


if __name__ == "__main__":
    main()
