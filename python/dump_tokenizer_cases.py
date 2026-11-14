"""Write min-dalle tokenizer outputs for a set of prompts, to test the TS port.

Usage: python -I python/dump_tokenizer_cases.py WEIGHTS_DIR OUT_JSON
"""
import json
import sys

from min_dalle.text_tokenizer import TextTokenizer

PROMPTS = [
    "an armchair in the shape of an avocado",
    "Nuclear explosion broccoli",
    "  double  spaces, punctuation! and CAPS?",
    "café naïve façade — unicode dashes",
    "a photo of a 🐶 on the 🌙",
    "trailing space ",
    "",
    "tabs\tand\nnewlines",
    "supercalifragilisticexpialidocious antidisestablishmentarianism",
    "1984 2001: a space odyssey, 3.14159",
    "word " * 80,
]

weights_dir, out = sys.argv[1], sys.argv[2]
vocab = json.load(open(f"{weights_dir}/vocab.json", encoding="utf8"))
merges = open(f"{weights_dir}/merges.txt", encoding="utf8").read().split("\n")[1:-1]
tokenizer = TextTokenizer(vocab, merges)
cases = [{"prompt": p, "tokens": tokenizer.tokenize(p)} for p in PROMPTS]
json.dump(cases, open(out, "w"), indent=1, ensure_ascii=False)
for c in cases:
    print(len(c["tokens"]), repr(c["prompt"][:50]), c["tokens"][:12])
