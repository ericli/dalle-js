"""Download DALL·E Mini weights from the min-dalle HF repo.

Streams each file to a .part, checks raise_for_status and Content-Length,
then renames into place (min-dalle's own downloader saves HTTP error pages
as .pt files).
"""
import os
import sys

import requests

REPO = "https://huggingface.co/kuprel/min-dalle/resolve/main/"
ROOT = os.path.expanduser("~/.cache/min-dalle")
FILES = [
    ("vocab_mini.json", "dalle_bart_mini/vocab.json"),
    ("merges_mini.txt", "dalle_bart_mini/merges.txt"),
    ("encoder_mini.pt", "dalle_bart_mini/encoder.pt"),
    ("decoder_mini.pt", "dalle_bart_mini/decoder.pt"),
    ("detoker.pt", "vqgan/detoker.pt"),
]


def download(name: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"exists: {dest} ({os.path.getsize(dest)} bytes)")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with requests.get(REPO + name, stream=True, timeout=60) as r:
        r.raise_for_status()
        expected = int(r.headers.get("Content-Length", -1))
        part = dest + ".part"
        got = 0
        with open(part, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                got += len(chunk)
                if got % (64 << 20) < (1 << 20):
                    print(f"  {name}: {got >> 20} / {expected >> 20} MiB", flush=True)
    if expected >= 0 and got != expected:
        sys.exit(f"size mismatch for {name}: got {got}, expected {expected}")
    os.rename(part, dest)
    print(f"done: {dest} ({got} bytes)", flush=True)


if __name__ == "__main__":
    for name, rel in FILES:
        download(name, os.path.join(ROOT, rel))
