"""
Downloads 15 images across 15 different ImageNette classes from the
ImageNette2 validation set and saves them as PNGs.

Classes: tench, english_springer, cassette_player, chain_saw, church,
         french_horn, garbage_truck, gas_pump, golf_ball, parachute

Output: ./my_images/<class_name>_<n>.png
"""

import os
import shutil
import tarfile
import urllib.request
from pathlib import Path
from PIL import Image

OUTPUT_DIR = Path("my_images")
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
TAR_PATH       = Path("imagenette2.tgz")
EXTRACT_DIR    = Path("imagenette2")

CLASS_NAMES = {
    "n01440764": "tench",
    "n02102040": "english_springer",
    "n02979186": "cassette_player",
    "n03000684": "chain_saw",
    "n03028079": "church",
    "n03394916": "french_horn",
    "n03417042": "garbage_truck",
    "n03425413": "gas_pump",
    "n03445777": "golf_ball",
    "n03888257": "parachute",
}

IMAGES_PER_CLASS = 1   # 1x10 classes = 10 images; set to 2 for 20, etc.


def download_imagenette():
    if EXTRACT_DIR.exists():
        print("[*] imagenette2 already extracted, skipping download.")
        return
    if not TAR_PATH.exists():
        print("[*] Downloading ImageNette2 (~1.4 GB)...")
        urllib.request.urlretrieve(IMAGENETTE_URL, TAR_PATH, reporthook=_progress)
        print()
    print("[*] Extracting...")
    with tarfile.open(TAR_PATH) as t:
        t.extractall()
    print("[*] Done.")


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
    bar = "#" * int(pct // 2)
    print(f"\r  [{bar:<50}] {pct:5.1f}%", end="", flush=True)


def copy_samples_as_png():
    val_dir = EXTRACT_DIR / "val"
    copied = 0
    for wnid, name in CLASS_NAMES.items():
        class_dir = val_dir / wnid
        if not class_dir.exists():
            print(f"[!] Missing class dir: {class_dir}")
            continue
        files = sorted(class_dir.glob("*.JPEG"))[:IMAGES_PER_CLASS]
        for i, src in enumerate(files):
            dst = OUTPUT_DIR / f"{name}_{i+1}.png"
            Image.open(src).convert("RGB").save(dst, "PNG")
            print(f"  Saved: {dst}")
            copied += 1
    return copied


if __name__ == "__main__":
    download_imagenette()
    print(f"\n[*] Converting and saving PNGs to {OUTPUT_DIR}/")
    n = copy_samples_as_png()
    print(f"\n[*] Done — {n} images saved to {OUTPUT_DIR}/")
    print(f"    Point IMAGE_FOLDER in 4detector.py to: '{OUTPUT_DIR}'")

    answer = input("\nDelete the .tgz archive to save ~1.4 GB? [y/N] ").strip().lower()
    if answer == "y":
        TAR_PATH.unlink(missing_ok=True)
        print("[*] Deleted imagenette2.tgz")
