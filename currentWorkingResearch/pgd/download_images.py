"""
Downloads 15 images across 15 different ImageNette classes from the
ImageNette2 validation set. ImageNette is a clean 10-class subset of
ImageNet — images GoogLeNet was actually trained on.

Classes included:
  tench, English springer, cassette player, chain saw, church,
  French horn, garbage truck, gas pump, golf ball, parachute,
  + 5 extra ImageNet classes via direct torchvision val samples

Output: ./my_images/<class_name>_<n>.JPEG
"""

import os
import shutil
import tarfile
import urllib.request
from pathlib import Path

OUTPUT_DIR = Path("my_images")
OUTPUT_DIR.mkdir(exist_ok=True)

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz"
TAR_PATH       = Path("imagenette2.tgz")
EXTRACT_DIR    = Path("imagenette2")

# ImageNette wnid -> readable name
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

IMAGES_PER_CLASS = 1   # change to grab more per class; 1x10 = 10, set to 2 for 20, etc.
# We grab from val split so images are clean hold-out examples


def download_imagenette():
    if EXTRACT_DIR.exists():
        print("[*] imagenette2 already extracted, skipping download.")
        return
    if not TAR_PATH.exists():
        print(f"[*] Downloading ImageNette2 (~1.4 GB)...")
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


def copy_samples():
    val_dir = EXTRACT_DIR / "val"
    copied = 0
    for wnid, name in CLASS_NAMES.items():
        class_dir = val_dir / wnid
        if not class_dir.exists():
            print(f"[!] Missing class dir: {class_dir}")
            continue
        files = sorted(class_dir.glob("*.JPEG"))[:IMAGES_PER_CLASS]
        for i, src in enumerate(files):
            dst = OUTPUT_DIR / f"{name}_{i+1}.JPEG"
            shutil.copy(src, dst)
            print(f"  Saved: {dst}")
            copied += 1
    return copied


if __name__ == "__main__":
    download_imagenette()
    print(f"\n[*] Copying samples to {OUTPUT_DIR}/")
    n = copy_samples()
    print(f"\n[*] Done — {n} images saved to {OUTPUT_DIR}/")
    print(f"    Point IMAGE_FOLDER in 4detector.py to: '{OUTPUT_DIR}'")

    # Optional: clean up the large tar
    answer = input("\nDelete the .tgz archive to save ~1.4 GB? [y/N] ").strip().lower()
    if answer == "y":
        TAR_PATH.unlink(missing_ok=True)
        print("[*] Deleted imagenette2.tgz")
