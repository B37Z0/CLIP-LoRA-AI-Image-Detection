"""
Download GenImage (Arrow format) from HuggingFace.

This script is entirely AI-generated.

Usage:
    python download_genimage.py          # download all generators
    python download_genimage.py --test   # test split only (much smaller, ~minutes)
"""

import sys
import time
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "nebula/GenImage-arrow"
DATA_DIR = Path(__file__).parent / "data" / "genimage"

GENERATORS = ["ADM", "BigGAN", "glide", "Midjourney",
              "stable_diffusion_v_1_4", "stable_diffusion_v_1_5", "VQDM", "wukong"]

SPLITS = ["train", "test"]


def download_subset(generator, split):
    """Download one generator/split subset."""
    pattern = f"data/{split}/{generator}/*"
    print(f"\n{'='*60}")
    print(f"Downloading: {split}/{generator}")
    print(f"Pattern: {pattern}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        local_path = snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns=pattern,
            local_dir=str(DATA_DIR),
        )
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.0f}s -> {local_path}")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def main():
    test_only = "--test" in sys.argv

    splits = ["test"] if test_only else SPLITS
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading GenImage to: {DATA_DIR}")
    print(f"Splits: {splits}")
    print(f"Generators: {GENERATORS}")
    print(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}
    overall_t0 = time.time()

    for split in splits:
        for gen in GENERATORS:
            ok = download_subset(gen, split)
            results[f"{split}/{gen}"] = "OK" if ok else "FAILED"

    elapsed = time.time() - overall_t0
    print(f"\n{'='*60}")
    print(f"DOWNLOAD COMPLETE — {elapsed/60:.1f} minutes total")
    print(f"{'='*60}")
    for key, status in results.items():
        print(f"  {key}: {status}")

    failed = [k for k, v in results.items() if v == "FAILED"]
    if failed:
        print(f"\n{len(failed)} FAILED downloads. Re-run the script to retry.")
    else:
        print(f"\nAll downloads successful.")


if __name__ == "__main__":
    main()
