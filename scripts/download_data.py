"""
Download Amazon Beauty (2018) and Yelp 2018 datasets.

Amazon Beauty:
  Source: https://nijianmo.github.io/amazon/index.html
  File: All_Beauty_5.json.gz (5-core, already filtered)

Yelp 2018:
  Source: https://www.yelp.com/dataset
  Must be downloaded manually (requires agreeing to ToS).
  Place yelp_academic_dataset_review.json in data/raw/yelp/

Usage:
  python scripts/download_data.py --dataset beauty
  python scripts/download_data.py --dataset yelp   # prints manual download instructions
"""
import argparse
import hashlib
import sys
from pathlib import Path
import requests
from tqdm import tqdm

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

AMAZON_BEAUTY_URL = (
    "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/"
    "All_Beauty.jsonl.gz"
)

AMAZON_BEAUTY_DEST = RAW_DIR / "beauty" / "All_Beauty.jsonl.gz"


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Already exists: {dest}")
        return
    print(f"Downloading {url}")
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
        for chunk in r.iter_content(chunk_size):
            f.write(chunk)
            bar.update(len(chunk))
    print(f"Saved to {dest}")


def download_beauty():
    download_file(AMAZON_BEAUTY_URL, AMAZON_BEAUTY_DEST)


def yelp_instructions():
    dest = RAW_DIR / "yelp"
    print(
        "\nYelp 2018 requires manual download:\n"
        "  1. Go to https://www.yelp.com/dataset/download\n"
        "  2. Accept the terms and download the JSON dataset\n"
        f"  3. Extract yelp_academic_dataset_review.json into:\n     {dest}/\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["beauty", "yelp", "all"], default="all")
    args = parser.parse_args()

    if args.dataset in ("beauty", "all"):
        download_beauty()
    if args.dataset in ("yelp", "all"):
        yelp_instructions()


if __name__ == "__main__":
    main()
