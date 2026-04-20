"""
Download Amazon Beauty and Amazon CDs and Vinyl datasets.

Both datasets are from the Amazon Reviews 2023 collection by McAuley Lab (UCSD).
Source: https://amazon-reviews-2023.github.io/

Usage:
  python scripts/download_data.py --dataset beauty
  python scripts/download_data.py --dataset cds
  python scripts/download_data.py --dataset all
"""
import argparse
from pathlib import Path
import requests
from tqdm import tqdm

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

DATASETS = {
    "beauty": {
        "url": (
            "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/"
            "All_Beauty.jsonl.gz"
        ),
        "dest": RAW_DIR / "beauty" / "All_Beauty.jsonl.gz",
    },
    "cds": {
        "url": (
            "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/"
            "CDs_and_Vinyl.jsonl.gz"
        ),
        "dest": RAW_DIR / "cds" / "CDs_and_Vinyl.jsonl.gz",
    },
}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["beauty", "cds", "all"], default="all")
    args = parser.parse_args()

    targets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for name in targets:
        cfg = DATASETS[name]
        download_file(cfg["url"], cfg["dest"])


if __name__ == "__main__":
    main()
