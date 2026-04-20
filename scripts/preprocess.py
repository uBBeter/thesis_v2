"""
Preprocess raw datasets and save to data/processed/.

Usage:
  python scripts/preprocess.py --dataset beauty
  python scripts/preprocess.py --dataset yelp
  python scripts/preprocess.py --dataset all
"""
import argparse
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_amazon, load_yelp
from src.data.preprocessor import build_dataset
from src.data.graph import build_graph
import torch

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")


def process(name: str, df_loader, out_name: str):
    print(f"\n=== Processing {name} ===")
    df = df_loader()
    print(f"Raw interactions: {len(df):,}")

    dataset = build_dataset(df, k=10, n_neg=99, seed=42)
    print(f"After 10-core: {dataset.n_users:,} users, {dataset.n_items:,} items, "
          f"{len(dataset.train):,} train interactions")

    out_path = OUT_DIR / out_name
    out_path.mkdir(parents=True, exist_ok=True)

    # Save dataset object
    with open(out_path / "dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

    # Build and save graph
    graph = build_graph(dataset)
    torch.save(graph, out_path / "graph.pt")

    print(f"Saved to {out_path}/")
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["beauty", "yelp", "all"], default="all")
    args = parser.parse_args()

    if args.dataset in ("beauty", "all"):
        beauty_raw = RAW_DIR / "beauty" / "All_Beauty.jsonl.gz"
        if not beauty_raw.exists():
            print(f"Not found: {beauty_raw}. Run: python scripts/download_data.py --dataset beauty")
        else:
            process("Amazon Beauty", lambda: load_amazon(beauty_raw), "beauty")

    if args.dataset in ("yelp", "all"):
        yelp_raw = RAW_DIR / "yelp" / "yelp_academic_dataset_review.json"
        if not yelp_raw.exists():
            print(f"Not found: {yelp_raw}. See: python scripts/download_data.py --dataset yelp")
        else:
            process("Yelp 2018", lambda: load_yelp(yelp_raw), "yelp")


if __name__ == "__main__":
    main()
