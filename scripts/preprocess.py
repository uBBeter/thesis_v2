import argparse
import pickle
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_amazon, load_steam
from src.data.preprocessor import build_dataset
from src.data.graph import build_graph
import torch

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

RAW_FILES = {
    "toys": RAW_DIR / "toys" / "Toys_and_Games.jsonl.gz",
    "cds": RAW_DIR / "cds" / "CDs_and_Vinyl.jsonl.gz",
    "steam": RAW_DIR / "steam" / "steam_reviews.json.gz",
}

LOADERS = {
    "toys": load_amazon,
    "cds": load_amazon,
    "steam": load_steam,
}

DISPLAY_NAMES = {
    "toys": "Amazon Toys and Games",
    "cds": "Amazon CDs and Vinyl",
    "steam": "Steam Game Store",
}


def process(name: str, df_loader, out_name: str):
    print(f"\n=== Processing {name} ===")
    df = df_loader()
    print(f"Raw interactions: {len(df):,}")

    dataset = build_dataset(df, k=10, n_neg=99, seed=42)
    print(f"After 10-core: {dataset.n_users:,} users, {dataset.n_items:,} items, "
          f"{len(dataset.train):,} train interactions")

    out_path = OUT_DIR / out_name
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

    graph = build_graph(dataset)
    torch.save(graph, out_path / "graph.pt")

    print(f"Saved to {out_path}/")
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["toys", "cds", "steam", "all"], default="all")
    args = parser.parse_args()

    targets = list(RAW_FILES.keys()) if args.dataset == "all" else [args.dataset]
    for key in targets:
        raw = RAW_FILES[key]
        loader = LOADERS[key]
        if not raw.exists():
            print(f"Not found: {raw}. Run: python scripts/download_data.py --dataset {key}")
        else:
            process(DISPLAY_NAMES[key], lambda p=raw, l=loader: l(p), key)


if __name__ == "__main__":
    main()
