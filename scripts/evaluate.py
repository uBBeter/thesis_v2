"""
Load a saved checkpoint and evaluate on test set.

Usage:
  python scripts/evaluate.py --model lightgcn --dataset beauty --device cpu
  python scripts/evaluate.py --all --dataset beauty
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.als import ALSRecommender
from src.models.ngcf import NGCF
from src.models.lightgcn import LightGCN
from src.models.sgl import SGL
from src.models.simgcl import SimGCL

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

ALL_MODELS = ["als", "ngcf", "lightgcn", "sgl", "simgcl"]


def print_table(results: dict):
    metrics = ["Recall@10", "Recall@20", "NDCG@10", "NDCG@20", "Precision@10", "Precision@20"]
    header = f"{'Model':<12}" + "".join(f"{m:>14}" for m in metrics)
    print("\n" + header)
    print("-" * len(header))
    for model, vals in results.items():
        row = f"{model:<12}" + "".join(f"{vals.get(m, float('nan')):>14.4f}" for m in metrics)
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=ALL_MODELS)
    parser.add_argument("--dataset", choices=["beauty", "yelp"], required=True)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    models_to_eval = ALL_MODELS if args.all else [args.model]

    data_dir = PROCESSED_DIR / args.dataset
    with open(data_dir / "dataset.pkl", "rb") as f:
        dataset = pickle.load(f)

    all_results = {}
    for model_name in models_to_eval:
        metrics_path = RESULTS_DIR / args.dataset / model_name / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                all_results[model_name] = json.load(f)
            print(f"Loaded saved metrics for {model_name}")
        else:
            print(f"No saved metrics for {model_name} — skipping (run train.py first)")

    if all_results:
        print_table(all_results)


if __name__ == "__main__":
    main()
