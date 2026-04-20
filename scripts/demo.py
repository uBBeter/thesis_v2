"""
Load a trained model and show top-K recommendations for sample users.

Weights are loaded from local results/ if present, otherwise downloaded
from HuggingFace Hub automatically.

Usage:
  # Local weights
  python scripts/demo.py --model lightgcn --dataset toys --config configs/lightgcn.yaml

  # Download from HF Hub
  python scripts/demo.py --model lightgcn --dataset toys --config configs/lightgcn.yaml \
      --hf-repo your-username/thesis-gnn-recsys

  # Show recommendations for specific users
  python scripts/demo.py --model lightgcn --dataset toys --config configs/lightgcn.yaml \
      --users 0 1 2 --topk 10
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.als import ALSRecommender
from src.models.lightgcn import LightGCN
from src.models.ngcf import NGCF
from src.models.sgl import SGL
from src.models.simgcl import SimGCL

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

MODEL_REGISTRY = {
    "als": ALSRecommender,
    "lightgcn": LightGCN,
    "ngcf": NGCF,
    "sgl": SGL,
    "simgcl": SimGCL,
}


def download_from_hub(repo: str, model_name: str, dataset: str):
    from huggingface_hub import hf_hub_download
    dest = RESULTS_DIR / dataset / model_name / "best.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo,
        filename=f"results/{dataset}/{model_name}/best.pt",
        repo_type="model",
        local_dir=".",
    )
    print(f"Downloaded checkpoint to {path}")


def build_model(cfg: dict, n_users: int, n_items: int):
    name = cfg["model"]
    cls = MODEL_REGISTRY[name]
    if name == "als":
        return cls(
            factors=cfg.get("factors", 64),
            iterations=cfg.get("iterations", 50),
            regularization=cfg.get("regularization", 0.01),
        )
    kwargs = {k: v for k, v in cfg.items()
              if k not in ("model", "lr", "n_epochs", "batch_size", "patience")}
    return cls(n_users=n_users, n_items=n_items, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    parser.add_argument("--dataset", required=True, choices=["toys", "cds"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--hf-repo", default=None,
                        help="HuggingFace repo to download weights from if not local")
    parser.add_argument("--users", nargs="+", type=int, default=None,
                        help="User indices to recommend for (default: 5 random users)")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Load dataset
    data_dir = PROCESSED_DIR / args.dataset
    if not (data_dir / "dataset.pkl").exists():
        print(f"Preprocessed data not found at {data_dir}. Run preprocess.py first.")
        sys.exit(1)

    with open(data_dir / "dataset.pkl", "rb") as f:
        dataset = pickle.load(f)

    idx2item = {v: k for k, v in dataset.item2idx.items()}
    idx2user = {v: k for k, v in dataset.user2idx.items()}

    # Locate checkpoint
    ckpt_path = RESULTS_DIR / args.dataset / args.model / "best.pt"
    if not ckpt_path.exists():
        if args.hf_repo:
            print(f"Checkpoint not found locally. Downloading from {args.hf_repo}...")
            download_from_hub(args.hf_repo, args.model, args.dataset)
        else:
            print(f"Checkpoint not found at {ckpt_path}.")
            print("Either run training first, or provide --hf-repo to download weights.")
            sys.exit(1)

    # Build and load model
    model = build_model(cfg, dataset.n_users, dataset.n_items)

    if args.model == "als":
        # ALS saves its own state differently — re-run training (fast, <1 min)
        print("ALS does not use a neural checkpoint. Re-fitting on training data...")
        model.fit(dataset)
    else:
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model._graph = torch.load(data_dir / "graph.pt", weights_only=False,
                                  map_location=args.device)
        model.eval()
        print(f"Loaded {args.model} checkpoint (epoch {ckpt['epoch']}, "
              f"val Recall@20={ckpt['metrics'].get('Recall@20', '?'):.4f})")

    # Select users
    rng = np.random.default_rng(0)
    user_ids = args.users if args.users else rng.choice(dataset.n_users, size=5, replace=False).tolist()

    # Build set of already-seen items per user (from train)
    user_train_items = (
        dataset.train.groupby("user_idx")["item_idx"]
        .apply(set).to_dict()
    )

    U = model.get_user_embeddings()
    V = model.get_item_embeddings()
    all_item_ids = np.arange(dataset.n_items)

    print(f"\n{'='*60}")
    print(f"Top-{args.topk} recommendations  |  model={args.model}  dataset={args.dataset}")
    print(f"{'='*60}")

    for uid in user_ids:
        scores = U[uid] @ V.T
        seen = user_train_items.get(uid, set())
        scores[list(seen)] = -np.inf       # exclude training items
        top_indices = np.argsort(scores)[::-1][:args.topk]

        original_user = idx2user.get(uid, str(uid))
        print(f"\nUser {uid} (id={original_user}):")
        for rank, item_idx in enumerate(top_indices, 1):
            original_item = idx2item.get(item_idx, str(item_idx))
            print(f"  {rank:2d}. item_idx={item_idx}  id={original_item}  score={scores[item_idx]:.4f}")

    print(f"\n{'='*60}")
    print("To verify metrics, run:")
    print(f"  python scripts/evaluate.py --all --dataset {args.dataset}")


if __name__ == "__main__":
    main()
