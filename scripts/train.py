import argparse
import json
import pickle
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.graph import build_graph
from src.models.als import ALSRecommender
from src.models.ngcf import NGCF
from src.models.lightgcn import LightGCN
from src.models.sgl import SGL
from src.models.simgcl import SimGCL

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

MODEL_REGISTRY = {
    "als": ALSRecommender,
    "ngcf": NGCF,
    "lightgcn": LightGCN,
    "sgl": SGL,
    "simgcl": SimGCL,
}


def build_model(cfg: dict, n_users: int, n_items: int):
    name = cfg["model"]
    cls = MODEL_REGISTRY[name]
    if name == "als":
        return cls(
            factors=cfg.get("factors", 64),
            iterations=cfg.get("iterations", 50),
            regularization=cfg.get("regularization", 0.01),
            use_gpu=cfg.get("use_gpu", False),
        )
    kwargs = {k: v for k, v in cfg.items() if k not in ("model", "lr", "n_epochs",
                                                          "batch_size", "patience")}
    return cls(n_users=n_users, n_items=n_items, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=["toys", "cds"], required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir = PROCESSED_DIR / args.dataset
    with open(data_dir / "dataset.pkl", "rb") as f:
        dataset = pickle.load(f)

    model_name = cfg["model"]
    out_dir = RESULTS_DIR / args.dataset / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(out_dir / "best.pt")

    model = build_model(cfg, dataset.n_users, dataset.n_items)

    if model_name == "als":
        model.fit(dataset)
    else:
        graph = torch.load(data_dir / "graph.pt", weights_only=False)
        model.fit(
            dataset=dataset,
            graph=graph,
            n_epochs=cfg.get("n_epochs", 200),
            lr=cfg.get("lr", 1e-3),
            batch_size=cfg.get("batch_size", 2048),
            device=args.device,
            checkpoint_path=checkpoint_path,
            val_df=dataset.val,
            patience=cfg.get("patience", 10),
        )

    if model_name != "als" and checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, weights_only=False,
                          map_location=args.device)
        model.load_state_dict(ckpt["state_dict"])

    metrics = model.evaluate(dataset.test)
    print(f"\n=== Test results [{model_name} / {args.dataset}] ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
