import argparse
import pickle
import sys
from pathlib import Path

import optuna
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.als import ALSRecommender
from src.models.lightgcn import LightGCN
from src.models.ngcf import NGCF
from src.models.sgl import SGL
from src.models.simgcl import SimGCL
from src.utils.helpers import EarlyStopping, save_checkpoint

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")


def sample_params(trial: optuna.Trial, model_name: str) -> dict:
    params = {}

    if model_name == "als":
        params["factors"] = trial.suggest_categorical("factors", [32, 64, 128])
        params["regularization"] = trial.suggest_float("regularization", 1e-3, 1.0, log=True)
        params["iterations"] = trial.suggest_categorical("iterations", [30, 50, 100])
        return params

    params["dim"] = trial.suggest_categorical("dim", [32, 64, 128])
    params["n_layers"] = trial.suggest_categorical("n_layers", [2, 3, 4])
    params["lr"] = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    params["reg"] = trial.suggest_float("reg", 1e-5, 1e-3, log=True)
    params["batch_size"] = trial.suggest_categorical("batch_size", [1024, 2048, 4096])

    if model_name == "ngcf":
        params["dropout"] = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2])

    if model_name == "sgl":
        params["ssl_lambda"] = trial.suggest_float("ssl_lambda", 0.01, 0.5, log=True)
        params["ssl_temp"] = trial.suggest_categorical("ssl_temp", [0.1, 0.2, 0.5])
        params["aug_ratio"] = trial.suggest_categorical("aug_ratio", [0.05, 0.1, 0.2])
        params["aug_type"] = "edge"

    if model_name == "simgcl":
        params["ssl_lambda"] = trial.suggest_float("ssl_lambda", 0.05, 1.0, log=True)
        params["ssl_temp"] = trial.suggest_categorical("ssl_temp", [0.1, 0.2, 0.5])
        params["noise_eps"] = trial.suggest_categorical("noise_eps", [0.05, 0.1, 0.2])

    return params


def build_model(model_name: str, params: dict, n_users: int, n_items: int):
    if model_name == "als":
        return ALSRecommender(
            factors=params["factors"],
            iterations=params["iterations"],
            regularization=params["regularization"],
        )
    cls = {"lightgcn": LightGCN, "ngcf": NGCF, "sgl": SGL, "simgcl": SimGCL}[model_name]
    init_keys = {"dim", "n_layers", "reg", "dropout", "ssl_lambda", "ssl_temp",
                 "aug_ratio", "aug_type", "noise_eps"}
    kwargs = {k: v for k, v in params.items() if k in init_keys}
    return cls(n_users=n_users, n_items=n_items, **kwargs)


def make_objective(model_name: str, dataset, graph, device: str):
    def objective(trial: optuna.Trial) -> float:
        params = sample_params(trial, model_name)

        model = build_model(model_name, params, dataset.n_users, dataset.n_items)

        if model_name == "als":
            model.fit(dataset)
            metrics = model.evaluate(dataset.val)
            return metrics["Recall@20"]

        _device = torch.device(device)
        model.to(_device)
        _graph = graph.to(_device)
        model._graph = _graph

        import numpy as np
        rng = np.random.default_rng(42)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        train = dataset.train
        users_arr = train["user_idx"].values
        items_arr = train["item_idx"].values
        n_items = dataset.n_items
        n_users = dataset.n_users
        batch_size = params["batch_size"]

        stopper = EarlyStopping(patience=10)
        best_recall = 0.0

        for epoch in range(1, 201):
            model.train()
            perm = rng.permutation(len(users_arr))
            total_loss = 0.0
            n_batches = 0

            for start in range(0, len(perm), batch_size):
                idx = perm[start: start + batch_size]
                u = torch.tensor(users_arr[idx], device=_device)
                p = torch.tensor(items_arr[idx], device=_device)
                neg = torch.tensor(rng.integers(0, n_items, size=len(idx)), device=_device)

                if model_name in ("sgl",):
                    from src.models.sgl import _augment_edge_dropout
                    g1 = _augment_edge_dropout(_graph, params["aug_ratio"], rng)
                    g2 = _augment_edge_dropout(_graph, params["aug_ratio"], rng)
                    embs_main = model._propagate(_graph)
                    embs1 = model._propagate(g1)
                    embs2 = model._propagate(g2)
                    bpr = model.bpr_loss(embs_main, u, p, neg)
                    u_u = u.unique()
                    p_u = (p + n_users).unique()
                    ssl = model._info_nce(embs1, embs2, u_u) + model._info_nce(embs1, embs2, p_u)
                    loss = bpr + params["ssl_lambda"] * ssl
                elif model_name == "simgcl":
                    embs_main = model._propagate(_graph)
                    z1 = model._propagate_with_noise(_graph)
                    z2 = model._propagate_with_noise(_graph)
                    bpr = model.bpr_loss(embs_main, u, p, neg)
                    u_u = u.unique()
                    p_u = (p + n_users).unique()
                    ssl = model._info_nce(z1, z2, u_u) + model._info_nce(z1, z2, p_u)
                    loss = bpr + params["ssl_lambda"] * ssl
                else:
                    embs = model.forward(_graph)
                    loss = model.bpr_loss(embs, u, p, neg)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            if epoch % 5 == 0:
                model.eval()
                with torch.no_grad():
                    metrics = model.evaluate(dataset.val)
                recall20 = metrics["Recall@20"]

                trial.report(recall20, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                if stopper.step(recall20):
                    best_recall = recall20
                if stopper.should_stop:
                    break

        return best_recall

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["als", "ngcf", "lightgcn", "sgl", "simgcl"])
    parser.add_argument("--dataset", required=True, choices=["toys", "cds", "steam"])
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data_dir = PROCESSED_DIR / args.dataset
    with open(data_dir / "dataset.pkl", "rb") as f:
        dataset = pickle.load(f)

    graph = None
    if args.model != "als":
        graph = torch.load(data_dir / "graph.pt", weights_only=False)

    out_dir = RESULTS_DIR / args.dataset / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{out_dir}/tuning.db"

    study = optuna.create_study(
        study_name=f"{args.model}_{args.dataset}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
        storage=storage,
        load_if_exists=True,
    )

    objective = make_objective(args.model, dataset, graph, args.device)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print(f"\nBest trial: Recall@20 = {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    best_params = {"model": args.model, **study.best_params}
    best_params["n_epochs"] = 200
    best_params["patience"] = 10
    if args.model == "als":
        best_params["use_gpu"] = False
    if args.model == "sgl" and "aug_type" not in best_params:
        best_params["aug_type"] = "edge"

    out_path = out_dir / "best_params.yaml"
    with open(out_path, "w") as f:
        yaml.dump(best_params, f, default_flow_style=False)
    print(f"Saved best params to {out_path}")


if __name__ == "__main__":
    main()
