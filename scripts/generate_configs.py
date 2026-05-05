import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_DIR = Path("results")
CONFIGS_DIR = Path("configs/tuned")
DATASETS = ["toys", "cds", "steam"]
MODELS = ["als", "ngcf", "lightgcn", "sgl", "simgcl"]


def main():
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    found = 0

    for dataset in DATASETS:
        for model in MODELS:
            params_path = RESULTS_DIR / dataset / model / "best_params.yaml"
            if not params_path.exists():
                print(f"Missing: {params_path} — run tune.py first")
                continue

            with open(params_path) as f:
                params = yaml.safe_load(f)

            out_path = CONFIGS_DIR / f"{model}_{dataset}.yaml"
            with open(out_path, "w") as f:
                yaml.dump(params, f, default_flow_style=False)

            print(f"Written: {out_path}")
            found += 1

    print(f"\nGenerated {found} config files in {CONFIGS_DIR}/")
    if found > 0:
        print("\nTo retrain with tuned configs, run:")
        for dataset in DATASETS:
            for model in MODELS:
                cfg = CONFIGS_DIR / f"{model}_{dataset}.yaml"
                if cfg.exists():
                    device = "cpu" if model == "als" else "cuda"
                    print(f"  python scripts/train.py --config {cfg} "
                          f"--dataset {dataset} --device {device}")


if __name__ == "__main__":
    main()
