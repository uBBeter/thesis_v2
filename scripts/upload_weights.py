"""
Upload trained checkpoints and metrics to HuggingFace Hub.

Run this on the GPU instance after training completes:
  python scripts/upload_weights.py --repo your-username/thesis-gnn-recsys

The repo will be public so supervisors can download without an account.
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True,
                        help="HuggingFace repo ID, e.g. your-username/thesis-gnn-recsys")
    parser.add_argument("--private", action="store_true",
                        help="Make the repo private (supervisors will need HF account)")
    args = parser.parse_args()

    api = HfApi()
    create_repo(args.repo, exist_ok=True, private=args.private, repo_type="model")
    print(f"Repo ready: https://huggingface.co/{args.repo}\n")

    results_dir = Path("results")
    if not results_dir.exists():
        print("No results/ directory found. Run training first.")
        return

    for f in sorted(results_dir.rglob("*")):
        if f.suffix not in (".pt", ".json"):
            continue
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=str(f),
            repo_id=args.repo,
            repo_type="model",
        )
        print(f"Uploaded {f}")

    print(f"\nDone. Share this URL with your supervisors:")
    print(f"  https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
