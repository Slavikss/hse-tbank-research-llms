"""Upload model and eval datasets to Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path


def upload_to_hf(
    model_dir: str | Path,
    dataset_dir: str | Path,
    model_repo_id: str,
    dataset_repo_id: str,
    private: bool = False,
) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required. Install it first.") from exc

    api = HfApi()

    api.create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True, private=private)
    api.upload_folder(
        repo_id=model_repo_id,
        repo_type="model",
        folder_path=str(model_dir),
        commit_message="Upload trained modular arithmetic model",
    )

    api.create_repo(repo_id=dataset_repo_id, repo_type="dataset", exist_ok=True, private=private)
    api.upload_folder(
        repo_id=dataset_repo_id,
        repo_type="dataset",
        folder_path=str(dataset_dir),
        commit_message="Upload fixed eval datasets",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload model and datasets to Hugging Face Hub")
    parser.add_argument("--model-dir", default="outputs/merged_model")
    parser.add_argument("--dataset-dir", default="data/eval")
    parser.add_argument("--model-repo-id", required=True, help="e.g. username/dz2-mod-arith-model")
    parser.add_argument("--dataset-repo-id", required=True, help="e.g. username/dz2-mod-arith-eval")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    upload_to_hf(
        model_dir=args.model_dir,
        dataset_dir=args.dataset_dir,
        model_repo_id=args.model_repo_id,
        dataset_repo_id=args.dataset_repo_id,
        private=args.private,
    )
    print("Upload completed.")


if __name__ == "__main__":
    main()
