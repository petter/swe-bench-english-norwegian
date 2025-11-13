#!/usr/bin/env python3
"""Prepare SWE-bench entries for model evaluation.

The script clones (or reuses) repositories referenced by the dataset, checks out
the commit specified for each entry, and provides a hook for invoking a model
runner. Only the repository setup is implemented for now; the actual model
integration can be plugged into :func:`run_model` later on.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_DATASET_PATH = Path("data/raw/test.jsonl")
DEFAULT_WORKTREE_ROOT = Path("/tmp/bench-english-norwegian")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare SWE-bench repositories and optionally run a model over a "
            "subset of the dataset."
        )
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET_PATH),
        help="Path to the SWE-bench jsonl file (default: data/raw/test.jsonl).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of dataset entries to process. Useful for quick "
            "tests when the full 500-entry dataset is unnecessary."
        ),
    )
    parser.add_argument(
        "--model",
        default="unspecified",
        help="Identifier of the model/agent to run for each entry.",
    )
    parser.add_argument(
        "--workdir-root",
        default=str(DEFAULT_WORKTREE_ROOT),
        help=(
            "Root directory for the temporary repositories. Each repo will "
            "live under <root>/<repo-name>."
        ),
    )
    return parser.parse_args()


def iter_entries(dataset_path: Path, limit: int | None) -> Iterator[dict]:
    processed = 0
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and processed >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)
            processed += 1


def run_git_command(args: list[str], cwd: Path | None = None) -> None:
    cmd = ["git", *args]
    if cwd is not None:
        cmd.insert(1, "-C")
        cmd.insert(2, str(cwd))
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def ensure_repository(repo_slug: str, root: Path) -> Path:
    repo_name = repo_slug.split("/")[-1]
    target_dir = root / repo_name
    root.mkdir(parents=True, exist_ok=True)

    if target_dir.exists():
        if not (target_dir / ".git").is_dir():
            raise RuntimeError(
                f"Existing directory {target_dir} is not a git repository."
            )
        print(f"Reusing existing clone at {target_dir}...")
        run_git_command(["fetch", "--all", "--tags", "--prune"], cwd=target_dir)
    else:
        repo_url = f"https://github.com/{repo_slug}.git"
        print(f"Cloning {repo_url} into {target_dir}...")
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", repo_url, str(target_dir)], check=True)

    return target_dir


def checkout_commit(repo_path: Path, commit: str) -> None:
    print(f"Checking out commit {commit} in {repo_path}...")
    run_git_command(["checkout", "--force", commit], cwd=repo_path)


def run_model(model_name: str, entry: dict, repo_path: Path) -> None:
    # Placeholder hook for the actual model invocation.
    print(
        "[TODO] Model '%s' would run on %s located at %s"
        % (model_name, entry.get("instance_id"), repo_path)
    )


def process_entries(entries: Iterable[dict], model_name: str, workdir_root: Path) -> None:
    for idx, entry in enumerate(entries, start=1):
        repo = entry["repo"]
        commit = entry["base_commit"]
        print(f"\n=== Entry {idx}: {entry['instance_id']} ({repo}@{commit}) ===")
        repo_path = ensure_repository(repo, workdir_root)
        checkout_commit(repo_path, commit)
        run_model(model_name, entry, repo_path)


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file {dataset_path} does not exist.")

    try:
        entries = iter_entries(dataset_path, args.limit)
        process_entries(entries, args.model, Path(args.workdir_root))
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}.", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
