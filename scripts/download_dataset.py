#!/usr/bin/env python3
"""Utility for downloading SWE-bench datasets from Hugging Face."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_dataset


class SplitNotFoundError(RuntimeError):
    """Raised when the requested dataset split does not exist."""


def raise_split_error(dataset_id: str, split: str, original_error: Exception) -> None:
    message = (
        f"Split '{split}' is unavailable for dataset '{dataset_id}'. "
        "Check the dataset card on Hugging Face for the list of supported splits."
    )
    raise SplitNotFoundError(message) from original_error


DEFAULT_DATASET_ID = "princeton-nlp/SWE-bench_Verified"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SWE-bench (or compatible) datasets from Hugging Face"
    )
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help="Full Hugging Face dataset identifier (e.g. swe-bench/SWE-bench_Verified)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help=(
            "Dataset split to download. The upstream SWE-bench_Verified fork currently "
            "only exposes a 'test' split."
        ),
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Optional dataset revision (branch, tag, or commit hash).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Directory where the exported dataset split will be written.",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="Optional output filename. Defaults to <split>.jsonl",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get("HF_DATASET_CACHE"),
        help="Custom cache directory for datasets library.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token. Falls back to HF_TOKEN env var.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    return parser.parse_args()


def download_split(
    dataset_id: str,
    split: str,
    revision: str,
    cache_dir: str | None,
    token: str | None,
):
    """Load the dataset split via the datasets Hub API."""
    print(f"Downloading {dataset_id} split={split} revision={revision}...")
    try:
        dataset = load_dataset(
            path=dataset_id,
            split=split,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
        )
    except ValueError as err:
        if "Unknown split" in str(err):
            raise_split_error(dataset_id, split, err)
        raise
    print(f"Loaded {len(dataset)} rows.")
    return dataset


def export_dataset(dataset, output_path: Path, overwrite: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file {output_path} exists. Use --overwrite to replace it."
        )
    print(f"Writing dataset to {output_path}...")
    dataset.to_json(str(output_path))
    print("Done.")


def main() -> None:
    args = parse_args()
    dataset = download_split(
        dataset_id=args.dataset_id,
        split=args.split,
        revision=args.revision,
        cache_dir=args.cache_dir,
        token=args.token,
    )
    filename = args.filename or f"{args.split}.jsonl"
    output_path = Path(args.output_dir) / filename
    export_dataset(dataset, output_path, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
