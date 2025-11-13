# SWE-bench English–Norwegian

Experiment sandbox for comparing English vs Norwegian prompts on the `princeton-nlp/SWE-bench_Verified` benchmark. This repository will host utilities for forking, translating, and evaluating the dataset.

## Getting started

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset (defaults to the `princeton-nlp/SWE-bench_Verified` test split, which is the only split the upstream fork exposes today):

   ```bash
   python scripts/download_dataset.py --split test
   ```

   Use `--token` or set `HF_TOKEN` if the dataset requires authentication. The script exports the dataset to `data/raw/<split>.jsonl`.

You can override the dataset identifier, split, revision, or output path via command-line flags. Run `python scripts/download_dataset.py --help` for the full list of options.
