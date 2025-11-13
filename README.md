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

## Running the benchmark with OpenCode

The benchmark script now supports running OpenCode as an agent to solve SWE-bench issues with a live progress TUI, parallel execution, and automatic evaluation:

```bash
python scripts/run_benchmark.py --dataset data/raw/test.jsonl --limit 5
```

This will:
1. Clone the necessary repositories to `/tmp/bench-english-norwegian/<instance_id>` (each instance gets its own directory)
2. Check out the appropriate commit for each issue
3. Invoke OpenCode in non-interactive mode with `--format json` to solve the issue
4. **Automatically evaluate the solution** by:
   - Running FAIL_TO_PASS tests (tests that should now pass after the fix)
   - Running PASS_TO_PASS tests (tests that should still pass after the fix)
   - Using an LLM via OpenRouter to assess solution quality
5. Display a live TUI showing progress, token usage, timing, and **evaluation results** for each entry
6. Save detailed results including test outputs and LLM assessments to the `results/` directory

By default, the script runs 32 workers in parallel to process multiple issues concurrently. Each instance gets its own isolated repository directory to avoid conflicts. You can adjust the number of workers with the `--workers` flag.

### Evaluation System

After OpenCode completes each task, the system automatically evaluates whether the solution is correct:

#### Test-Based Evaluation
- **FAIL_TO_PASS tests**: Tests from the dataset that should now pass after the fix is applied
- **PASS_TO_PASS tests**: Tests that should continue passing after the fix (regression testing)
- Tests are run using pytest or unittest depending on the repository setup

#### LLM-Based Evaluation
- Uses OpenRouter (Claude Sonnet 4.5 by default) to review the solution
- Compares agent's changes against the gold patch from the dataset
- Provides:
  - Correctness score (0.0-1.0)
  - Assessment of whether it addresses the core issue
  - Implementation quality rating
  - Identification of potential problems
  - Detailed reasoning

To enable LLM evaluation, set the `OPENROUTER_API_KEY` environment variable:

```bash
export OPENROUTER_API_KEY="your-api-key"
python scripts/run_benchmark.py --dataset data/raw/test.jsonl --limit 5
```

Without an API key, only test-based evaluation will be performed.

### Live Progress TUI

The script displays a real-time terminal UI that shows:
- Status of each dataset entry (pending, cloning, running, evaluating, completed, failed)
- Token usage for each OpenCode session
- Elapsed time for each entry
- **Evaluation results**: ✅ Resolved / ⚠️ Partial / ❌ Failed
- **Test results**: FAIL_TO_PASS and PASS_TO_PASS test counts
- **LLM correctness scores** (when available)
- Summary statistics:
  - Total entries, completed, failed
  - Accuracy (percentage of fully resolved issues)
  - Average LLM correctness score
  - Total tokens used

The TUI updates live as OpenCode processes and evaluates each entry, so you can track progress in real-time.

### Options

- `--dataset`: Path to the SWE-bench jsonl file (default: `data/raw/test.jsonl`)
- `--limit`: Maximum number of entries to process (useful for testing)
- `--model`: Model/agent to use (default: `opencode`)
- `--workdir-root`: Root directory for cloned repositories (default: `/tmp/bench-english-norwegian`)
- `--output-dir`: Directory to save results and logs (default: `results`)
- `--workers`: Number of parallel workers to run (default: `32`)

### Example: Run on first 10 issues with 5 parallel workers

```bash
python scripts/run_benchmark.py --limit 10 --workers 5 --output-dir results/run1
```

### How it works

The script runs OpenCode in non-interactive mode for each task by:
- Spawning OpenCode in a shell with the working directory set to the repository (`cwd=repo_path`)
- Using `--format json` to get structured output instead of the TUI
- Setting `CI=true` environment variable to disable interactive features
- Redirecting stdin to prevent interactive prompts
- Creating a temporary `opencode.json` config file in each repository with all permissions set to `"allow"` to avoid permission prompts
- Each issue has a 10-minute timeout
- Parsing JSON events from OpenCode to track token usage in real-time

After OpenCode completes:
1. The agent's changes are captured as a git diff
2. Test patches from the dataset are applied (if present)
3. FAIL_TO_PASS and PASS_TO_PASS tests are executed
4. An LLM reviews the solution and provides detailed feedback
5. All results are saved to JSON files for analysis

This means each task gets its own isolated execution environment where OpenCode operates directly within the repository context, just like a developer would work in that directory.

The JSON events from OpenCode are parsed and stored along with the full output for analysis. Token usage is extracted from `usage` or `token_usage` events in the JSON stream. The config file is automatically cleaned up after execution.

### Results Format

Each run produces:
- Individual result files: `results/<instance_id>.json` containing:
  - OpenCode execution details (stdout, stderr, exit code)
  - Token usage
  - Test evaluation results (pass/fail counts)
  - LLM evaluation (scores, reasoning, potential issues)
  - Full test output
- Summary file: `results/summary.json` with aggregate statistics

### Requirements

- OpenCode must be installed and available in your PATH
- The script will invoke `opencode run --format json` with the problem statement for each issue
- Optional: Set `OPENROUTER_API_KEY` for LLM-based evaluation
