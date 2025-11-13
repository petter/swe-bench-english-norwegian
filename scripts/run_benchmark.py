#!/usr/bin/env python3
"""Prepare SWE-bench entries for model evaluation.

The script clones (or reuses) repositories referenced by the dataset, checks out
the commit specified for each entry, and invokes OpenCode to solve each issue.

For OpenCode integration, each task is executed in its own shell environment with
the working directory set to the repository root. This ensures OpenCode operates
within the proper context, with all permissions pre-configured to allow automated
execution without user prompts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator


DEFAULT_DATASET_PATH = Path("data/raw/test.jsonl")
DEFAULT_WORKTREE_ROOT = Path("/tmp/bench-english-norwegian")
DEFAULT_OUTPUT_DIR = Path("results")


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
        default="opencode",
        help="Identifier of the model/agent to run for each entry (default: opencode).",
    )
    parser.add_argument(
        "--workdir-root",
        default=str(DEFAULT_WORKTREE_ROOT),
        help=(
            "Root directory for the temporary repositories. Each repo will "
            "live under <root>/<repo-name>."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save results and logs (default: results).",
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


def run_model(model_name: str, entry: dict, repo_path: Path, output_dir: Path) -> dict | None:
    """Run the specified model/agent on the repository.
    
    For OpenCode, this invokes the CLI with the problem statement and lets
    the agent work autonomously to solve the issue.
    
    Returns a result dictionary if applicable, None otherwise.
    """
    instance_id = entry.get("instance_id", "unknown")
    
    if model_name == "opencode":
        return run_opencode(entry, repo_path, output_dir)
    else:
        print(
            f"[TODO] Model '{model_name}' would run on {instance_id} at {repo_path}"
        )
        return None


def run_opencode(entry: dict, repo_path: Path, output_dir: Path) -> dict:
    """Invoke OpenCode to solve the issue described in the entry.
    
    Returns a result dictionary with success status and metadata.
    """
    instance_id = entry.get("instance_id", "unknown")
    problem_statement = entry.get("problem_statement", "")
    fail_to_pass = entry.get("FAIL_TO_PASS", "")
    
    result_data = {
        "instance_id": instance_id,
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "error": None,
    }
    
    if not problem_statement:
        print(f"⚠ No problem statement found for {instance_id}, skipping.")
        result_data["error"] = "No problem statement"
        return result_data
    
    print(f"\n{'='*80}")
    print(f"Running OpenCode on {instance_id}")
    print(f"Repository: {repo_path}")
    print(f"{'='*80}\n")
    
    # Set up OpenCode config in the repository to allow all permissions
    config_file = repo_path / "opencode.json"
    config_content = {
        "$schema": "https://opencode.ai/config.json",
        "permission": {
            "edit": "allow",
            "bash": "allow",
            "webfetch": "allow",
        },
    }
    
    try:
        with config_file.open("w", encoding="utf-8") as f:
            json.dump(config_content, f, indent=2)
        print(f"✓ Created opencode.json with 'allow' permissions")
    except Exception as e:
        print(f"⚠ Warning: Could not create opencode.json: {e}")
    
    # Construct the prompt for OpenCode
    prompt = f"""Please solve the following issue:

{problem_statement}

The following tests should pass after your fix:
{fail_to_pass}

Please:
1. Understand the problem by reading the relevant code
2. Implement a fix for the issue
3. Run the tests to verify your solution works
"""
    
    # Invoke OpenCode CLI in non-interactive mode
    # Run it with cwd set to the repository directory so it operates within that context
    # Use --format json to get structured output instead of TUI
    try:
        # Set environment to disable TTY/interactive mode
        env = os.environ.copy()
        env["CI"] = "true"  # Many tools detect CI mode and disable interactive features
        
        # Run opencode run with the prompt, executing in the repo directory
        result = subprocess.run(
            ["opencode", "run", "--format", "json", prompt],
            check=False,  # Don't raise on non-zero exit
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout per issue
            stdin=subprocess.DEVNULL,  # Disable stdin to prevent interactive prompts
            cwd=str(repo_path),  # Run in the repository directory
            env=env,
        )
        
        result_data["exit_code"] = result.returncode
        result_data["stdout"] = result.stdout
        result_data["stderr"] = result.stderr
        result_data["success"] = result.returncode == 0
        
        print(f"\n{'='*80}")
        print(f"OpenCode exit code: {result.returncode}")
        print(f"{'='*80}")
        
        # Parse JSON events from stdout if format is json
        json_events = []
        if result.stdout:
            for line in result.stdout.strip().split("\n"):
                try:
                    event = json.loads(line)
                    json_events.append(event)
                    # Print relevant events
                    if event.get("type") == "text":
                        print(event.get("text", ""))
                    elif event.get("type") == "error":
                        print(f"Error: {event.get('message', '')}")
                except json.JSONDecodeError:
                    # Not JSON, print as-is
                    print(line)
        
        result_data["json_events"] = json_events
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
            
    except FileNotFoundError:
        error_msg = "OpenCode CLI not found in PATH"
        print(f"❌ {error_msg}")
        print("   You may need to install it first.")
        result_data["error"] = error_msg
    except subprocess.TimeoutExpired:
        error_msg = "OpenCode timed out after 10 minutes"
        print(f"⏱ {error_msg}")
        result_data["error"] = error_msg
    except Exception as e:
        error_msg = f"Error running OpenCode: {e}"
        print(f"❌ {error_msg}")
        result_data["error"] = error_msg
    finally:
        # Clean up the config file
        try:
            if config_file.exists():
                config_file.unlink()
                print("✓ Cleaned up opencode.json")
        except Exception as e:
            print(f"⚠ Warning: Could not remove opencode.json: {e}")
    
    # Save individual result
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / f"{instance_id}.json"
    with result_file.open("w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)
    
    print(f"\n✓ Result saved to {result_file}")
    
    return result_data


def process_entries(
    entries: Iterable[dict], model_name: str, workdir_root: Path, output_dir: Path
) -> None:
    """Process each entry by checking out the repo and running the model."""
    results = []
    
    for idx, entry in enumerate(entries, start=1):
        repo = entry["repo"]
        commit = entry["base_commit"]
        print(f"\n=== Entry {idx}: {entry['instance_id']} ({repo}@{commit}) ===")
        repo_path = ensure_repository(repo, workdir_root)
        checkout_commit(repo_path, commit)
        result = run_model(model_name, entry, repo_path, output_dir)
        if result:
            results.append(result)
    
    # Save summary of all results
    if results:
        summary_file = output_dir / "summary.json"
        with summary_file.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "total": len(results),
                    "successful": sum(1 for r in results if r.get("success")),
                    "failed": sum(1 for r in results if not r.get("success")),
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"\n{'='*80}")
        print(f"Summary saved to {summary_file}")
        print(f"Total: {len(results)}, Successful: {sum(1 for r in results if r.get('success'))}")
        print(f"{'='*80}")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file {dataset_path} does not exist.")

    try:
        entries = iter_entries(dataset_path, args.limit)
        process_entries(
            entries, args.model, Path(args.workdir_root), Path(args.output_dir)
        )
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}.", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
