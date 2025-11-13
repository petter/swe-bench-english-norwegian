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
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

from rich.console import Console
from rich.live import Live
from rich.table import Table

from evaluate import EvaluationResult, evaluate_solution


DEFAULT_DATASET_PATH = Path("data/raw/test.jsonl")
DEFAULT_WORKTREE_ROOT = Path("/tmp/bench-english-norwegian")
DEFAULT_OUTPUT_DIR = Path("results")


class EntryStatus:
    """Track status and metrics for a dataset entry."""
    
    def __init__(self, instance_id: str, repo: str):
        self.instance_id = instance_id
        self.repo = repo
        self.status = "pending"  # pending, cloning, running, evaluating, completed, failed
        self.tokens_used = 0
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.error: str | None = None
        self.evaluation: EvaluationResult | None = None
        
    @property
    def elapsed_time(self) -> str:
        """Get elapsed time as a formatted string."""
        if not self.start_time:
            return "-"
        end = self.end_time or datetime.now()
        delta = end - self.start_time
        minutes, seconds = divmod(int(delta.total_seconds()), 60)
        if minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"
    
    @property
    def status_icon(self) -> str:
        """Get status icon for display."""
        icons = {
            "pending": "⏳",
            "cloning": "📥",
            "running": "🔄",
            "evaluating": "🧪",
            "completed": "✅",
            "failed": "❌",
        }
        return icons.get(self.status, "❓")
    
    @property
    def eval_status(self) -> str:
        """Get evaluation status for display."""
        if not self.evaluation:
            return "-"
        
        # Show LLM score if available
        if self.evaluation.llm_evaluation:
            score = self.evaluation.llm_evaluation.correctness_score
            score_str = f"({score:.1f})"
        else:
            score_str = ""
        
        if self.evaluation.success:
            return f"✅ Resolved {score_str}"
        elif self.evaluation.resolved and not self.evaluation.maintained:
            return f"⚠️  Partial {score_str}"
        else:
            return f"❌ Failed {score_str}"
    
    @property
    def test_results(self) -> str:
        """Get test results summary for display."""
        if not self.evaluation:
            return "-"
        f2p = f"{self.evaluation.fail_to_pass_passed}/{self.evaluation.fail_to_pass_total}"
        p2p = f"{self.evaluation.pass_to_pass_passed}/{self.evaluation.pass_to_pass_total}"
        return f"F2P:{f2p} P2P:{p2p}"


class ProgressTracker:
    """Manages progress tracking and TUI display for benchmark runs."""
    
    def __init__(self):
        self.entries: list[EntryStatus] = []
        self.console = Console()
        self.lock = threading.Lock()
        
    def add_entry(self, instance_id: str, repo: str) -> EntryStatus:
        """Add a new entry to track."""
        with self.lock:
            entry = EntryStatus(instance_id, repo)
            self.entries.append(entry)
            return entry
    
    def generate_table(self) -> Table:
        """Generate the Rich table for live display."""
        table = Table(title="SWE-Bench Progress", show_lines=True)
        table.add_column("#", style="cyan", width=4)
        table.add_column("Status", width=11)
        table.add_column("Instance ID", style="yellow", width=24)
        table.add_column("Repository", style="blue", width=18)
        table.add_column("Tokens", style="green", justify="right", width=8)
        table.add_column("Time", style="magenta", width=8)
        table.add_column("Result", width=19)
        table.add_column("Tests", width=30)
        
        with self.lock:
            for idx, entry in enumerate(self.entries, start=1):
                table.add_row(
                    str(idx),
                    f"{entry.status_icon} {entry.status}",
                    entry.instance_id[:25],  # Truncate long IDs
                    entry.repo.split("/")[-1][:20],  # Show just repo name, truncated
                    str(entry.tokens_used) if entry.tokens_used > 0 else "-",
                    entry.elapsed_time,
                    entry.eval_status,
                    entry.test_results,
                )
                
            # Add summary row
            total_tokens = sum(e.tokens_used for e in self.entries)
            completed = sum(1 for e in self.entries if e.status == "completed")
            failed = sum(1 for e in self.entries if e.status == "failed")
            total = len(self.entries)
            
            # Calculate evaluation metrics
            evaluated = sum(1 for e in self.entries if e.evaluation is not None)
            resolved = sum(1 for e in self.entries if e.evaluation and e.evaluation.success)
            partial = sum(1 for e in self.entries if e.evaluation and e.evaluation.resolved and not e.evaluation.maintained)
            
            # Calculate average LLM score
            llm_scores = [e.evaluation.llm_evaluation.correctness_score 
                         for e in self.entries 
                         if e.evaluation and e.evaluation.llm_evaluation]
            avg_llm_score = sum(llm_scores) / len(llm_scores) if llm_scores else 0.0
            
            accuracy = f"{resolved}/{evaluated}" if evaluated > 0 else "0/0"
            accuracy_pct = f"{resolved*100//evaluated}%" if evaluated > 0 else "0%"
            
            table.add_section()
            table.add_row(
                "",
                "SUMMARY",
                f"Total:{total} Done:{completed}",
                f"Failed:{failed}",
                f"{total_tokens:,}",
                "",
                f"Pass:{resolved} Part:{partial} Fail:{evaluated-resolved-partial}",
                f"Acc:{accuracy_pct} LLM:{avg_llm_score:.2f}",
                style="bold",
            )
        
        return table


tracker = ProgressTracker()


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
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of parallel workers to run (default: 32).",
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


def run_git_command(args: list[str], cwd: Path | None = None, quiet: bool = False) -> None:
    cmd = ["git", *args]
    if cwd is not None:
        cmd.insert(1, "-C")
        cmd.insert(2, str(cwd))
    subprocess.run(cmd, check=True, capture_output=quiet, text=True)


def ensure_repository(repo_slug: str, root: Path, instance_id: str) -> Path:
    """Clone or reuse a repository for a specific instance.
    
    Each instance gets its own directory to avoid conflicts when running in parallel.
    """
    target_dir = root / instance_id
    root.mkdir(parents=True, exist_ok=True)

    if target_dir.exists():
        if not (target_dir / ".git").is_dir():
            raise RuntimeError(
                f"Existing directory {target_dir} is not a git repository."
            )
        # Clean up any dirty state before fetching
        run_git_command(["reset", "--hard"], cwd=target_dir, quiet=True)
        run_git_command(["clean", "-fdx"], cwd=target_dir, quiet=True)
        run_git_command(["fetch", "--all", "--tags", "--prune"], cwd=target_dir, quiet=True)
    else:
        repo_url = f"https://github.com/{repo_slug}.git"
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--quiet", repo_url, str(target_dir)],
            check=True,
            capture_output=True,
            text=True,
        )

    return target_dir


def checkout_commit(repo_path: Path, commit: str) -> None:
    run_git_command(["checkout", "--force", commit], cwd=repo_path, quiet=True)


def run_model(model_name: str, entry: dict, repo_path: Path, output_dir: Path, status: EntryStatus, live) -> dict | None:
    """Run the specified model/agent on the repository.
    
    For OpenCode, this invokes the CLI with the problem statement and lets
    the agent work autonomously to solve the issue.
    
    Returns a result dictionary if applicable, None otherwise.
    """
    instance_id = entry.get("instance_id", "unknown")
    
    if model_name == "opencode":
        return run_opencode(entry, repo_path, output_dir, status, live)
    else:
        status.status = "failed"
        status.error = f"Unknown model: {model_name}"
        status.end_time = datetime.now()
        return None


def run_opencode(entry: dict, repo_path: Path, output_dir: Path, status: EntryStatus, live) -> dict:
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
        status.status = "failed"
        status.error = "No problem statement"
        status.end_time = datetime.now()
        result_data["error"] = "No problem statement"
        return result_data
    
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
    except Exception as e:
        status.status = "failed"
        status.error = f"Could not create opencode.json: {e}"
        status.end_time = datetime.now()
        result_data["error"] = status.error
        return result_data
    
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
    
    # Invoke OpenCode CLI in non-interactive mode with streaming output
    try:
        # Set environment to disable TTY/interactive mode
        env = os.environ.copy()
        env["CI"] = "true"  # Many tools detect CI mode and disable interactive features
        
        status.status = "running"
        status.start_time = datetime.now()
        live.update(tracker.generate_table(), refresh=True)
        
        # Start a background thread to update the display periodically
        stop_updater = threading.Event()
        
        def periodic_update():
            while not stop_updater.is_set():
                live.update(tracker.generate_table(), refresh=True)
                time.sleep(0.5)  # Update every 500ms
        
        updater_thread = threading.Thread(target=periodic_update, daemon=True)
        updater_thread.start()
        
        try:
            # Run opencode with Popen to stream output line by line
            process = subprocess.Popen(
                ["opencode", "run", "--format", "json", prompt],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,  # Line buffered
                cwd=str(repo_path),
                env=env,
            )
            
            stdout_lines = []
            stderr_lines = []
            json_events = []
            
            # Read stdout line by line as they come in
            try:
                if process.stdout:
                    for line in process.stdout:
                        line = line.rstrip()
                        if line:
                            stdout_lines.append(line)
                            try:
                                event = json.loads(line)
                                json_events.append(event)
                                
                                # Track token usage from step_finish events
                                if event.get("type") == "step_finish":
                                    part = event.get("part", {})
                                    tokens_data = part.get("tokens", {})
                                    if tokens_data:
                                        # Sum up input, output, and cache read tokens
                                        input_tokens = tokens_data.get("input", 0)
                                        output_tokens = tokens_data.get("output", 0)
                                        cache_read = tokens_data.get("cache", {}).get("read", 0)
                                        
                                        # Update cumulative token count
                                        step_total = input_tokens + output_tokens + cache_read
                                        status.tokens_used += step_total
                            except json.JSONDecodeError:
                                # Not JSON, skip
                                pass
                
                # Wait for process to complete and get exit code
                process.wait(timeout=600)
                
                # Read any remaining stderr
                if process.stderr:
                    stderr_output = process.stderr.read()
                    if stderr_output:
                        stderr_lines = stderr_output.strip().split("\n")
                    
            except subprocess.TimeoutExpired:
                process.kill()
                raise
            
            result_data["exit_code"] = process.returncode
            result_data["stdout"] = "\n".join(stdout_lines)
            result_data["stderr"] = "\n".join(stderr_lines) if stderr_lines else ""
            result_data["success"] = process.returncode == 0
            result_data["json_events"] = json_events
            
            # Calculate final token count from all step_finish events
            # (in case we missed any during streaming)
            total_tokens = 0
            for event in json_events:
                if event.get("type") == "step_finish":
                    part = event.get("part", {})
                    tokens_data = part.get("tokens", {})
                    if tokens_data:
                        input_tokens = tokens_data.get("input", 0)
                        output_tokens = tokens_data.get("output", 0)
                        cache_read = tokens_data.get("cache", {}).get("read", 0)
                        total_tokens += input_tokens + output_tokens + cache_read
            
            status.tokens_used = total_tokens
            result_data["tokens_used"] = status.tokens_used
            
            # Evaluate the solution if OpenCode completed successfully
            if process.returncode == 0:
                status.status = "evaluating"
                live.update(tracker.generate_table(), refresh=True)
                
                try:
                    eval_result = evaluate_solution(
                        repo_path, 
                        entry, 
                        test_timeout=300,
                        use_llm=True,  # Enable LLM evaluation
                    )
                    status.evaluation = eval_result
                    result_data["evaluation"] = {
                        "resolved": eval_result.resolved,
                        "maintained": eval_result.maintained,
                        "fail_to_pass_total": eval_result.fail_to_pass_total,
                        "fail_to_pass_passed": eval_result.fail_to_pass_passed,
                        "pass_to_pass_total": eval_result.pass_to_pass_total,
                        "pass_to_pass_passed": eval_result.pass_to_pass_passed,
                        "status": eval_result.status,
                        "success": eval_result.success,
                        "error": eval_result.error,
                    }
                    
                    # Add LLM evaluation results if available
                    if eval_result.llm_evaluation:
                        result_data["llm_evaluation"] = {
                            "correctness_score": eval_result.llm_evaluation.correctness_score,
                            "reasoning": eval_result.llm_evaluation.reasoning,
                            "addresses_issue": eval_result.llm_evaluation.addresses_issue,
                            "implementation_quality": eval_result.llm_evaluation.implementation_quality,
                            "potential_issues": eval_result.llm_evaluation.potential_issues,
                        }
                    
                    result_data["test_output"] = eval_result.test_output
                except Exception as eval_error:
                    status.error = f"Evaluation error: {eval_error}"
                    result_data["evaluation_error"] = str(eval_error)
            
            status.status = "completed" if process.returncode == 0 else "failed"
            status.end_time = datetime.now()
            
            if not result_data["success"]:
                status.error = "Non-zero exit code"
        
        finally:
            # Stop the background updater thread
            stop_updater.set()
            updater_thread.join(timeout=1)
            
    except FileNotFoundError:
        error_msg = "OpenCode CLI not found in PATH"
        status.status = "failed"
        status.error = error_msg
        status.end_time = datetime.now()
        result_data["error"] = error_msg
    except subprocess.TimeoutExpired:
        error_msg = "OpenCode timed out after 10 minutes"
        status.status = "failed"
        status.error = error_msg
        status.end_time = datetime.now()
        result_data["error"] = error_msg
    except Exception as e:
        error_msg = f"Error running OpenCode: {e}"
        status.status = "failed"
        status.error = error_msg
        status.end_time = datetime.now()
        result_data["error"] = error_msg
    finally:
        # Clean up the config file
        try:
            if config_file.exists():
                config_file.unlink()
        except Exception:
            pass
    
    # Save individual result
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / f"{instance_id}.json"
    with result_file.open("w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)
    
    return result_data


def process_single_entry(
    entry: dict, 
    status: EntryStatus, 
    model_name: str, 
    workdir_root: Path, 
    output_dir: Path, 
    live: Live
) -> dict | None:
    """Process a single entry (used by worker threads)."""
    repo = entry["repo"]
    commit = entry["base_commit"]
    instance_id = entry["instance_id"]
    
    try:
        # Update status to cloning
        status.status = "cloning"
        
        repo_path = ensure_repository(repo, workdir_root, instance_id)
        checkout_commit(repo_path, commit)
        
        # Run the model (this will update status internally)
        result = run_model(model_name, entry, repo_path, output_dir, status, live)
        return result
        
    except Exception as e:
        status.status = "failed"
        status.error = str(e)
        status.end_time = datetime.now()
        return None


def process_entries(
    entries: Iterable[dict], model_name: str, workdir_root: Path, output_dir: Path, num_workers: int = 32
) -> None:
    """Process each entry by checking out the repo and running the model in parallel."""
    results = []
    
    # Convert entries to list to get count and pre-register them
    entry_list = list(entries)
    
    # Pre-register all entries in the tracker
    for entry in entry_list:
        tracker.add_entry(entry["instance_id"], entry["repo"])
    
    # Create a mapping from entry to status
    entry_status_map = {entry["instance_id"]: tracker.entries[idx] for idx, entry in enumerate(entry_list)}
    
    # Start the live display
    with Live(tracker.generate_table(), refresh_per_second=4, console=tracker.console, auto_refresh=True) as live:
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_entry = {
                executor.submit(
                    process_single_entry,
                    entry,
                    entry_status_map[entry["instance_id"]],
                    model_name,
                    workdir_root,
                    output_dir,
                    live
                ): entry
                for entry in entry_list
            }
            
            # Process completed futures as they finish
            for future in as_completed(future_to_entry):
                entry = future_to_entry[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    # Handle any unexpected exceptions from the worker
                    status = entry_status_map[entry["instance_id"]]
                    status.status = "failed"
                    status.error = f"Worker exception: {e}"
                    status.end_time = datetime.now()
    
    # Save summary of all results after TUI closes
    if results:
        summary_file = output_dir / "summary.json"
        with summary_file.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "total": len(results),
                    "successful": sum(1 for r in results if r.get("success")),
                    "failed": sum(1 for r in results if not r.get("success")),
                    "total_tokens": sum(r.get("tokens_used", 0) for r in results),
                    "results": results,
                },
                f,
                indent=2,
            )
        
        print(f"\n{'='*80}")
        print(f"Summary saved to {summary_file}")
        print(f"Total: {len(results)}, Successful: {sum(1 for r in results if r.get('success'))}")
        print(f"Total tokens used: {sum(r.get('tokens_used', 0) for r in results):,}")
        print(f"{'='*80}")


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file {dataset_path} does not exist.")

    try:
        entries = iter_entries(dataset_path, args.limit)
        process_entries(
            entries, args.model, Path(args.workdir_root), Path(args.output_dir), args.workers
        )
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}.", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
