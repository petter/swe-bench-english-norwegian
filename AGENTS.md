# Agent Guidelines

## Running Scripts
- Download dataset: `python scripts/download_dataset.py --split test`
- Run benchmark: `python scripts/run_benchmark.py --dataset data/raw/test.jsonl --limit 5`
- Run with parallel workers: `python scripts/run_benchmark.py --dataset data/raw/test.jsonl --workers 32`
- Evaluate solution: Use `evaluate.py` functions (imported by `run_benchmark.py`)
- No test suite currently exists (TDD encouraged for future work)

## Code Style
- Python 3.10+ with `from __future__ import annotations` as first import
- Type hints: Use `str | None` (PEP 604) over `Optional[str]`; always annotate function params/returns
- Imports: `from __future__ import annotations` → stdlib → third-party → local
- Docstrings: Module-level triple-quoted docstrings; concise function docstrings with Args/Returns
- Data structures: Prefer `@dataclass` for structured data over dicts
- Error handling: Custom exception classes inheriting from `RuntimeError`, `ValueError`, etc.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants
- CLI: Use `argparse` with help text; support env vars (e.g., `HF_TOKEN`, `OPENROUTER_API_KEY`)
- File operations: Always use `pathlib.Path` over `os.path`
- Subprocess: Use `subprocess.run(check=True)` for git/external commands; `capture_output=True` for silent ops
- Formatting: 4-space indentation, max line length ~88-100 chars (Black-compatible)
- Output: Use descriptive `print()` for user messages; `rich` library for TUI/tables
- Type safety: Avoid `Any`; use proper unions and `Literal` types where appropriate
