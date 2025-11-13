# Agent Guidelines

## Running Scripts
- Download dataset: `python scripts/download_dataset.py --split test`
- Run benchmark: `python scripts/run_benchmark.py --dataset data/raw/test.jsonl --limit 5`
- Run benchmark with parallel workers: `python scripts/run_benchmark.py --dataset data/raw/test.jsonl --workers 32`
- No test suite currently exists

## Code Style
- Python 3.10+ with `from __future__ import annotations` for forward compatibility
- Type hints: Use `str | None` (PEP 604) over `Optional[str]`
- Imports: stdlib → third-party → local, with `from __future__ import annotations` first
- Docstrings: Module-level docstrings with triple quotes; concise function docstrings
- Error handling: Custom exception classes (inherit from `RuntimeError`, `ValueError`, etc.)
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_CASE` for constants
- CLI: Use `argparse` with clear help text; support env vars (e.g., `HF_TOKEN`)
- File operations: Use `pathlib.Path` over `os.path`
- Subprocess: Use `subprocess.run(check=True)` for git/external commands
- Formatting: 4-space indentation, max line length ~88-100 chars
- Output: Use descriptive `print()` statements for user-facing progress messages
