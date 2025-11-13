#!/usr/bin/env python3
"""Evaluation utilities for SWE-bench solutions.

This module provides functionality to evaluate whether an agent's solution
correctly solves a SWE-bench issue by:
1. Running the FAIL_TO_PASS tests (should now pass after the fix)
2. Running the PASS_TO_PASS tests (should still pass after the fix)
3. Using an LLM to review the changes and assess correctness
4. Comparing results to determine if the solution is correct
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import requests


@dataclass
class LLMEvaluation:
    """Results from LLM-based evaluation."""
    
    correctness_score: float  # 0.0-1.0 score for correctness
    reasoning: str  # LLM's reasoning about the solution
    addresses_issue: bool  # Does it address the problem statement?
    implementation_quality: str  # "good", "acceptable", "poor"
    potential_issues: list[str]  # List of potential problems identified
    

@dataclass
class EvaluationResult:
    """Results from evaluating a solution."""
    
    resolved: bool  # True if FAIL_TO_PASS tests now pass
    maintained: bool  # True if PASS_TO_PASS tests still pass
    fail_to_pass_total: int
    fail_to_pass_passed: int
    pass_to_pass_total: int
    pass_to_pass_passed: int
    error: str | None = None
    test_output: str = ""
    llm_evaluation: LLMEvaluation | None = None
    
    @property
    def success(self) -> bool:
        """Solution is successful if it resolves the issue and maintains passing tests."""
        return self.resolved and self.maintained
    
    @property
    def status(self) -> Literal["resolved", "partial", "failed"]:
        """Get human-readable status."""
        if self.success:
            return "resolved"
        elif self.resolved and not self.maintained:
            return "partial"
        else:
            return "failed"


def get_diff(repo_path: Path) -> str:
    """Get the git diff of current changes in the repository.
    
    Args:
        repo_path: Path to the repository
    
    Returns:
        Git diff as a string
    """
    try:
        result = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout
    except Exception as e:
        return f"Error getting diff: {e}"


def evaluate_with_llm(
    problem_statement: str,
    diff: str,
    gold_patch: str,
    test_results: str,
    model: str = "anthropic/claude-sonnet-4.5",
    api_key: str | None = None,
) -> LLMEvaluation:
    """Use an LLM via OpenRouter to evaluate the solution quality.
    
    Args:
        problem_statement: The original issue description
        diff: The agent's changes as a git diff
        gold_patch: The reference solution patch
        test_results: Results from running tests
        model: OpenRouter model identifier
        api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
    
    Returns:
        LLMEvaluation with assessment of the solution
    """
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return LLMEvaluation(
            correctness_score=0.0,
            reasoning="No OpenRouter API key provided",
            addresses_issue=False,
            implementation_quality="poor",
            potential_issues=["Unable to perform LLM evaluation - no API key"],
        )
    
    prompt = f"""You are evaluating a software engineering solution to a GitHub issue.

**Original Problem:**
{problem_statement}

**Agent's Solution (git diff):**
```diff
{diff[:8000]}  # Truncate to avoid token limits
```

**Reference Solution (gold patch):**
```diff
{gold_patch[:8000]}
```

**Test Results:**
{test_results[:2000]}

Please evaluate the agent's solution and provide:
1. A correctness score (0.0-1.0) - how well does it solve the issue?
2. Whether it addresses the core problem
3. Implementation quality (good/acceptable/poor)
4. Any potential issues or concerns
5. Your reasoning

Respond in JSON format:
{{
  "correctness_score": 0.85,
  "addresses_issue": true,
  "implementation_quality": "good",
  "potential_issues": ["concern1", "concern2"],
  "reasoning": "Detailed explanation..."
}}"""

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/swe-bench-english-norwegian",
            },
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "response_format": {"type": "json_object"},
            },
            timeout=60,
        )
        response.raise_for_status()
        
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        evaluation_data = json.loads(content)
        
        return LLMEvaluation(
            correctness_score=float(evaluation_data.get("correctness_score", 0.0)),
            reasoning=evaluation_data.get("reasoning", ""),
            addresses_issue=bool(evaluation_data.get("addresses_issue", False)),
            implementation_quality=evaluation_data.get("implementation_quality", "poor"),
            potential_issues=evaluation_data.get("potential_issues", []),
        )
        
    except requests.exceptions.RequestException as e:
        return LLMEvaluation(
            correctness_score=0.0,
            reasoning=f"API request failed: {e}",
            addresses_issue=False,
            implementation_quality="poor",
            potential_issues=[f"LLM evaluation failed: {e}"],
        )
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return LLMEvaluation(
            correctness_score=0.0,
            reasoning=f"Failed to parse LLM response: {e}",
            addresses_issue=False,
            implementation_quality="poor",
            potential_issues=[f"Failed to parse evaluation: {e}"],
        )


def parse_test_identifiers(test_spec: str) -> list[str]:
    """Parse test identifiers from JSON string or list format.
    
    Args:
        test_spec: JSON string like '["test1", "test2"]' or already parsed list
    
    Returns:
        List of test identifier strings
    """
    if not test_spec:
        return []
    
    # Handle already-parsed lists
    if isinstance(test_spec, list):
        return test_spec
    
    # Try to parse as JSON
    try:
        parsed = json.loads(test_spec)
        if isinstance(parsed, list):
            return [str(t) for t in parsed]
        return [str(parsed)]
    except (json.JSONDecodeError, TypeError):
        # If not valid JSON, treat as a single test identifier
        return [test_spec]


def run_tests(
    repo_path: Path,
    test_identifiers: list[str],
    timeout: int = 300,
) -> tuple[int, int, str]:
    """Run specified tests and count how many passed.
    
    Args:
        repo_path: Path to the repository
        test_identifiers: List of test identifiers (e.g., ["tests/test_foo.py::test_bar"])
        timeout: Timeout in seconds for test execution
    
    Returns:
        Tuple of (total_tests, passed_tests, output)
    """
    if not test_identifiers:
        return 0, 0, ""
    
    total = len(test_identifiers)
    
    # Try to detect test framework and run tests
    # Common patterns: pytest for Python, npm test for JS, etc.
    
    # Check if it's a Python project with pytest
    if (repo_path / "pytest.ini").exists() or (repo_path / "setup.py").exists():
        return run_pytest_tests(repo_path, test_identifiers, timeout)
    
    # Check if it's a Python project with unittest
    if (repo_path / "setup.py").exists():
        return run_unittest_tests(repo_path, test_identifiers, timeout)
    
    # Default to pytest (most common for Python projects in SWE-bench)
    return run_pytest_tests(repo_path, test_identifiers, timeout)


def run_pytest_tests(
    repo_path: Path,
    test_identifiers: list[str],
    timeout: int,
) -> tuple[int, int, str]:
    """Run tests using pytest.
    
    Args:
        repo_path: Path to the repository
        test_identifiers: List of pytest test identifiers
        timeout: Timeout in seconds
    
    Returns:
        Tuple of (total_tests, passed_tests, output)
    """
    total = len(test_identifiers)
    
    try:
        # Run pytest with verbose output and capture results
        cmd = ["python", "-m", "pytest", "-xvs"] + test_identifiers
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        output = result.stdout + result.stderr
        
        # Count passed tests from pytest output
        # pytest outputs lines like "test_foo.py::test_bar PASSED"
        passed = 0
        for line in output.split("\n"):
            if " PASSED" in line:
                passed += 1
        
        return total, passed, output
        
    except subprocess.TimeoutExpired:
        return total, 0, f"Tests timed out after {timeout}s"
    except Exception as e:
        return total, 0, f"Error running pytest: {e}"


def run_unittest_tests(
    repo_path: Path,
    test_identifiers: list[str],
    timeout: int,
) -> tuple[int, int, str]:
    """Run tests using Python's unittest.
    
    Args:
        repo_path: Path to the repository
        test_identifiers: List of unittest test identifiers
        timeout: Timeout in seconds
    
    Returns:
        Tuple of (total_tests, passed_tests, output)
    """
    total = len(test_identifiers)
    
    try:
        # Convert pytest-style identifiers to unittest format
        # e.g., "tests/test_foo.py::TestClass::test_method" -> "tests.test_foo.TestClass.test_method"
        unittest_ids = []
        for tid in test_identifiers:
            # Remove .py extension and convert path separators
            tid = tid.replace(".py", "").replace("/", ".")
            # Convert :: to .
            tid = tid.replace("::", ".")
            unittest_ids.append(tid)
        
        cmd = ["python", "-m", "unittest", "-v"] + unittest_ids
        result = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        
        output = result.stdout + result.stderr
        
        # Count passed tests from unittest output
        # unittest outputs "test_foo (module.TestClass) ... ok"
        passed = 0
        for line in output.split("\n"):
            if " ... ok" in line:
                passed += 1
        
        return total, passed, output
        
    except subprocess.TimeoutExpired:
        return total, 0, f"Tests timed out after {timeout}s"
    except Exception as e:
        return total, 0, f"Error running unittest: {e}"


def apply_patch(repo_path: Path, patch_content: str) -> bool:
    """Apply a git patch to the repository.
    
    Args:
        repo_path: Path to the repository
        patch_content: Content of the patch file
    
    Returns:
        True if patch applied successfully, False otherwise
    """
    if not patch_content or not patch_content.strip():
        return False
    
    try:
        # Write patch to a temporary file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
            f.write(patch_content)
            patch_file = Path(f.name)
        
        try:
            # Apply the patch
            result = subprocess.run(
                ["git", "apply", "--whitespace=fix", str(patch_file)],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=False,
            )
            
            return result.returncode == 0
            
        finally:
            # Clean up temp file
            patch_file.unlink()
            
    except Exception:
        return False


def evaluate_solution(
    repo_path: Path,
    entry: dict,
    test_timeout: int = 300,
    use_llm: bool = True,
    llm_model: str = "anthropic/claude-sonnet-4.5",
) -> EvaluationResult:
    """Evaluate an agent's solution against the test suite and optionally with LLM.
    
    This function assumes the agent has already made changes to the repository.
    It will:
    1. Get the diff of changes made by the agent
    2. Apply the test_patch from the dataset (adds the test cases)
    3. Run FAIL_TO_PASS tests (should now pass if solution is correct)
    4. Run PASS_TO_PASS tests (should still pass)
    5. Optionally use an LLM to assess solution quality
    
    Args:
        repo_path: Path to the repository with the agent's changes applied
        entry: Dataset entry containing test specifications
        test_timeout: Timeout for running tests
        use_llm: Whether to use LLM evaluation
        llm_model: OpenRouter model to use for evaluation
    
    Returns:
        EvaluationResult with test results and success status
    """
    # Get the diff before applying test patch
    agent_diff = get_diff(repo_path)
    
    # Parse test specifications
    fail_to_pass = parse_test_identifiers(entry.get("FAIL_TO_PASS", ""))
    pass_to_pass = parse_test_identifiers(entry.get("PASS_TO_PASS", ""))
    test_patch = entry.get("test_patch", "")
    gold_patch = entry.get("patch", "")
    problem_statement = entry.get("problem_statement", "")
    
    # Apply the test patch if present
    if test_patch:
        if not apply_patch(repo_path, test_patch):
            return EvaluationResult(
                resolved=False,
                maintained=False,
                fail_to_pass_total=len(fail_to_pass),
                fail_to_pass_passed=0,
                pass_to_pass_total=len(pass_to_pass),
                pass_to_pass_passed=0,
                error="Failed to apply test patch",
            )
    
    # Run FAIL_TO_PASS tests
    f2p_total, f2p_passed, f2p_output = run_tests(repo_path, fail_to_pass, test_timeout)
    
    # Run PASS_TO_PASS tests
    p2p_total, p2p_passed, p2p_output = run_tests(repo_path, pass_to_pass, test_timeout)
    
    # Combine outputs
    full_output = f"=== FAIL_TO_PASS Tests ===\n{f2p_output}\n\n=== PASS_TO_PASS Tests ===\n{p2p_output}"
    
    # Determine if solution is successful
    resolved = f2p_passed == f2p_total if f2p_total > 0 else True
    maintained = p2p_passed == p2p_total if p2p_total > 0 else True
    
    # Perform LLM evaluation if requested
    llm_eval = None
    if use_llm and agent_diff:
        try:
            llm_eval = evaluate_with_llm(
                problem_statement=problem_statement,
                diff=agent_diff,
                gold_patch=gold_patch,
                test_results=full_output,
                model=llm_model,
            )
        except Exception as e:
            # Don't fail the entire evaluation if LLM eval fails
            llm_eval = LLMEvaluation(
                correctness_score=0.0,
                reasoning=f"LLM evaluation failed: {e}",
                addresses_issue=False,
                implementation_quality="poor",
                potential_issues=[str(e)],
            )
    
    return EvaluationResult(
        resolved=resolved,
        maintained=maintained,
        fail_to_pass_total=f2p_total,
        fail_to_pass_passed=f2p_passed,
        pass_to_pass_total=p2p_total,
        pass_to_pass_passed=p2p_passed,
        test_output=full_output,
        llm_evaluation=llm_eval,
    )
