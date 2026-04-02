"""
CodeFixEnv — A minimal RL environment for learning to fix buggy code.

This is our mini version of R2E-Gym. The real R2E-Gym uses Docker containers
to isolate full repositories (Django, scikit-learn, etc.) and runs their entire
test suites. Ours uses simple Python functions + assert statements.

The RL loop is the same:
  1. env.reset()    → Agent sees a buggy function + description
  2. Agent writes a fix
  3. env.step(fix)  → Environment runs tests, returns reward (0 or 1)
  4. Repeat

Key concepts:
  - BINARY REWARD: 1.0 if ALL tests pass, 0.0 otherwise.
    This is what DeepSWE uses. Simple but effective.
  - EPISODE: One attempt at fixing one bug. Reset → Step → Done.
  - ROLLOUT: Multiple episodes on the same bug (for GRPO, we need several
    attempts to compare which ones were better).
"""

import traceback
from typing import Optional
from puzzles import PUZZLES
from puzzles_hard import PUZZLES_HARD
from puzzles_medium import PUZZLES_MEDIUM

ALL_PUZZLES = PUZZLES + PUZZLES_HARD + PUZZLES_MEDIUM


class CodeFixEnv:
    """
    Gym-style environment for code fixing.

    Usage:
        env = CodeFixEnv()
        obs = env.reset()                    # Get a problem
        print(obs["description"])            # Read the bug report
        print(obs["buggy_code"])             # See the broken code
        reward, info = env.step(fixed_code)  # Submit your fix
        print(f"Reward: {reward}")           # 1.0 = solved, 0.0 = failed
    """

    def __init__(self, puzzle_ids: Optional[list[str]] = None, difficulty: Optional[int] = None,
                 hard_only: bool = False):
        """
        Args:
            puzzle_ids: If provided, only use these specific puzzles.
            difficulty: If provided, only use puzzles of this difficulty level.
            hard_only: If True, only use hard puzzles.
        """
        if hard_only:
            self.puzzles = PUZZLES_HARD
        else:
            self.puzzles = ALL_PUZZLES

        if puzzle_ids is not None:
            self.puzzles = [p for p in self.puzzles if p["id"] in puzzle_ids]
        if difficulty is not None:
            self.puzzles = [p for p in self.puzzles if p["difficulty"] == difficulty]

        if not self.puzzles:
            raise ValueError("No puzzles match the given filters!")

        self.current_puzzle = None
        self.current_index = 0

    def reset(self, puzzle_id: Optional[str] = None) -> dict:
        """
        Start a new episode.

        Returns an observation dict that the agent will use to generate a fix.
        This is analogous to R2E-Gym's env.reset() which spins up a Docker
        container and presents the GitHub issue to the agent.
        """
        if puzzle_id:
            matches = [p for p in self.puzzles if p["id"] == puzzle_id]
            if not matches:
                raise ValueError(f"Puzzle '{puzzle_id}' not found")
            self.current_puzzle = matches[0]
        else:
            self.current_puzzle = self.puzzles[self.current_index % len(self.puzzles)]
            self.current_index += 1

        return {
            "id": self.current_puzzle["id"],
            "description": self.current_puzzle["description"],
            "buggy_code": self.current_puzzle["buggy_code"],
            "difficulty": self.current_puzzle["difficulty"],
        }

    def step(self, fixed_code: str) -> tuple[float, dict]:
        """
        Submit a fix and get the reward.

        This is analogous to R2E-Gym's env.step() + env.compute_reward().
        In the real system, this runs the repository's test suite inside Docker.
        Here, we exec() the fixed code and run the assert statements.

        Args:
            fixed_code: The agent's proposed fix (a complete function definition).

        Returns:
            reward: 1.0 if ALL tests pass, 0.0 otherwise (binary reward).
            info: Dict with details about what happened:
                - passed: number of tests that passed
                - total: total number of tests
                - errors: list of error messages for failed tests
                - solved: boolean, True if reward == 1.0
        """
        if self.current_puzzle is None:
            raise RuntimeError("Call reset() before step()!")

        tests = self.current_puzzle["tests"]
        passed = 0
        errors = []

        for test in tests:
            try:
                # Create a clean namespace, exec the fixed code, then the test
                namespace = {}
                exec(fixed_code, namespace)
                exec(test, namespace)
                passed += 1
            except AssertionError as e:
                errors.append(f"FAIL: {test} — {e}")
            except Exception as e:
                errors.append(f"ERROR: {test} — {type(e).__name__}: {e}")

        total = len(tests)
        solved = passed == total
        reward = 1.0 if solved else 0.0

        info = {
            "passed": passed,
            "total": total,
            "errors": errors,
            "solved": solved,
            "puzzle_id": self.current_puzzle["id"],
        }

        return reward, info

    def get_prompt(self) -> str:
        """
        Format the current puzzle as a prompt for the LLM.

        This is what we'll send to the model. It mirrors what SWE agents see:
        a description of the problem + the buggy code.
        """
        if self.current_puzzle is None:
            raise RuntimeError("Call reset() first!")

        return f"""Fix the buggy Python function below.

PROBLEM: {self.current_puzzle["description"]}

BUGGY CODE:
```python
{self.current_puzzle["buggy_code"]}
```

Return ONLY the corrected Python function. No explanation, no tests, no markdown — just the function code."""

    def __len__(self):
        return len(self.puzzles)

    def __repr__(self):
        return f"CodeFixEnv({len(self.puzzles)} puzzles)"


# ── Quick test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    env = CodeFixEnv()
    print(f"Environment: {env}")
    print(f"Total puzzles: {len(env)}")
    print()

    # Demo: solve one puzzle manually
    obs = env.reset(puzzle_id="fix_add")
    print(f"Puzzle: {obs['id']} (difficulty {obs['difficulty']})")
    print(f"Description: {obs['description']}")
    print(f"Buggy code:\n{obs['buggy_code']}")
    print()

    # The correct fix
    fix = "def add(a, b):\n    return a + b"
    reward, info = env.step(fix)
    print(f"Submitted fix: {fix}")
    print(f"Reward: {reward}")
    print(f"Tests passed: {info['passed']}/{info['total']}")
    print(f"Solved: {info['solved']}")
    print()

    # Now try a wrong fix
    bad_fix = "def add(a, b):\n    return a * b"
    reward, info = env.step(bad_fix)
    print(f"Submitted bad fix: {bad_fix}")
    print(f"Reward: {reward}")
    print(f"Tests passed: {info['passed']}/{info['total']}")
    if info['errors']:
        print(f"Errors: {info['errors'][0]}")

    # Run all puzzles with their buggy code (expect all to fail)
    print("\n--- Testing all puzzles with their buggy code (should all fail) ---")
    for puzzle in PUZZLES:
        obs = env.reset(puzzle_id=puzzle["id"])
        reward, info = env.step(obs["buggy_code"])
        status = "PASS (unexpected!)" if reward == 1.0 else f"FAIL ({info['passed']}/{info['total']} tests)"
        print(f"  {puzzle['id']:25s} difficulty={puzzle['difficulty']}  {status}")
