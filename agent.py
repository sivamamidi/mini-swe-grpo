"""
Agent — The LLM-powered code fixer.

This module connects our CodeFixEnv to a local LLM via Ollama.
The agent:
  1. Reads the bug description + buggy code from the environment
  2. Sends it to the LLM as a prompt
  3. Parses the LLM's response to extract the fixed code
  4. Submits it back to the environment

This is our mini version of DeepSWE's inference pipeline.
The real system uses vLLM to serve a 32B model and has a multi-turn
scaffold (grep, edit, run tests). Ours is single-turn: one prompt, one fix.
"""

import json
import re
import time
import urllib.request
from env import CodeFixEnv
from puzzles import PUZZLES
from puzzles_hard import PUZZLES_HARD


OLLAMA_URL = "http://localhost:11434/api/generate"


def query_ollama(prompt: str, model: str = "qwen2.5-coder:1.5b", temperature: float = 0.0) -> str:
    """
    Send a prompt to Ollama and get the response.

    Args:
        prompt: The text prompt to send.
        model: Which Ollama model to use.
        temperature: 0.0 = deterministic, higher = more random.
                     For baseline we use 0.0 (greedy).
                     For GRPO training we'll use ~0.7 (need diversity).
    """
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,       # Get the full response at once
        "options": {
            "temperature": temperature,
            "num_predict": 512, # Max tokens — our fixes are short
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["response"]
    except Exception as e:
        return f"ERROR: {e}"


def extract_code(response: str) -> str:
    """
    Extract Python code from the LLM's response.

    LLMs love wrapping code in ```python ... ``` blocks,
    adding explanations, etc. We need to strip all that.
    """
    # Try to find code in ```python ... ``` blocks
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Try ``` ... ``` blocks (no language tag)
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # If no code blocks, look for a def statement
    lines = response.strip().split("\n")
    code_lines = []
    in_function = False
    for line in lines:
        if line.strip().startswith("def "):
            in_function = True
        if in_function:
            # Stop at empty line after function or non-indented non-def line
            if code_lines and line.strip() == "" and not line.startswith(" "):
                break
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines).strip()

    # Last resort: return the whole response stripped
    return response.strip()


def run_agent_on_puzzle(env: CodeFixEnv, puzzle_id: str, model: str = "qwen2.5-coder:1.5b",
                        temperature: float = 0.0, verbose: bool = True) -> dict:
    """
    Run the agent on a single puzzle. One complete episode.

    Returns a dict with all the details (for later analysis).
    """
    # Step 1: Reset environment — get the problem
    obs = env.reset(puzzle_id=puzzle_id)

    # Step 2: Build the prompt
    prompt = env.get_prompt()

    if verbose:
        print(f"\n{'='*60}")
        print(f"PUZZLE: {obs['id']} (difficulty {obs['difficulty']})")
        print(f"{'='*60}")
        print(f"Description: {obs['description']}")
        print(f"Buggy code:\n  {obs['buggy_code']}")
        print(f"\nQuerying {model}...")

    # Step 3: Query the LLM
    start_time = time.time()
    raw_response = query_ollama(prompt, model=model, temperature=temperature)
    elapsed = time.time() - start_time

    # Step 4: Extract code from response
    fixed_code = extract_code(raw_response)

    if verbose:
        print(f"Response ({elapsed:.1f}s):\n  {fixed_code}")

    # Step 5: Submit to environment — get reward
    reward, info = env.step(fixed_code)

    if verbose:
        status = "SOLVED" if info["solved"] else "FAILED"
        print(f"\nResult: {status}  (reward={reward}, tests={info['passed']}/{info['total']})")
        if info["errors"]:
            for err in info["errors"][:3]:
                print(f"  {err}")

    return {
        "puzzle_id": puzzle_id,
        "difficulty": obs["difficulty"],
        "prompt": prompt,
        "raw_response": raw_response,
        "fixed_code": fixed_code,
        "reward": reward,
        "passed": info["passed"],
        "total": info["total"],
        "solved": info["solved"],
        "errors": info["errors"],
        "time": elapsed,
        "model": model,
        "temperature": temperature,
    }


def run_baseline(model: str = "qwen2.5-coder:1.5b", hard_only: bool = False,
                  verbose: bool = True) -> list[dict]:
    """
    Run the agent on ALL puzzles. This is our baseline — before any RL training.

    The solve rate here is what we're trying to improve with RL.
    """
    puzzle_set = PUZZLES_HARD if hard_only else PUZZLES + PUZZLES_HARD
    env = CodeFixEnv(hard_only=hard_only) if hard_only else CodeFixEnv()
    results = []

    label = "HARD PUZZLES" if hard_only else "ALL PUZZLES"
    print(f"\n{'#'*60}")
    print(f"  BASELINE EVALUATION: {model}")
    print(f"  {len(puzzle_set)} {label}, temperature=0.0 (greedy)")
    print(f"{'#'*60}")

    for puzzle in puzzle_set:
        result = run_agent_on_puzzle(env, puzzle["id"], model=model, verbose=verbose)
        results.append(result)

    # Summary
    solved = sum(1 for r in results if r["solved"])
    total = len(results)
    total_time = sum(r["time"] for r in results)

    print(f"\n{'='*60}")
    print(f"  BASELINE RESULTS")
    print(f"{'='*60}")
    print(f"  Solve rate: {solved}/{total} ({100*solved/total:.1f}%)")
    print(f"  Total time: {total_time:.1f}s")
    print()

    # By difficulty
    for diff in [1, 2, 3]:
        diff_results = [r for r in results if r["difficulty"] == diff]
        diff_solved = sum(1 for r in diff_results if r["solved"])
        print(f"  Difficulty {diff}: {diff_solved}/{len(diff_results)} solved")

    print()
    print("  Detailed results:")
    for r in results:
        status = "PASS" if r["solved"] else "FAIL"
        print(f"    {r['puzzle_id']:25s} diff={r['difficulty']}  {status}  ({r['passed']}/{r['total']} tests, {r['time']:.1f}s)")

    return results


if __name__ == "__main__":
    import sys
    hard_only = "--hard" in sys.argv
    results = run_baseline(hard_only=hard_only)
