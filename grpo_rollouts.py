"""
GRPO Rollout Collection — "Try many times, see what works"

This is Phase 1 of the GRPO algorithm:
  For each puzzle, generate G attempts (a "group") with temperature > 0.
  Record which ones solved the puzzle and which didn't.

This data becomes the training signal for Phase 2 (gradient updates).

Key concepts:
  - GROUP SIZE (G): How many attempts per puzzle. DeepSWE uses 16.
    We'll use 8 to keep it manageable locally.
  - TEMPERATURE: Controls randomness. 0.0 = always same answer.
    0.7-1.0 = diverse attempts. We NEED diversity so some attempts
    succeed and some fail — that's what creates the learning signal.
  - ADVANTAGE: How much better/worse an attempt is vs the group average.
    If 3/8 succeed (avg reward = 0.375), a success gets advantage
    = 1.0 - 0.375 = +0.625, a failure gets = 0.0 - 0.375 = -0.375.
"""

import json
import os
import time
from agent import query_ollama, extract_code
from env import CodeFixEnv
from puzzles_hard import PUZZLES_HARD


def collect_rollouts(
    puzzles=None,
    group_size: int = 8,
    temperature: float = 0.8,
    model: str = "qwen2.5-coder:1.5b",
    verbose: bool = True,
) -> list[dict]:
    """
    Collect GRPO rollouts: multiple attempts per puzzle.

    For each puzzle, generate `group_size` attempts with temperature > 0.
    This creates the training data for GRPO.

    Returns a list of "group" dicts, each containing:
      - puzzle_id: which puzzle
      - prompt: the prompt sent to the model
      - rollouts: list of (response, reward) pairs
      - mean_reward: average reward in the group
      - advantages: per-rollout advantage (reward - mean)
    """
    if puzzles is None:
        puzzles = PUZZLES_HARD

    env = CodeFixEnv(hard_only=True)
    groups = []

    for p_idx, puzzle in enumerate(puzzles):
        obs = env.reset(puzzle_id=puzzle["id"])
        prompt = env.get_prompt()

        if verbose:
            print(f"\n{'='*60}")
            print(f"[{p_idx+1}/{len(puzzles)}] Puzzle: {puzzle['id']} (difficulty {puzzle['difficulty']})")
            print(f"  Generating {group_size} rollouts (temp={temperature})...")

        rollouts = []
        for g in range(group_size):
            # Query with temperature > 0 for diversity
            raw_response = query_ollama(prompt, model=model, temperature=temperature)
            fixed_code = extract_code(raw_response)
            reward, info = env.step(fixed_code)

            rollouts.append({
                "response": raw_response,
                "fixed_code": fixed_code,
                "reward": reward,
                "solved": info["solved"],
                "passed": info["passed"],
                "total": info["total"],
            })

            if verbose:
                status = "OK" if info["solved"] else "X "
                print(f"    rollout {g+1}/{group_size}: {status} (reward={reward}, "
                      f"tests={info['passed']}/{info['total']})")

        # Compute group statistics
        rewards = [r["reward"] for r in rollouts]
        mean_reward = sum(rewards) / len(rewards)

        # GRPO advantage: reward_i - mean(rewards)
        # This is the key insight: we don't need a learned value function.
        # We just compare each attempt to the group average.
        advantages = [r - mean_reward for r in rewards]

        # Normalize advantages (optional but helps stability)
        std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5
        if std_reward > 0:
            normalized_advantages = [a / std_reward for a in advantages]
        else:
            # All same reward (all solved or all failed) — no learning signal
            normalized_advantages = [0.0] * len(advantages)

        group = {
            "puzzle_id": puzzle["id"],
            "difficulty": puzzle["difficulty"],
            "prompt": prompt,
            "buggy_code": puzzle["buggy_code"],
            "rollouts": rollouts,
            "rewards": rewards,
            "mean_reward": mean_reward,
            "advantages": advantages,
            "normalized_advantages": normalized_advantages,
            "num_solved": sum(1 for r in rollouts if r["solved"]),
        }
        groups.append(group)

        if verbose:
            solved = group["num_solved"]
            print(f"  Summary: {solved}/{group_size} solved, "
                  f"mean_reward={mean_reward:.3f}")
            if solved > 0 and solved < group_size:
                print(f"  ^ MIXED GROUP — good learning signal!")
            elif solved == 0:
                print(f"  ^ ALL FAILED — no positive signal (will be skipped in training)")
            else:
                print(f"  ^ ALL SOLVED — no negative signal (limited learning)")

    return groups


def analyze_rollouts(groups: list[dict]):
    """Print a summary of collected rollouts."""
    print(f"\n{'#'*60}")
    print(f"  ROLLOUT COLLECTION SUMMARY")
    print(f"{'#'*60}")

    total_rollouts = sum(len(g["rollouts"]) for g in groups)
    total_solved = sum(g["num_solved"] for g in groups)
    group_size = len(groups[0]["rollouts"]) if groups else 0

    print(f"\n  Puzzles: {len(groups)}")
    print(f"  Group size: {group_size}")
    print(f"  Total rollouts: {total_rollouts}")
    print(f"  Total solved: {total_solved}/{total_rollouts} "
          f"({100*total_solved/total_rollouts:.1f}%)")

    # Categorize groups by learning signal quality
    mixed = [g for g in groups if 0 < g["num_solved"] < group_size]
    all_solved = [g for g in groups if g["num_solved"] == group_size]
    all_failed = [g for g in groups if g["num_solved"] == 0]

    print(f"\n  Group categories:")
    print(f"    Mixed (best for learning):  {len(mixed)} puzzles")
    print(f"    All solved (limited signal): {len(all_solved)} puzzles")
    print(f"    All failed (no signal):      {len(all_failed)} puzzles")

    print(f"\n  Per-puzzle breakdown:")
    for g in groups:
        bar = "".join(["+" if r["solved"] else "-" for r in g["rollouts"]])
        signal = "MIXED" if 0 < g["num_solved"] < group_size else \
                 "ALL OK" if g["num_solved"] == group_size else "NONE"
        print(f"    {g['puzzle_id']:25s} [{bar}] "
              f"{g['num_solved']}/{group_size} solved  "
              f"mean={g['mean_reward']:.3f}  signal={signal}")

    # Show advantage distribution for mixed groups
    if mixed:
        print(f"\n  Advantage examples (from mixed groups):")
        g = mixed[0]
        print(f"    Puzzle: {g['puzzle_id']}")
        print(f"    Mean reward: {g['mean_reward']:.3f}")
        for i, (r, adv) in enumerate(zip(g["rollouts"], g["normalized_advantages"])):
            status = "SOLVED" if r["solved"] else "FAILED"
            print(f"      Rollout {i+1}: {status}  advantage={adv:+.3f}")

    return {
        "total_rollouts": total_rollouts,
        "total_solved": total_solved,
        "mixed_groups": len(mixed),
        "all_solved_groups": len(all_solved),
        "all_failed_groups": len(all_failed),
    }


def save_rollouts(groups: list[dict], path: str = "rollouts.json"):
    """Save rollouts to disk for training."""
    with open(path, "w") as f:
        json.dump(groups, f, indent=2)
    print(f"\nSaved {len(groups)} groups to {path}")


if __name__ == "__main__":
    import sys

    group_size = 8
    if "--group-size" in sys.argv:
        idx = sys.argv.index("--group-size")
        group_size = int(sys.argv[idx + 1])

    print("Collecting GRPO rollouts...")
    print(f"This will query the model {len(PUZZLES_HARD)} × {group_size} = "
          f"{len(PUZZLES_HARD) * group_size} times.")
    print("Each query takes ~0.5-1.5s, so expect ~1-3 minutes total.\n")

    groups = collect_rollouts(group_size=group_size)
    stats = analyze_rollouts(groups)
    save_rollouts(groups)
