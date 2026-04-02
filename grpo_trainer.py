"""
GRPO Trainer — The RL training loop that makes the model better at fixing code.

This implements the full GRPO (Group Relative Policy Optimization) algorithm:

  1. For each puzzle, generate G completions (rollouts) from the current policy
  2. Score each rollout: reward = 1.0 if tests pass, 0.0 otherwise
  3. Compute advantages: advantage_i = reward_i - mean(rewards in group)
  4. Update the policy: increase probability of high-advantage completions,
     decrease probability of low-advantage completions
  5. Add KL penalty to prevent the model from drifting too far from the original

This is the SAME algorithm DeepSWE uses (via rLLM), simplified for clarity.

Requirements:
  - torch
  - transformers (HuggingFace)
  - A GPU for real training (CPU works for testing but is very slow)

On your Mac (M4 Pro), this will run on MPS (Apple GPU) for small experiments.
On cloud (A100), this trains for real.
"""

import json
import os
import time
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local imports
from env import CodeFixEnv
from puzzles_hard import PUZZLES_HARD


# ── Configuration ─────────────────────────────────────────────────────

class GRPOConfig:
    """All hyperparameters in one place."""

    # Model
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    # GRPO
    group_size: int = 4          # Rollouts per puzzle (DeepSWE uses 16, we use 4 for speed)
    temperature: float = 0.8     # Sampling temperature for diversity
    max_new_tokens: int = 256    # Max tokens per completion

    # Training
    learning_rate: float = 1e-6  # Small LR — we're fine-tuning, not training from scratch
    num_epochs: int = 5          # How many passes over the puzzle set
    kl_coef: float = 0.01        # KL penalty coefficient (prevents catastrophic forgetting)
    max_grad_norm: float = 1.0   # Gradient clipping (stability)

    # Device
    device: str = "auto"         # "auto", "cuda", "mps", "cpu"

    def get_device(self):
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


# ── Core GRPO Trainer ─────────────────────────────────────────────────

class GRPOTrainer:
    """
    The GRPO training loop.

    This is the simplified version of what rLLM's AgentPPOTrainer does.
    The key difference: rLLM handles multi-turn trajectories (agent takes
    multiple actions). Ours is single-turn (one prompt → one response).
    The RL math is identical.
    """

    def __init__(self, config: GRPOConfig = None):
        self.config = config or GRPOConfig()
        self.device = self.config.get_device()

        print(f"Device: {self.device}")
        print(f"Loading model: {self.config.model_name}")

        # Load the model we're training (the "policy")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)

        # Keep a frozen copy for KL penalty (the "reference model")
        # KL penalty = how far our updated model has drifted from the original.
        # This prevents the model from "forgetting" everything it knew.
        print("Loading reference model (frozen copy for KL penalty)...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
        )

        # Environment
        self.env = CodeFixEnv(hard_only=True)

        # Logging
        self.history = []

        print(f"Model loaded. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Config: group_size={self.config.group_size}, lr={self.config.learning_rate}, "
              f"kl_coef={self.config.kl_coef}")

    def generate_rollouts(self, prompt: str) -> list[dict]:
        """
        Generate G completions for a single prompt.

        Uses temperature sampling for diversity — we need some completions
        to succeed and some to fail to create a learning signal.
        """
        self.model.eval()
        rollouts = []

        # Tokenize the prompt once
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
        prompt_len = inputs["input_ids"].shape[1]

        for _ in range(self.config.group_size):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Extract only the generated tokens (not the prompt)
            generated_ids = outputs[0][prompt_len:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            rollouts.append({
                "response": response,
                "input_ids": outputs[0],          # Full sequence (prompt + response)
                "prompt_len": prompt_len,
                "generated_ids": generated_ids,
            })

        return rollouts

    def compute_log_probs(self, model, input_ids, prompt_len):
        """
        Compute log P(response | prompt) under a given model.

        We only care about the response tokens — the prompt tokens are
        "masked" (not included in the loss). This is the same masking
        strategy DeepSWE uses to only train on agent actions.
        """
        with torch.set_grad_enabled(model.training):
            outputs = model(input_ids=input_ids.unsqueeze(0))
            logits = outputs.logits[0]  # [seq_len, vocab_size]

        # Shift: logits[t] predicts token[t+1]
        # We want log_prob of each response token given everything before it
        response_logits = logits[prompt_len - 1:-1]    # Logits predicting response tokens
        response_tokens = input_ids[prompt_len:]        # Actual response tokens

        log_probs = F.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(1, response_tokens.unsqueeze(1)).squeeze(1)

        return token_log_probs  # [response_len]

    def grpo_step(self, puzzle: dict) -> dict:
        """
        One GRPO update step for a single puzzle.

        This is the core algorithm:
        1. Generate G rollouts
        2. Score them (binary reward)
        3. Compute advantages (group-relative)
        4. Policy gradient update with KL penalty
        """
        # Step 1: Get the prompt
        obs = self.env.reset(puzzle_id=puzzle["id"])
        prompt = self.env.get_prompt()

        # Step 2: Generate rollouts
        rollouts = self.generate_rollouts(prompt)

        # Step 3: Score each rollout
        rewards = []
        for r in rollouts:
            # Extract code from model response
            from agent import extract_code
            fixed_code = extract_code(r["response"])
            reward, info = self.env.step(fixed_code)
            r["reward"] = reward
            r["solved"] = info["solved"]
            r["fixed_code"] = fixed_code
            rewards.append(reward)

        # Step 4: Compute group-relative advantages
        mean_reward = sum(rewards) / len(rewards)
        std_reward = (sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)) ** 0.5

        if std_reward < 1e-8:
            # All same reward — skip this puzzle (no learning signal)
            return {
                "puzzle_id": puzzle["id"],
                "mean_reward": mean_reward,
                "num_solved": sum(1 for r in rewards if r == 1.0),
                "loss": 0.0,
                "kl": 0.0,
                "skipped": True,
            }

        advantages = [(r - mean_reward) / std_reward for r in rewards]

        # Step 5: Policy gradient with KL penalty
        self.model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        total_kl = torch.tensor(0.0, device=self.device)
        valid_rollouts = 0

        for rollout, advantage in zip(rollouts, advantages):
            input_ids = rollout["input_ids"].to(self.device)
            prompt_len = rollout["prompt_len"]

            # Skip if no response tokens
            if len(input_ids) <= prompt_len:
                continue

            # Log probs under current policy
            policy_log_probs = self.compute_log_probs(
                self.model, input_ids, prompt_len
            )

            # Log probs under reference model (for KL penalty)
            with torch.no_grad():
                ref_log_probs = self.compute_log_probs(
                    self.ref_model, input_ids, prompt_len
                )

            # KL divergence: how far we've drifted from the reference
            kl = (ref_log_probs.exp() * (ref_log_probs - policy_log_probs)).sum()

            # Policy gradient loss:
            # L = -advantage * sum(log_prob(response)) + kl_coef * KL
            #
            # If advantage > 0 (good response): minimize -advantage * log_prob
            #   → maximize log_prob → make this response MORE likely
            # If advantage < 0 (bad response): minimize -advantage * log_prob
            #   → minimize log_prob → make this response LESS likely
            pg_loss = -advantage * policy_log_probs.sum()
            kl_loss = self.config.kl_coef * kl

            total_loss = total_loss + pg_loss + kl_loss
            total_kl = total_kl + kl.detach()
            valid_rollouts += 1

        if valid_rollouts > 0:
            total_loss = total_loss / valid_rollouts

            # Backward pass + gradient update
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )
            self.optimizer.step()

        return {
            "puzzle_id": puzzle["id"],
            "mean_reward": mean_reward,
            "num_solved": sum(1 for r in rewards if r == 1.0),
            "loss": total_loss.item() if valid_rollouts > 0 else 0.0,
            "kl": (total_kl / max(valid_rollouts, 1)).item(),
            "skipped": False,
            "advantages": advantages,
            "rewards": rewards,
        }

    def train(self, puzzles=None, num_epochs=None):
        """
        Full training loop: multiple epochs over all puzzles.

        Each epoch:
          - Iterate over all puzzles
          - For each puzzle, do one GRPO step (generate + score + update)
          - Log metrics
        """
        if puzzles is None:
            puzzles = PUZZLES_HARD
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        print(f"\n{'#'*60}")
        print(f"  GRPO TRAINING")
        print(f"  {len(puzzles)} puzzles × {num_epochs} epochs")
        print(f"  {self.config.group_size} rollouts per puzzle per epoch")
        print(f"{'#'*60}\n")

        for epoch in range(num_epochs):
            epoch_start = time.time()
            epoch_rewards = []
            epoch_losses = []
            epoch_kls = []
            epoch_solved = 0
            epoch_total = 0

            print(f"── Epoch {epoch+1}/{num_epochs} ──")

            for p_idx, puzzle in enumerate(puzzles):
                step_result = self.grpo_step(puzzle)

                epoch_rewards.append(step_result["mean_reward"])
                epoch_losses.append(step_result["loss"])
                epoch_kls.append(step_result["kl"])
                epoch_solved += step_result["num_solved"]
                epoch_total += self.config.group_size

                status = "SKIP" if step_result["skipped"] else \
                         f"reward={step_result['mean_reward']:.3f} loss={step_result['loss']:.4f}"
                print(f"  [{p_idx+1}/{len(puzzles)}] {puzzle['id']:25s} "
                      f"solved={step_result['num_solved']}/{self.config.group_size}  {status}")

            # Epoch summary
            elapsed = time.time() - epoch_start
            avg_reward = sum(epoch_rewards) / len(epoch_rewards)
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_kl = sum(epoch_kls) / len(epoch_kls)
            solve_rate = epoch_solved / epoch_total

            epoch_log = {
                "epoch": epoch + 1,
                "avg_reward": avg_reward,
                "solve_rate": solve_rate,
                "avg_loss": avg_loss,
                "avg_kl": avg_kl,
                "time": elapsed,
                "solved": epoch_solved,
                "total": epoch_total,
            }
            self.history.append(epoch_log)

            print(f"\n  Epoch {epoch+1} summary:")
            print(f"    Solve rate: {epoch_solved}/{epoch_total} ({100*solve_rate:.1f}%)")
            print(f"    Avg reward: {avg_reward:.3f}")
            print(f"    Avg loss:   {avg_loss:.4f}")
            print(f"    Avg KL:     {avg_kl:.4f}")
            print(f"    Time:       {elapsed:.1f}s")
            print()

        return self.history

    def save(self, path: str = "checkpoints/grpo_trained"):
        """Save the trained model."""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        with open(os.path.join(path, "training_history.json"), "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Model saved to {path}")

    def evaluate(self, puzzles=None) -> dict:
        """Run greedy evaluation (temperature=0) on all puzzles."""
        if puzzles is None:
            puzzles = PUZZLES_HARD

        self.model.eval()
        solved = 0

        print(f"\n{'='*60}")
        print(f"  EVALUATION (greedy, temperature=0)")
        print(f"{'='*60}")

        for puzzle in puzzles:
            obs = self.env.reset(puzzle_id=puzzle["id"])
            prompt = self.env.get_prompt()

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            from agent import extract_code
            fixed_code = extract_code(response)
            reward, info = self.env.step(fixed_code)

            status = "PASS" if info["solved"] else "FAIL"
            if info["solved"]:
                solved += 1
            print(f"  {puzzle['id']:25s} {status}  ({info['passed']}/{info['total']} tests)")

        total = len(puzzles)
        print(f"\n  Result: {solved}/{total} ({100*solved/total:.1f}%)")
        return {"solved": solved, "total": total, "solve_rate": solved / total}


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    config = GRPOConfig()

    # Allow quick test with fewer puzzles
    if "--quick" in sys.argv:
        config.num_epochs = 2
        config.group_size = 2
        puzzles = PUZZLES_HARD[:3]  # Just 3 puzzles
        print("Quick mode: 3 puzzles, 2 epochs, group_size=2")
    elif "--mixed" in sys.argv:
        # Focus on puzzles where the model has mixed results (best learning signal)
        # These are the ones from our rollout analysis that weren't all-pass or all-fail
        mixed_ids = [
            "hard_scope_bug",       # 2/8 solved — closure bug, very hard
            "hard_spiral_order",    # 4/8 solved — algorithmic
            "hard_lru_cache",       # 5/8 solved — data structure
            "hard_merge_intervals", # 5/8 solved — sorting edge case
            "hard_balanced_parens", # 0/8 solved — but close (4/5 tests each time)
            "hard_graph_cycle",     # 2/8 solved — needs rec_stack understanding
            "hard_eval_rpn",        # 5/8 solved — integer division edge case
            "hard_knapsack",        # 6/8 solved — DP
        ]
        puzzles = [p for p in PUZZLES_HARD if p["id"] in mixed_ids]
        config.num_epochs = 3
        config.group_size = 4
        print(f"Mixed mode: {len(puzzles)} challenging puzzles, 3 epochs, group_size=4")
    else:
        puzzles = None  # All puzzles

    trainer = GRPOTrainer(config)

    # Evaluate before training
    print("\n=== BEFORE TRAINING ===")
    before = trainer.evaluate(puzzles)

    # Train
    history = trainer.train(puzzles)

    # Evaluate after training
    print("\n=== AFTER TRAINING ===")
    after = trainer.evaluate(puzzles)

    # Summary
    print(f"\n{'#'*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'#'*60}")
    print(f"  Before: {before['solved']}/{before['total']} ({100*before['solve_rate']:.1f}%)")
    print(f"  After:  {after['solved']}/{after['total']} ({100*after['solve_rate']:.1f}%)")
    change = after['solve_rate'] - before['solve_rate']
    print(f"  Change: {100*change:+.1f}%")

    # Save
    trainer.save()
