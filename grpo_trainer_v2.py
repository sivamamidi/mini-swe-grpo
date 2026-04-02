"""
GRPO Trainer v2 — Improved for real local training results.

Key improvements over v1:
  1. HYBRID GENERATION: Uses Ollama for fast rollout collection,
     HuggingFace for log-prob computation and gradient updates.
     This is 5-10x faster than generating through HuggingFace on MPS.
  2. FOCUSED PUZZLE SET: Only trains on "mixed" puzzles where the
     model sometimes succeeds — these have the best learning signal.
  3. MORE EPOCHS + LARGER GROUPS: 10 epochs, group_size=8.
  4. LIVE METRICS: Shows a learning curve after training.
  5. PROPER EVALUATION: Tests on ALL puzzles (not just training set).
"""

import json
import math
import os
import time
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from env import CodeFixEnv
from puzzles_hard import PUZZLES_HARD
from puzzles_medium import PUZZLES_MEDIUM
from agent import query_ollama, extract_code


# ── Configuration ─────────────────────────────────────────────────────

class Config:
    # Model (HuggingFace for training, Ollama for generation)
    hf_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    ollama_model: str = "qwen2.5-coder:1.5b"

    # GRPO
    group_size: int = 8          # 8 rollouts per puzzle (up from 4)
    temperature: float = 0.8

    # Training
    learning_rate: float = 5e-6  # Higher LR (was 1e-6)
    num_epochs: int = 10
    kl_coef: float = 0.05        # KL penalty coefficient
    max_grad_norm: float = 1.0
    advantage_clip: float = 5.0  # Clip extreme advantages for stability
    use_ref_model: bool = True   # Set False to skip ref model (saves memory)
    max_response_tokens: int = 384

    device: str = "auto"

    def get_device(self):
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


# ── Select puzzles with good learning signal ──────────────────────────

# From our testing: these puzzles have 25-75% solve rate at temp=0.8
TRAIN_PUZZLES = [p for p in PUZZLES_HARD if p["id"] in {
    "hard_scope_bug",        # 2/8 — closure bug
    "hard_spiral_order",     # 4/8 — algorithmic
    "hard_lru_cache",        # 5/8 — data structure
    "hard_merge_intervals",  # 5/8 — sorting
    "hard_graph_cycle",      # 2/8 — rec_stack
    "hard_eval_rpn",         # 5/8 — int division
    "hard_knapsack",         # 6/8 — DP
    "hard_off_by_one",       # 7/8 — edge case
}] + [p for p in PUZZLES_MEDIUM if p["id"] in {
    "med_recursive_power",     # 3/4
    "med_case_dedup",          # 2/4
    "med_class_shared_state",  # 1/4
    "med_sliding_window_avg",  # 2/4
    "med_flatten_depth",       # 2/4
    "med_running_total",       # 1/4
    "med_nested_depth",        # 1/4
    "med_zip_truncation",      # 1/4
}]

# Eval includes everything (train + unseen)
EVAL_PUZZLES = PUZZLES_HARD + PUZZLES_MEDIUM


def format_prompt(puzzle: dict) -> str:
    return f"""Fix the buggy Python function below.

PROBLEM: {puzzle["description"]}

BUGGY CODE:
```python
{puzzle["buggy_code"]}
```

Return ONLY the corrected Python function. No explanation, no tests, no markdown — just the function code."""


# ── Trainer ───────────────────────────────────────────────────────────

class GRPOTrainerV2:
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.device = self.config.get_device()

        print(f"Device: {self.device}")
        print(f"Loading policy model: {self.config.hf_model}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.hf_model, trust_remote_code=True, padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.hf_model, dtype=torch.bfloat16, trust_remote_code=True,
        ).to(self.device)

        if self.config.use_ref_model:
            print("Loading reference model (frozen, on CPU)...")
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                self.config.hf_model, dtype=torch.bfloat16, trust_remote_code=True,
            ).eval()
            self.ref_device = torch.device("cpu")
            for p in self.ref_model.parameters():
                p.requires_grad = False
        else:
            print("Skipping reference model (no KL penalty — saves memory)")
            self.ref_model = None
            self.ref_device = None

        self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate)

        params = sum(p.numel() for p in self.model.parameters())
        print(f"Ready. {params:,} parameters. {len(TRAIN_PUZZLES)} training puzzles.")
        self.history = []

    def collect_rollouts_ollama(self, puzzle: dict) -> list[dict]:
        """Fast rollout collection via Ollama."""
        prompt = format_prompt(puzzle)
        rollouts = []

        for _ in range(self.config.group_size):
            raw = query_ollama(prompt, model=self.config.ollama_model,
                               temperature=self.config.temperature)
            code = extract_code(raw)

            # Score it
            ns = {}
            passed = 0
            total = len(puzzle["tests"])
            try:
                exec(code, ns)
                for t in puzzle["tests"]:
                    try:
                        exec(t, ns)
                        passed += 1
                    except:
                        pass
            except:
                pass

            reward = 1.0 if passed == total else 0.0
            rollouts.append({
                "response": raw,
                "fixed_code": code,
                "reward": reward,
                "solved": passed == total,
            })

        return rollouts

    def compute_log_probs(self, model, input_ids, prompt_len):
        """Log P(response | prompt) under model. Only response tokens."""
        with torch.set_grad_enabled(model.training):
            logits = model(input_ids=input_ids.unsqueeze(0)).logits[0]

        response_logits = logits[prompt_len - 1:-1]
        response_tokens = input_ids[prompt_len:]

        log_probs = F.log_softmax(response_logits, dim=-1)
        token_log_probs = log_probs.gather(1, response_tokens.unsqueeze(1)).squeeze(1)
        return token_log_probs

    def grpo_update(self, puzzle: dict, rollouts: list[dict]) -> dict:
        """Compute GRPO gradient update from pre-collected rollouts."""
        prompt = format_prompt(puzzle)
        rewards = [r["reward"] for r in rollouts]
        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5

        if std_r < 1e-8:
            return {"loss": 0.0, "kl": 0.0, "skipped": True,
                    "mean_reward": mean_r, "num_solved": sum(1 for r in rewards if r == 1.0)}

        advantages = [(r - mean_r) / std_r for r in rewards]
        # Clip extreme advantages
        advantages = [max(-self.config.advantage_clip, min(self.config.advantage_clip, a))
                      for a in advantages]

        # Accumulate gradients one rollout at a time to avoid OOM.
        # Each rollout: forward pass → compute loss → backward → free graph.
        # Then do one optimizer step at the end.
        self.model.train()
        self.optimizer.zero_grad()
        total_loss_val = 0.0
        total_kl = 0.0
        n_valid = 0

        prompt_enc = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_enc["input_ids"].shape[1]

        for rollout, adv in zip(rollouts, advantages):
            full_text = prompt + rollout["response"]
            enc = self.tokenizer(full_text, return_tensors="pt",
                                 max_length=self.config.max_response_tokens + 512,
                                 truncation=True)
            input_ids = enc["input_ids"][0].to(self.device)

            if input_ids.shape[0] <= prompt_len + 1:
                continue
            if prompt_len >= input_ids.shape[0]:
                continue

            policy_lp = self.compute_log_probs(self.model, input_ids, prompt_len)

            pg_loss = -adv * policy_lp.mean()
            kl_val = 0.0

            if self.ref_model is not None:
                with torch.no_grad():
                    ref_lp = self.compute_log_probs(
                        self.ref_model, input_ids.to(self.ref_device), prompt_len
                    ).to(self.device)
                kl = (ref_lp.exp() * (ref_lp - policy_lp)).mean()
                kl_val = kl.detach().item()
                pg_loss = pg_loss + self.config.kl_coef * kl

            rollout_loss = pg_loss

            # Backward immediately — accumulates into .grad, then frees the graph
            rollout_loss.backward()

            total_loss_val += rollout_loss.detach().item()
            total_kl += kl_val
            n_valid += 1

        if n_valid > 0:
            # Scale gradients by 1/n_valid (since we accumulated)
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad.div_(n_valid)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            self.optimizer.step()

        return {
            "loss": total_loss_val / max(n_valid, 1),
            "kl": total_kl / max(n_valid, 1),
            "skipped": False,
            "mean_reward": mean_r,
            "num_solved": sum(1 for r in rewards if r == 1.0),
        }

    def evaluate_ollama(self, puzzles: list[dict] = None) -> dict:
        """Evaluate using Ollama (greedy, temp=0)."""
        if puzzles is None:
            puzzles = EVAL_PUZZLES
        solved = 0
        results = []
        for p in puzzles:
            prompt = format_prompt(p)
            raw = query_ollama(prompt, model=self.config.ollama_model, temperature=0.0)
            code = extract_code(raw)

            ns = {}
            passed = 0
            total = len(p["tests"])
            try:
                exec(code, ns)
                for t in p["tests"]:
                    try:
                        exec(t, ns)
                        passed += 1
                    except:
                        pass
            except:
                pass

            ok = passed == total
            if ok:
                solved += 1
            results.append({"id": p["id"], "solved": ok, "passed": passed, "total": total})

        return {"solved": solved, "total": len(puzzles),
                "rate": solved / len(puzzles), "details": results}

    def evaluate_hf(self, puzzles: list[dict] = None) -> dict:
        """Evaluate using the HuggingFace model (the one being trained)."""
        if puzzles is None:
            puzzles = EVAL_PUZZLES

        self.model.eval()
        solved = 0
        results = []

        for p in puzzles:
            prompt = format_prompt(p)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_response_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )
            code = extract_code(response)

            ns = {}
            passed = 0
            total = len(p["tests"])
            try:
                exec(code, ns)
                for t in p["tests"]:
                    try:
                        exec(t, ns)
                        passed += 1
                    except:
                        pass
            except:
                pass

            ok = passed == total
            if ok:
                solved += 1
            results.append({"id": p["id"], "solved": ok, "passed": passed, "total": total})

        return {"solved": solved, "total": len(puzzles),
                "rate": solved / len(puzzles), "details": results}

    def train(self):
        """Main training loop with hybrid Ollama generation + HF gradient updates."""
        cfg = self.config
        puzzles = TRAIN_PUZZLES

        # Pre-training eval
        print(f"\n{'='*60}")
        print("  PRE-TRAINING EVALUATION (Ollama baseline)")
        print(f"{'='*60}")
        pre_eval = self.evaluate_ollama(EVAL_PUZZLES)
        print(f"  Baseline: {pre_eval['solved']}/{pre_eval['total']} "
              f"({100*pre_eval['rate']:.1f}%)")
        for r in pre_eval["details"]:
            tag = "PASS" if r["solved"] else "FAIL"
            print(f"    {r['id']:35s} {tag}  ({r['passed']}/{r['total']})")

        print(f"\n{'#'*60}")
        print(f"  GRPO TRAINING v2")
        print(f"  {len(puzzles)} training puzzles × {cfg.num_epochs} epochs × "
              f"{cfg.group_size} rollouts")
        print(f"  LR={cfg.learning_rate}, KL={cfg.kl_coef}")
        print(f"{'#'*60}\n")

        for epoch in range(cfg.num_epochs):
            t0 = time.time()
            epoch_solved = 0
            epoch_total = 0
            epoch_losses = []
            epoch_kls = []

            for pi, puzzle in enumerate(puzzles):
                # Phase 1: Collect rollouts via Ollama (fast)
                rollouts = self.collect_rollouts_ollama(puzzle)
                n_solved = sum(1 for r in rollouts if r["solved"])

                # Phase 2: Gradient update via HuggingFace
                result = self.grpo_update(puzzle, rollouts)

                epoch_solved += n_solved
                epoch_total += cfg.group_size
                epoch_losses.append(result["loss"])
                epoch_kls.append(result["kl"])

                tag = "SKIP" if result["skipped"] else f"loss={result['loss']:.2f}"
                print(f"  E{epoch+1} [{pi+1}/{len(puzzles)}] "
                      f"{puzzle['id']:30s} {n_solved}/{cfg.group_size} solved  {tag}")

            elapsed = time.time() - t0
            avg_reward = epoch_solved / epoch_total
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            avg_kl = sum(epoch_kls) / len(epoch_kls)

            log = {
                "epoch": epoch + 1,
                "solve_rate": avg_reward,
                "solved": epoch_solved,
                "total": epoch_total,
                "avg_loss": avg_loss,
                "avg_kl": avg_kl,
                "time": elapsed,
            }
            self.history.append(log)

            print(f"\n  ── Epoch {epoch+1}/{cfg.num_epochs}: "
                  f"solve_rate={100*avg_reward:.1f}% "
                  f"loss={avg_loss:.3f} kl={avg_kl:.4f} "
                  f"time={elapsed:.0f}s ──\n")

        # Post-training eval (using the HF model that was actually updated)
        print(f"\n{'='*60}")
        print("  POST-TRAINING EVALUATION (HuggingFace trained model)")
        print(f"{'='*60}")
        post_eval = self.evaluate_hf(EVAL_PUZZLES)
        print(f"  After training: {post_eval['solved']}/{post_eval['total']} "
              f"({100*post_eval['rate']:.1f}%)")
        for r in post_eval["details"]:
            tag = "PASS" if r["solved"] else "FAIL"
            print(f"    {r['id']:35s} {tag}  ({r['passed']}/{r['total']})")

        # Compare
        print(f"\n{'#'*60}")
        print(f"  RESULTS")
        print(f"{'#'*60}")
        print(f"  Before (Ollama/base): {pre_eval['solved']}/{pre_eval['total']} "
              f"({100*pre_eval['rate']:.1f}%)")
        print(f"  After  (HF/trained):  {post_eval['solved']}/{post_eval['total']} "
              f"({100*post_eval['rate']:.1f}%)")
        delta = post_eval['rate'] - pre_eval['rate']
        print(f"  Delta: {100*delta:+.1f}%")

        # Show per-puzzle changes
        print(f"\n  Per-puzzle changes:")
        for pre_r, post_r in zip(pre_eval["details"], post_eval["details"]):
            before = "PASS" if pre_r["solved"] else "FAIL"
            after = "PASS" if post_r["solved"] else "FAIL"
            if before != after:
                arrow = "FAIL->PASS" if post_r["solved"] else "PASS->FAIL"
                print(f"    {pre_r['id']:35s} {arrow}")

        # Learning curve
        print(f"\n  Learning curve (solve rate during training):")
        for h in self.history:
            bar_len = int(h["solve_rate"] * 40)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            print(f"    Epoch {h['epoch']:2d}: [{bar}] {100*h['solve_rate']:.1f}%")

        # Save
        self.save()
        return self.history

    def save(self, path: str = "checkpoints/grpo_v2"):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        with open(os.path.join(path, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"\n  Model saved to {path}")


if __name__ == "__main__":
    import sys

    config = Config()

    if "--quick" in sys.argv:
        config.num_epochs = 3
        config.group_size = 4

    if "--no-ref" in sys.argv:
        config.use_ref_model = False

    trainer = GRPOTrainerV2(config)
    trainer.train()
