"""
Generate publication-quality figures for the Mini-SWE-RL project.
Uses SciencePlots for academic styling.
"""

import json
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os

# Use science style with no-latex fallback
try:
    plt.style.use(["science", "no-latex", "grid"])
except:
    plt.style.use("seaborn-v0_8-whitegrid")

# Color palette (colorblind-friendly)
COLORS = {
    "blue": "#2166AC",
    "red": "#B2182B",
    "green": "#1B7837",
    "orange": "#E08214",
    "purple": "#762A83",
    "gray": "#666666",
    "light_blue": "#92C5DE",
    "light_red": "#FDDBC7",
}

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)


def fig1_training_curve():
    """Figure 1: GRPO Training Learning Curve."""
    with open("checkpoints/grpo_v2/history.json") as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    solve_rates = [h["solve_rate"] * 100 for h in history]
    losses = [h["avg_loss"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Solve rate
    ax1.plot(epochs, solve_rates, "o-", color=COLORS["blue"], linewidth=2,
             markersize=6, label="Training solve rate")
    # Smoothed trend
    from scipy.signal import savgol_filter
    if len(solve_rates) >= 5:
        smooth = savgol_filter(solve_rates, 5, 2)
        ax1.plot(epochs, smooth, "--", color=COLORS["light_blue"], linewidth=1.5,
                 label="Smoothed trend", alpha=0.8)
    ax1.axhline(y=66.7, color=COLORS["red"], linestyle=":", linewidth=1.5,
                label="Baseline (66.7%)", alpha=0.7)
    ax1.axhline(y=73.3, color=COLORS["green"], linestyle=":", linewidth=1.5,
                label="Final eval (73.3%)", alpha=0.7)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Solve Rate (%)")
    ax1.set_title("(a) Solve Rate During Training")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.set_ylim(40, 80)
    ax1.set_xlim(0.5, 10.5)

    # Policy loss
    ax2.plot(epochs, losses, "s-", color=COLORS["purple"], linewidth=2, markersize=6)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Policy Loss")
    ax2.set_title("(b) Average Policy Loss")
    ax2.axhline(y=0, color=COLORS["gray"], linestyle="-", linewidth=0.5, alpha=0.5)
    ax2.set_xlim(0.5, 10.5)

    fig.suptitle("GRPO Training on Qwen2.5-Coder-1.5B (Apple M4 Pro, MPS)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig1_training_curve.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/fig1_training_curve.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig1_training_curve")


def fig2_before_after():
    """Figure 2: Before vs After comparison bar chart."""
    # Pre-training results (from Ollama baseline)
    before = {
        "hard_off_by_one": True, "hard_default_mutable": True, "hard_integer_division": True,
        "hard_scope_bug": False, "hard_string_compare": True, "hard_deep_copy": True,
        "hard_spiral_order": True, "hard_lru_cache": True, "hard_merge_intervals": True,
        "hard_balanced_parens": False, "hard_longest_subseq": True, "hard_knapsack": True,
        "hard_trie_insert_search": True, "hard_graph_cycle": False, "hard_eval_rpn": False,
        "med_shallow_copy_accumulate": True, "med_generator_exhaustion": False,
        "med_float_equality": False, "med_recursive_power": True, "med_case_dedup": True,
        "med_class_shared_state": False, "med_sliding_window_avg": True,
        "med_filter_dict": True, "med_flatten_depth": False, "med_running_total": False,
        "med_merge_sorted": True, "med_interleave_strings": True, "med_counter_reset": True,
        "med_nested_depth": True, "med_zip_truncation": False,
    }

    # Post-training results (from HF trained model)
    after = {
        "hard_off_by_one": True, "hard_default_mutable": False, "hard_integer_division": True,
        "hard_scope_bug": True, "hard_string_compare": True, "hard_deep_copy": True,
        "hard_spiral_order": False, "hard_lru_cache": False, "hard_merge_intervals": True,
        "hard_balanced_parens": True, "hard_longest_subseq": True, "hard_knapsack": True,
        "hard_trie_insert_search": True, "hard_graph_cycle": True, "hard_eval_rpn": True,
        "med_shallow_copy_accumulate": False, "med_generator_exhaustion": False,
        "med_float_equality": False, "med_recursive_power": True, "med_case_dedup": True,
        "med_class_shared_state": True, "med_sliding_window_avg": True,
        "med_filter_dict": False, "med_flatten_depth": True, "med_running_total": True,
        "med_merge_sorted": True, "med_interleave_strings": True, "med_counter_reset": True,
        "med_nested_depth": True, "med_zip_truncation": True,
    }

    fig, ax = plt.subplots(figsize=(10, 4))

    categories = ["Overall", "Hard Puzzles\n(15)", "Medium Puzzles\n(15)",
                   "Newly Solved", "Regressions"]

    hard_ids = [k for k in before if k.startswith("hard_")]
    med_ids = [k for k in before if k.startswith("med_")]

    before_hard = sum(before[k] for k in hard_ids)
    after_hard = sum(after[k] for k in hard_ids)
    before_med = sum(before[k] for k in med_ids)
    after_med = sum(after[k] for k in med_ids)

    newly_solved = sum(1 for k in before if not before[k] and after[k])
    regressions = sum(1 for k in before if before[k] and not after[k])

    before_vals = [20/30*100, before_hard/15*100, before_med/15*100, 0, 0]
    after_vals = [22/30*100, after_hard/15*100, after_med/15*100, newly_solved, regressions]

    x = np.arange(3)  # Just the first 3 categories
    width = 0.35

    bars1 = ax.bar(x - width/2, before_vals[:3], width, label="Before RL",
                   color=COLORS["light_blue"], edgecolor=COLORS["blue"], linewidth=1)
    bars2 = ax.bar(x + width/2, after_vals[:3], width, label="After RL",
                   color=COLORS["blue"], edgecolor=COLORS["blue"], linewidth=1)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel("Solve Rate (%)")
    ax.set_title("Before vs After GRPO Training (Greedy Evaluation)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Overall\n(30 puzzles)", "Hard Puzzles\n(15)", "Medium Puzzles\n(15)"])
    ax.legend()
    ax.set_ylim(0, 100)

    # Add annotation box
    textstr = f"Newly solved: {newly_solved} puzzles\nRegressions: {regressions} puzzles\nNet gain: +{newly_solved - regressions}"
    props = dict(boxstyle="round", facecolor=COLORS["light_blue"], alpha=0.3)
    ax.text(0.98, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right", bbox=props)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig2_before_after.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/fig2_before_after.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig2_before_after")


def fig3_rollout_analysis():
    """Figure 3: Rollout analysis from GRPO data collection."""
    with open("rollouts.json") as f:
        groups = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # (a) Per-puzzle solve rate heatmap
    puzzle_ids = [g["puzzle_id"].replace("hard_", "").replace("med_", "") for g in groups]
    solve_rates = [g["mean_reward"] * 100 for g in groups]
    difficulties = [g["difficulty"] for g in groups]

    colors = []
    for sr in solve_rates:
        if sr == 0:
            colors.append(COLORS["red"])
        elif sr == 100:
            colors.append(COLORS["light_blue"])
        else:
            colors.append(COLORS["green"])  # Mixed = best for learning

    bars = ax1.barh(range(len(puzzle_ids)), solve_rates, color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_yticks(range(len(puzzle_ids)))
    ax1.set_yticklabels(puzzle_ids, fontsize=7)
    ax1.set_xlabel("Solve Rate (%, 8 rollouts at temp=0.8)")
    ax1.set_title("(a) Per-Puzzle Solve Rate")
    ax1.set_xlim(0, 105)
    ax1.invert_yaxis()

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["green"], label="Mixed (best signal)"),
        Patch(facecolor=COLORS["light_blue"], label="All solved"),
        Patch(facecolor=COLORS["red"], label="All failed"),
    ]
    ax1.legend(handles=legend_elements, fontsize=8, loc="lower right")

    # (b) Advantage distribution for a mixed group
    mixed_groups = [g for g in groups if 0 < g["num_solved"] < 8]
    if mixed_groups:
        # Pick the most balanced mixed group
        best_mixed = min(mixed_groups, key=lambda g: abs(g["mean_reward"] - 0.5))
        advs = best_mixed["normalized_advantages"]
        rewards = best_mixed["rewards"]

        bar_colors = [COLORS["green"] if r == 1.0 else COLORS["red"] for r in rewards]
        ax2.bar(range(len(advs)), advs, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax2.axhline(y=0, color=COLORS["gray"], linestyle="-", linewidth=1)
        ax2.set_xlabel("Rollout Index")
        ax2.set_ylabel("Normalized Advantage")
        ax2.set_title(f"(b) GRPO Advantages: '{best_mixed['puzzle_id']}'")
        ax2.set_xticks(range(len(advs)))
        ax2.set_xticklabels([f"R{i+1}" for i in range(len(advs))])

        legend_elements2 = [
            Patch(facecolor=COLORS["green"], label="Solved (positive advantage)"),
            Patch(facecolor=COLORS["red"], label="Failed (negative advantage)"),
        ]
        ax2.legend(handles=legend_elements2, fontsize=8)

    fig.suptitle("GRPO Rollout Analysis (Group Size = 8, Temperature = 0.8)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig3_rollout_analysis.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/fig3_rollout_analysis.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig3_rollout_analysis")


def fig4_architecture():
    """Figure 4: System architecture diagram."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Boxes
    box_style = dict(boxstyle="round,pad=0.4", facecolor=COLORS["light_blue"],
                     edgecolor=COLORS["blue"], linewidth=2)
    box_env = dict(boxstyle="round,pad=0.4", facecolor="#E6F5E6",
                   edgecolor=COLORS["green"], linewidth=2)
    box_rl = dict(boxstyle="round,pad=0.4", facecolor="#F5E6F5",
                  edgecolor=COLORS["purple"], linewidth=2)
    box_data = dict(boxstyle="round,pad=0.3", facecolor=COLORS["light_red"],
                    edgecolor=COLORS["orange"], linewidth=2)

    # Environment
    ax.text(1.5, 5, "CodeFixEnv\n(env.py)", ha="center", va="center",
            fontsize=11, fontweight="bold", bbox=box_env)
    ax.text(1.5, 3.5, "45 Puzzles\n(3 difficulty levels)", ha="center", va="center",
            fontsize=9, bbox=box_data)

    # Agent
    ax.text(5, 5, "LLM Agent\n(Ollama / HuggingFace)", ha="center", va="center",
            fontsize=11, fontweight="bold", bbox=box_style)
    ax.text(5, 3.5, "Qwen2.5-Coder\n1.5B parameters", ha="center", va="center",
            fontsize=9, bbox=box_data)

    # Trainer
    ax.text(8.5, 5, "GRPO Trainer\n(grpo_trainer_v2.py)", ha="center", va="center",
            fontsize=11, fontweight="bold", bbox=box_rl)
    ax.text(8.5, 3.5, "Policy Gradient\n+ Advantage Norm", ha="center", va="center",
            fontsize=9, bbox=box_data)

    # Arrows
    arrow_style = dict(arrowstyle="->,head_width=0.3", color=COLORS["gray"], linewidth=2)
    ax.annotate("", xy=(3.5, 5), xytext=(2.8, 5), arrowprops=arrow_style)
    ax.annotate("", xy=(6.8, 5), xytext=(6.2, 5), arrowprops=arrow_style)
    # Feedback loop
    ax.annotate("", xy=(5, 4.3), xytext=(1.5, 4.3),
                arrowprops=dict(arrowstyle="->", color=COLORS["green"],
                                connectionstyle="arc3,rad=-0.3", linewidth=1.5))

    # Labels on arrows
    ax.text(3.15, 5.3, "prompt", fontsize=8, ha="center", color=COLORS["gray"])
    ax.text(6.5, 5.3, "rollouts", fontsize=8, ha="center", color=COLORS["gray"])
    ax.text(3.2, 3.8, "reward (0 or 1)", fontsize=8, ha="center", color=COLORS["green"])

    # RL Loop label
    ax.text(5, 1.8, "GRPO Loop: Generate G rollouts → Score → Compute group-relative advantages → Update policy",
            ha="center", va="center", fontsize=10, style="italic",
            bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="goldenrod", linewidth=1.5))

    # Bottom text
    ax.text(5, 0.7, "Hardware: Apple M4 Pro (48GB) · MPS Backend · ~30 min training\n"
            "Same algorithm as DeepSWE (Qwen3-32B, 64×H100, 6 days) — just at miniature scale",
            ha="center", va="center", fontsize=8.5, color=COLORS["gray"])

    ax.set_title("Mini-SWE-RL: System Architecture", fontsize=14, fontweight="bold", pad=20)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig4_architecture.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/fig4_architecture.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig4_architecture")


def fig5_puzzle_changes():
    """Figure 5: Per-puzzle before/after with change annotations."""
    puzzles_order = [
        "hard_scope_bug", "hard_balanced_parens", "hard_graph_cycle", "hard_eval_rpn",
        "med_class_shared_state", "med_flatten_depth", "med_running_total", "med_zip_truncation",
        "hard_default_mutable", "hard_spiral_order", "hard_lru_cache",
        "med_shallow_copy_accumulate", "med_filter_dict",
    ]

    before_map = {
        "hard_scope_bug": False, "hard_balanced_parens": False, "hard_graph_cycle": False,
        "hard_eval_rpn": False, "med_class_shared_state": False, "med_flatten_depth": False,
        "med_running_total": False, "med_zip_truncation": False,
        "hard_default_mutable": True, "hard_spiral_order": True, "hard_lru_cache": True,
        "med_shallow_copy_accumulate": True, "med_filter_dict": True,
    }
    after_map = {
        "hard_scope_bug": True, "hard_balanced_parens": True, "hard_graph_cycle": True,
        "hard_eval_rpn": True, "med_class_shared_state": True, "med_flatten_depth": True,
        "med_running_total": True, "med_zip_truncation": True,
        "hard_default_mutable": False, "hard_spiral_order": False, "hard_lru_cache": False,
        "med_shallow_copy_accumulate": False, "med_filter_dict": False,
    }

    labels = [p.replace("hard_", "").replace("med_", "") for p in puzzles_order]

    fig, ax = plt.subplots(figsize=(10, 5))

    y = range(len(puzzles_order))
    for i, pid in enumerate(puzzles_order):
        b = before_map[pid]
        a = after_map[pid]
        if not b and a:
            # Newly solved
            ax.barh(i, 1, color=COLORS["green"], alpha=0.8, edgecolor="white")
            ax.text(0.5, i, "FAIL → PASS", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")
        elif b and not a:
            # Regression
            ax.barh(i, 1, color=COLORS["red"], alpha=0.7, edgecolor="white")
            ax.text(0.5, i, "PASS → FAIL", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticks([])
    ax.set_title("Per-Puzzle Changes After GRPO Training", fontsize=12, fontweight="bold")
    ax.invert_yaxis()

    # Divider line
    ax.axhline(y=7.5, color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.text(0.9, 3.5, f"7 newly\nsolved", ha="center", fontsize=10,
            color=COLORS["green"], fontweight="bold")
    ax.text(0.9, 10.5, f"5\nregressions", ha="center", fontsize=10,
            color=COLORS["red"], fontweight="bold")

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig5_puzzle_changes.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT_DIR}/fig5_puzzle_changes.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved fig5_puzzle_changes")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_training_curve()
    fig2_before_after()
    fig3_rollout_analysis()
    fig4_architecture()
    fig5_puzzle_changes()
    print(f"\nAll figures saved to {OUT_DIR}/")
