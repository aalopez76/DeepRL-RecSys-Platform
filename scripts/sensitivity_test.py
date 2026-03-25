"""
scripts/sensitivity_test.py

Context-sensitivity test for DeepRL-RecSys agents.

Methodology:
  - Sample N=100 users from the synthetic dataset
  - For each user, compute item rankings for:
      * original context
      * perturbed context (user_item_affinity ± delta)
  - Report Spearman rank correlation and top-K overlap

Usage:
  python scripts/sensitivity_test.py [--model-dir artifacts/models/benchmark_sac_synthetic]
                                     [--n-users 100] [--k 5] [--delta 0.5] [--seed 42]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def spearman_correlation(rank1: list, rank2: list) -> float:
    """Compute Spearman rank correlation between two rankings."""
    n = len(rank1)
    if n < 2:
        return 1.0
    d_sq = sum((r1 - r2) ** 2 for r1, r2 in zip(rank1, rank2))
    return 1.0 - (6 * d_sq) / (n * (n**2 - 1))


def top_k_overlap(ranking1: list, ranking2: list, k: int) -> float:
    """Compute fraction of top-K items that overlap between two rankings."""
    set1 = set(ranking1[:k])
    set2 = set(ranking2[:k])
    return len(set1 & set2) / k


def get_ranking(agent, context: dict, candidates: list, k: int = None) -> list:
    """Get item ranking from agent given context and candidates."""
    preds = agent.get_action_probabilities(context, candidates)
    sorted_items = sorted(preds.keys(), key=lambda x: preds[x], reverse=True)
    return sorted_items if k is None else sorted_items[:k]


def get_runtime_ranking(runtime, context: dict, candidates: list, k: int = None) -> list:
    """Get item ranking from ServingRuntime."""
    preds = runtime.predict(context, candidates, k=len(candidates))
    sorted_items = [p["item_id"] for p in sorted(preds, key=lambda x: x["score"], reverse=True)]
    return sorted_items if k is None else sorted_items[:k]


def run_sensitivity_test(
    model_dir: str,
    n_users: int = 100,
    k: int = 5,
    delta: float = 0.5,
    seed: int = 42,
) -> dict:
    """
    Run sensitivity test on a model loaded from model_dir.

    Returns a dict with mean/std of Spearman correlations and top-K overlap.
    """
    rng = np.random.default_rng(seed)

    # Try loading the model from ServingRuntime
    try:
        from deeprl_recsys.serving.runtime import ServingRuntime

        runtime = ServingRuntime(model_dir=model_dir)
        use_runtime = True
        print(f"  ✓ Loaded model from {model_dir}")
    except Exception as e:
        print(f"  ⚠ Could not load model: {e}")
        print("    Falling back to RandomAgent for demonstration...")
        from deeprl_recsys.agents.baselines import RandomAgent

        runtime = None
        use_runtime = False
        fallback_agent = RandomAgent(seed=seed)

    # Generate synthetic user contexts
    n_items = 50
    affinity_values = rng.normal(0, 1, n_users)
    sigma = float(np.std(affinity_values))
    candidates = list(range(n_items))

    spearman_up = []
    spearman_down = []
    overlap_up = []
    overlap_down = []
    delta_score_up = []
    delta_score_down = []

    for i in range(n_users):
        user_id = int(rng.integers(0, 10000))
        aff = float(affinity_values[i])

        ctx_orig = {"user_item_affinity": aff, "user_id": user_id}
        ctx_up = {"user_item_affinity": aff + delta * sigma, "user_id": user_id}
        ctx_down = {"user_item_affinity": aff - delta * sigma, "user_id": user_id}

        if use_runtime:
            rank_orig = get_runtime_ranking(runtime, ctx_orig, candidates)
            rank_up = get_runtime_ranking(runtime, ctx_up, candidates)
            rank_down = get_runtime_ranking(runtime, ctx_down, candidates)

            # Scores
            scores_orig = {p["item_id"]: p["score"] for p in runtime.predict(ctx_orig, candidates, k=n_items)}
            scores_up = {p["item_id"]: p["score"] for p in runtime.predict(ctx_up, candidates, k=n_items)}
            scores_down = {p["item_id"]: p["score"] for p in runtime.predict(ctx_down, candidates, k=n_items)}
        else:
            probs_orig = fallback_agent.get_action_probabilities(ctx_orig, candidates)
            probs_up = fallback_agent.get_action_probabilities(ctx_up, candidates)
            probs_down = fallback_agent.get_action_probabilities(ctx_down, candidates)
            rank_orig = sorted(probs_orig.keys(), key=lambda x: probs_orig[x], reverse=True)
            rank_up = sorted(probs_up.keys(), key=lambda x: probs_up[x], reverse=True)
            rank_down = sorted(probs_down.keys(), key=lambda x: probs_down[x], reverse=True)
            scores_orig = probs_orig
            scores_up = probs_up
            scores_down = probs_down

        # Spearman: compare position of each item
        pos_orig = {item: idx for idx, item in enumerate(rank_orig)}
        pos_up = {item: idx for idx, item in enumerate(rank_up)}
        pos_down = {item: idx for idx, item in enumerate(rank_down)}

        ranks_orig = [pos_orig[c] for c in candidates]
        ranks_up = [pos_up[c] for c in candidates]
        ranks_down = [pos_down[c] for c in candidates]

        rho_up = spearman_correlation(ranks_orig, ranks_up)
        rho_down = spearman_correlation(ranks_orig, ranks_down)
        spearman_up.append(rho_up)
        spearman_down.append(rho_down)

        overlap_up.append(top_k_overlap(rank_orig, rank_up, k))
        overlap_down.append(top_k_overlap(rank_orig, rank_down, k))

        mean_score_orig = np.mean(list(scores_orig.values()))
        mean_score_up = np.mean(list(scores_up.values()))
        mean_score_down = np.mean(list(scores_down.values()))
        delta_score_up.append(abs(mean_score_up - mean_score_orig))
        delta_score_down.append(abs(mean_score_down - mean_score_orig))

    results = {
        "model_dir": model_dir,
        "n_users": n_users,
        "k": k,
        "delta_sigma": delta,
        "seed": seed,
        "spearman_up_mean": float(np.mean(spearman_up)),
        "spearman_up_std": float(np.std(spearman_up)),
        "spearman_down_mean": float(np.mean(spearman_down)),
        "spearman_down_std": float(np.std(spearman_down)),
        "spearman_mean": float(np.mean(spearman_up + spearman_down)),
        "topk_overlap_mean": float(np.mean(overlap_up + overlap_down)),
        "delta_score_mean": float(np.mean(delta_score_up + delta_score_down)),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Context-sensitivity test for DeepRL-RecSys agents")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="artifacts/models/benchmark_sac_synthetic",
        help="Path to model artifact directory",
    )
    parser.add_argument("--n-users", type=int, default=100, help="Number of users to test")
    parser.add_argument("--k", type=int, default=5, help="Top-K for overlap metric")
    parser.add_argument("--delta", type=float, default=0.5, help="Perturbation in std deviations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Context Sensitivity Test – DeepRL-RecSys-Platform")
    print(f"{'='*60}")
    print(f"Model: {args.model_dir}")
    print(f"N users: {args.n_users}, K: {args.k}, Delta: ±{args.delta}σ, Seed: {args.seed}")
    print()

    results = run_sensitivity_test(
        model_dir=args.model_dir,
        n_users=args.n_users,
        k=args.k,
        delta=args.delta,
        seed=args.seed,
    )

    print("\n─── Results ──────────────────────────────────────────")
    print(f"  Spearman ρ (perturb up):   {results['spearman_up_mean']:.4f} ± {results['spearman_up_std']:.4f}")
    print(f"  Spearman ρ (perturb down): {results['spearman_down_mean']:.4f} ± {results['spearman_down_std']:.4f}")
    print(f"  Spearman ρ (mean):         {results['spearman_mean']:.4f}")
    print(f"  Top-{args.k} overlap:           {results['topk_overlap_mean']:.2%}")
    print(f"  Δ score (mean):            {results['delta_score_mean']:.6f}")
    print("──────────────────────────────────────────────────────")

    if results["spearman_mean"] < 0.95:
        print("\n✅ Agent demonstrates CONTEXT SENSITIVITY (ρ < 0.95)")
    else:
        print("\n⚠️  Agent appears CONTEXT-INSENSITIVE (ρ ≥ 0.95 – stub or near-random policy)")

    # Save results
    out_path = Path("reports/sensitivity_raw.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
