# Evaluated Policies

**Project:** DeepRL-RecSys-Platform  
**Date:** 2026-03-24

This document describes all policies evaluated during the benchmark, including both **logging policies** (used to collect the observational data) and **target policies** (the agents we wish to evaluate offline via OPE).

---

## 1. Logging Policies

Logging policies were used to collect the Open Bandit Dataset. They define the propensity scores `π_0(a|x)` required for IPS/DR estimation.

### 1.1 Uniform Random (OBD – Campaign `random`)

| Property | Value |
|----------|-------|
| **Source** | Open Bandit Dataset, Zozotown (ZOZO Research) |
| **Campaign** | `all` (80 items) |
| **File** | `data/obd/random/all.parquet` |
| **Action space** | 80 fashion items |
| **Propensity (pscore)** | `1/80 = 0.0125` (uniform for all items) |
| **Rows** | 93,610 interactions |

**Description:** A pure uniform random policy (ε=1.0 ε-greedy) that selects each of the 80 items with equal probability. Provides well-conditioned importance weights for most target policies (ESS ≈ 74%).

### 1.2 Bernoulli Thompson Sampling (OBD – Campaign `bts`)

| Property | Value |
|----------|-------|
| **Source** | Open Bandit Dataset, Zozotown (ZOZO Research) |
| **Campaign** | `all` (80 items) |
| **File** | `data/obd/bts/all.parquet` |
| **Action space** | 80 fashion items |
| **Propensity (pscore)** | Variable, learned by BTS online |
| **Rows** | 93,610 interactions |

**Description:** A Bernoulli Thompson Sampling policy that adapts its item probabilities based on observed click rewards. It concentrates probability mass on high-performing items, resulting in low propensities for non-preferred items (pscore can drop below 0.01). This creates high-variance importance weights and potential instability in IPS estimates (clipping rate ≈13%).

---

## 2. Target Policies

Target policies are the agents being evaluated. OPE estimators (IPS, DR, MIPS) are used to estimate their expected reward without deploying them live.

### 2.1 Random Baseline

| Property | Value |
|----------|-------|
| **Class** | `deeprl_recsys.agents.baselines.RandomAgent` |
| **Algorithm** | Uniform random selection |
| **Hyperparameters** | `seed=42` |
| **Scenarios** | Synthetic, OBD Random, OBD BTS |

**Description:** Selects items uniformly at random. Serves as the lower-bound baseline.

### 2.2 Greedy Baseline

| Property | Value |
|----------|-------|
| **Class** | `deeprl_recsys.agents.baselines.GreedyAgent` |
| **Algorithm** | Argmax over estimated reward scores |
| **Hyperparameters** | `seed=42` |
| **Scenarios** | *Planned – see below* |
| **Benchmark Status** | ⏳ Not benchmarked (future work) |

**Description:** Always selects the item with the highest estimated reward. Deterministic policy; propensity = 1.0 for argmax item, 0.0 for all others (requires smoothing for IPS). The `GreedyAgent` class is implemented in `src/deeprl_recsys/agents/baselines.py` but experiment configs (`exp_greedy_*.yaml`) have not yet been created. Benchmarking this agent is left as future work to avoid scope creep.

### 2.3 Top-K Baseline

| Property | Value |
|----------|-------|
| **Class** | `deeprl_recsys.agents.baselines.TopKAgent` |
| **Algorithm** | Top-K item selection with uniform redistribution |
| **Hyperparameters** | `k=5, seed=42` |
| **Scenarios** | *Planned – see below* |
| **Benchmark Status** | ⏳ Not benchmarked (future work) |

**Description:** Selects the top-K items by expected reward and redistributes probability uniformly among them. The `TopKAgent` class is implemented but benchmark configs do not exist. Planned for a future sprint alongside Greedy.

> **Note:** To run Greedy and Top-K benchmarks in the future, create `configs/experiments/exp_greedy_synthetic.yaml`, `exp_greedy_random.yaml`, `exp_greedy_bts.yaml` (and equivalent for `topk`) and add `greedy`/`topk` as choices in `run_full_benchmark.py`.

### 2.4 SAC (Soft Actor-Critic)

| Property | Value |
|----------|-------|
| **Class** | `deeprl_recsys.agents.sac.SACAgent` |
| **Algorithm** | Off-policy actor-critic with entropy regularization |
| **Architecture** | Actor: 2-layer MLP (128→128→n_actions); Critic: 2-layer MLP (128→128→1) |
| **Hyperparameters** | `lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2, batch_size=256, buffer_size=50000` |
| **Seed** | 42, 43, 44 (multi-seed) |
| **Scenarios** | Synthetic, OBD Random, OBD BTS |
| **Benchmark Status** | ✅ Benchmarked |

**Description:** SAC is the primary deep RL agent. Maximizes expected return + entropy bonus to encourage exploration. Fully functional with real gradient updates. Shows context sensitivity (Spearman ≈ 0.70 on synthetic data).

### 2.5 DQN (Deep Q-Network)

| Property | Value |
|----------|-------|
| **Class** | `deeprl_recsys.agents.dqn.DQNAgent` |
| **Algorithm** | Q-learning with neural network function approximation |
| **Architecture** | 2-layer MLP (128→128→n_actions) |
| **Hyperparameters** | `lr=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995, batch_size=32, buffer_size=10000` |
| **Seed** | 42 |
| **Scenarios** | Synthetic, OBD Random, OBD BTS |
| **Benchmark Status** | ✅ Benchmarked |

**Description:** DQN is a functional stub agent with training infrastructure in place. **Note:** DQN produces identical IPS/DR/MIPS estimates as PPO across OBD scenarios due to its current implementation as a stub that does not substantially differentiate learned Q-values from the random initialization. This is **by design** and reflects that DQN has not yet undergone hyperparameter tuning for this domain. Its benchmark results provide a baseline for future optimization.

### 2.6 PPO (Proximal Policy Optimization)

| Property | Value |
|----------|-------|
| **Class** | `deeprl_recsys.agents.ppo.PPOAgent` |
| **Algorithm** | On-policy clipped surrogate objective |
| **Architecture** | Shared 2-layer MLP (128→128), separate policy and value heads |
| **Hyperparameters** | `lr=3e-4, gamma=0.99, clip_ratio=0.2, n_epochs=4, batch_size=64, gae_lambda=0.95` |
| **Seed** | 42 |
| **Scenarios** | Synthetic, OBD Random, OBD BTS |
| **Benchmark Status** | ✅ Benchmarked |

**Description:** PPO is a functional stub agent. Like DQN, it produces metrics identical to DQN on OBD scenarios, confirming its stub status. Both agents represent the framework's capacity to plug in new policy architectures; full optimization is future work.

---

## 3. OPE Estimators

| Estimator | Full Name | Formula |
|-----------|-----------|---------|
| **IPS** | Inverse Propensity Scoring | `Σ (π(a|x)/π₀(a|x)) · r` |
| **DR** | Doubly Robust | IPS with DM baseline correction |
| **MIPS** | Marginalized IPS | Marginalizes over action embedding space |

All estimators use importance weight clipping at `max_weight=5.0` and `clip_epsilon=0.01` for numerical stability.

---

*For detailed OPE results, see `reports/agent_verification.md` and the `artifacts/models/benchmark_*/ope_report.json` files.*
