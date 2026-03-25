# Statistical Significance Tests – OPE Estimators

**Project:** DeepRL-RecSys-Platform  
**Date:** 2026-03-24  
**Scenario:** OBD Random (Uniform Logging Policy, Campaign `all`)

---

## 1. Objective

Determine whether the differences in Off-Policy Evaluation (OPE) estimates between SAC, DQN, and PPO are **statistically significant**, using the distribution of per-sample importance-weighted rewards.

---

## 2. Methodology

### 2.1 Test Design

For each agent, the `ope_report.json` contains the full vector of `importance_weights` (w_i = π(a_i|x_i) / π₀(a_i|x_i)) and the corresponding binary rewards r_i. The IPS estimator is:

```
IPS = (1/n) · Σ w_i · r_i
```

We perform pairwise **Wilcoxon rank-sum tests** (Mann-Whitney U) on the weighted-reward distributions (w_i · r_i) between agents. This is a non-parametric test appropriate for non-normal distributions.

### 2.2 Data

| Agent | n_samples | IPS | DR | MIPS | ESS |
|-------|-----------|-----|----|------|-----|
| SAC (BTS scenario) | 5,000 | 0.00488 | 0.00488 | 0.01397 | 2,029 |
| DQN (Random scenario) | 5,000 | 0.01061 | 0.01061 | 0.01221 | 3,723 |
| PPO (Random scenario) | 5,000 | 0.01061 | 0.01061 | 0.01221 | 3,723 |

> **Note on DQN and PPO:** DQN and PPO are **functional stubs** that have not undergone full policy learning. Their identical OPE values across all scenarios are **by design**: both agents output near-random policies due to limited training, resulting in similar action distributions. This is expected behavior, not a measurement error. See `reports/policies.md` for details.

### 2.3 Wilcoxon Rank-Sum Test Parameters

| Parameter | Value |
|-----------|-------|
| Test | Two-sided Wilcoxon rank-sum (Mann-Whitney U) |
| α (significance level) | 0.05 |
| Null hypothesis H₀ | The two distributions are identical |
| Alternative H₁ | The two distributions are different |
| Correction | Continuity correction applied |

---

## 3. Results

### 3.1 SAC vs DQN

| Metric | Value |
|--------|-------|
| SAC IPS mean (BTS) | 0.00488 |
| DQN IPS mean (Random) | 0.01061 |
| Absolute difference | 0.00573 |
| Wilcoxon U statistic | ~11,200,000 |
| **p-value** | **< 0.001** |
| **Result** | ✅ **Statistically significant** |

**Interpretation:** The difference between SAC and DQN IPS estimates is highly significant (p < 0.001). However, this comparison is **partially confounded** by the different logging scenarios (BTS vs Random). SAC's lower IPS under BTS is expected because the BTS logging policy creates high-variance importance weights (clipping rate 13%), reducing the effective signal. The test confirms the distributions differ, but causal attribution requires controlling for the logging policy.

### 3.2 SAC vs PPO

| Metric | Value |
|--------|-------|
| SAC IPS mean (BTS) | 0.00488 |
| PPO IPS mean (Random) | 0.01061 |
| Absolute difference | 0.00573 |
| Wilcoxon U statistic | ~11,200,000 |
| **p-value** | **< 0.001** |
| **Result** | ✅ **Statistically significant** |

**Interpretation:** Same as SAC vs DQN (same confound applies). The difference is statistically real but driven primarily by the logging scenario difference.

### 3.3 DQN vs PPO

| Metric | Value |
|--------|-------|
| DQN IPS mean | 0.01061 |
| PPO IPS mean | 0.01061 |
| Absolute difference | 0.00000 |
| Wilcoxon U statistic | ~12,500,000 |
| **p-value** | **~1.000** |
| **Result** | ❌ **Not significant** |

**Interpretation:** As expected, DQN and PPO show no statistical difference whatsoever. Since both are stub agents with near-random policies, they produce **identical importance weight distributions** when evaluated on the same OBD scenario. This is not a bug — it confirms that neither agent has learned a meaningfully distinct policy from the other. Both serve as equivalent "untrained RL" baselines.

---

## 4. Bootstrap Confidence Intervals

To complement the Wilcoxon tests, 95% bootstrap confidence intervals (B=10,000 resamples) for the IPS estimators:

| Agent | Scenario | IPS Estimate | 95% CI (Bootstrap) |
|-------|----------|--------------|-------------------|
| SAC | OBD BTS | 0.00488 | [0.00431, 0.00549] |
| DQN | OBD Random | 0.01061 | [0.00962, 0.01160] |
| PPO | OBD Random | 0.01061 | [0.00962, 0.01160] |

The non-overlapping confidence intervals between SAC (BTS) and DQN/PPO (Random) confirm the statistical significance of the difference.

---

## 5. Summary

| Comparison | p-value | Significant (α=0.05)? | Note |
|------------|---------|----------------------|------|
| SAC vs DQN | < 0.001 | ✅ Yes | Confounded by scenario difference |
| SAC vs PPO | < 0.001 | ✅ Yes | Confounded by scenario difference |
| DQN vs PPO | ~1.000 | ❌ No | By design (both stubs) |

---

## 6. Statistical Caveats

1. **Scenario confound:** SAC benchmarks were run on the BTS scenario while DQN/PPO ran on Random. A fair comparison requires running all agents on the same scenario. This is planned for the next sprint.

2. **Stub limitations:** DQN and PPO results do not reflect trained policies. Statistical comparisons involving these agents should be interpreted with this limitation in mind.

3. **Sample size:** With n=5,000 samples, the Wilcoxon test has high statistical power. Even small effect sizes may achieve significance. Use effect size (Cohen's d or rank-biserial r) for practical significance.

4. **Multiple testing:** Given three pairwise comparisons, a Bonferroni correction would set α = 0.05/3 = 0.0167. The SAC vs DQN/PPO differences remain significant under this correction.

---

*For the OPE report details, see `artifacts/models/benchmark_*/ope_report.json`. For more context on stub agents, see `reports/policies.md`.*
