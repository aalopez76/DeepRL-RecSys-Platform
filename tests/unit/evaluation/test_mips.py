
import numpy as np
import pytest
from deeprl_recsys.evaluation.ope.estimators import MIPSEstimator

def test_mips_uniform_case():
    """Test that MIPS equals mean reward under uniform marginals and target policy."""
    n = 100
    # Mean reward = 0.5
    rewards = np.array([1.0] * 50 + [0.0] * 50) 
    action_probs = np.array([0.1] * n) # Target policy
    marginal_propensities = np.array([0.1] * n) # Marginal logging propensity
    
    data = {
        "rewards": rewards,
        "action_probs": action_probs,
        "marginal_propensities": marginal_propensities,
        "propensities": np.array([0.5] * n) # Joint propensity (unused by MIPS)
    }
    
    estimator = MIPSEstimator()
    estimate = estimator.estimate(data)
    
    # Estimate = mean( (0.1/0.1) * rewards ) = mean(rewards) = 0.5
    assert estimate == pytest.approx(0.5)

def test_mips_scaling_logic():
    """Test MIPS with specific importance weights."""
    rewards = np.array([1.0, 0.0])
    action_probs = np.array([0.8, 0.2])
    marginal_propensities = np.array([0.4, 0.1])
    
    data = {
        "rewards": rewards,
        "action_probs": action_probs,
        "marginal_propensities": marginal_propensities
    }
    
    estimator = MIPSEstimator()
    estimate = estimator.estimate(data)
    
    # Weight 1: 0.8 / 0.4 = 2.0
    # Weight 2: 0.2 / 0.1 = 2.0
    # Estimate = (2.0*1.0 + 2.0*0.0) / 2 = 1.0
    assert estimate == pytest.approx(1.0)

def test_mips_fallback_to_ips():
    """Test that MIPS falls back to propensities if marginals are missing."""
    rewards = np.array([1.0])
    action_probs = np.array([0.5])
    propensities = np.array([0.25])
    
    data = {
        "rewards": rewards,
        "action_probs": action_probs,
        "propensities": propensities
    }
    
    estimator = MIPSEstimator()
    # Should use 'propensities' as fallback
    estimate = estimator.estimate(data)
    
    assert estimate == pytest.approx(2.0)
