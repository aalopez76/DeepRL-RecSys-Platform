
import math
import numpy as np
from deeprl_recsys.agents.baselines import RandomAgent, GreedyAgent, TopKAgent
from deeprl_recsys.serving.runtime import ServingRuntime
from pathlib import Path

def test_random_agent_uniformity():
    print("Testing RandomAgent uniformity...")
    agent = RandomAgent(seed=42)
    candidates = [1, 2, 3, 4, 5]
    probs = agent.get_action_probabilities({}, candidates)
    
    expected = 1.0 / len(candidates)
    for c in candidates:
        assert math.isclose(probs[c], expected), f"Item {c} prob {probs[c]} != {expected}"
    
    total = sum(probs.values())
    assert math.isclose(total, 1.0, rel_tol=1e-5), f"Sum {total} != 1.0"
    print("  ✓ RandomAgent probabilities are uniform and sum to 1.0")

def test_runtime_predict_validation():
    print("Testing ServingRuntime predict validation & fallback...")
    runtime = ServingRuntime()
    # Mocking runtime.agent since we don't want to load a full SAC agent just for this smoke test
    class MockAgent:
        def get_action_probabilities(self, ctx, candidates):
            # Return unnormalized scores (logits)
            return {c: float(c) for c in candidates}
    
    runtime.agent = MockAgent()
    candidates = [10, 20, 30]
    preds = runtime.predict({}, candidates, k=3)
    
    scores = [p["score"] for p in preds]
    total = sum(scores)
    
    assert math.isclose(total, 1.0, rel_tol=1e-5), f"Runtime predict sum {total} != 1.0"
    assert all(0.0 <= s <= 1.0 for s in scores), "Scores out of range [0, 1]"
    assert scores[0] > scores[1] > scores[2], "Scores not correctly ordered or softmaxed"
    print("  ✓ ServingRuntime applies softmax to unnormalized scores")
    
    # Test with different candidate lengths
    candidates_long = list(range(10))
    preds_long = runtime.predict({}, candidates_long, k=10)
    total_long = sum(p["score"] for p in preds_long)
    assert math.isclose(total_long, 1.0, rel_tol=1e-5), f"Long candidates sum {total_long} != 1.0"
    print("  ✓ ServingRuntime handles different candidate lengths correctly")

if __name__ == "__main__":
    try:
        test_random_agent_uniformity()
        test_runtime_predict_validation()
        print("\n✅ Verification successful: All tests passed.")
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        exit(1)
