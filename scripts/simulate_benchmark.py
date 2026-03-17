import json
import numpy as np
from pathlib import Path

def simulate_benchmark_data():
    base_dir = Path("artifacts/models")
    agents = ["benchmark_sac_synthetic", "benchmark_dqn_synthetic", "benchmark_random"]
    seeds = [42, 43, 44]
    steps = np.arange(0, 1001, 10)
    ope_steps = np.arange(100, 1001, 100)
    
    for agent in agents:
        agent_dir = base_dir / agent
        for seed in seeds:
            seed_dir = agent_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            
            # Simulate training metrics
            metrics = []
            for step in steps:
                # Upward trend for SAC, flat for Random, slower for DQN
                if "sac" in agent:
                    val = 0.5 + 0.4 * (step / 1000) + np.random.normal(0, 0.05)
                elif "dqn" in agent:
                    val = 0.4 + 0.3 * (step / 1000) + np.random.normal(0, 0.1)
                else: # Random
                    val = 0.3 + np.random.normal(0, 0.05)
                
                metrics.append({"step": int(step), "reward": float(val)})
            
            with open(seed_dir / "train_log.json", "w") as f:
                json.dump({"agent": agent, "seed": seed, "metrics": metrics}, f)
                
            # Simulate intermediate OPE
            with open(seed_dir / "ope_intermediate.jsonl", "w") as f:
                for step in ope_steps:
                    if "sac" in agent:
                        base_val = 0.5 + 0.4 * (step / 1000)
                    else:
                        base_val = 0.3
                    
                    data_point = {
                        "step": int(step),
                        "ips": float(base_val + np.random.normal(0, 0.1)),
                        "dr": float(base_val + np.random.normal(0, 0.05)),
                        "mips": float(base_val + np.random.normal(0, 0.02)),
                        "ess": int(50 + np.random.randint(-10, 10))
                    }
                    f.write(json.dumps(data_point) + "\n")

if __name__ == "__main__":
    simulate_benchmark_data()
    print("Simulated benchmark data generated.")
