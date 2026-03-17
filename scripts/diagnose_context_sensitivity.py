import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from deeprl_recsys.core.registry import create
from pathlib import Path
import argparse

def diagnose(agent_type="sac", model_path="artifacts/models/benchmark_random/model.pt", data_path="data/obd/random/all.parquet", output_dir=None):
    print(f"--- Diagnóstico de Sensibilidad al Contexto: {agent_type.upper()} ---")
    
    model_path = Path(model_path)
    data_path = Path(data_path)
    
    if not model_path.exists() or not data_path.exists():
        print(f"Error: No se encontraron archivos en {model_path} o {data_path}")
        return None

    # Load Agent
    agent = create("agents", agent_type, num_items=50, embedding_dim=32)
    agent.load(str(model_path))
    
    # Load Data
    df = pd.read_parquet(data_path).head(10)
    candidates = list(range(50))
    
    results = []
    
    for i, row in df.iterrows():
        ctx = json.loads(row["context"])
        
        # Original Inference
        probs_orig = agent.get_action_probabilities(ctx, candidates)
        scores_orig = np.array([probs_orig.get(c, 0.0) for c in candidates])
        
        # Perturbation
        ctx_pert = ctx.copy()
        found_numeric = False
        if "user_item_affinity" in ctx_pert:
            ctx_pert["user_item_affinity"] *= 1.5
            found_numeric = True
        elif "features" in ctx_pert:
             ctx_pert["features"] = [f * 1.5 for f in ctx_pert["features"]]
             found_numeric = True
        else:
            for k, v in ctx_pert.items():
                if isinstance(v, (int, float)) and not k.endswith("_id"):
                    ctx_pert[k] = v * 1.5
                    found_numeric = True
                    break
        
        if not found_numeric:
            ctx_pert["user_item_affinity"] = 1.0
        
        probs_pert = agent.get_action_probabilities(ctx_pert, candidates)
        scores_pert = np.array([probs_pert.get(c, 0.0) for c in candidates])
        
        # Calculate Spearman
        coef, _ = spearmanr(scores_orig, scores_pert)
        if np.isnan(coef): coef = 1.0
        
        # Calculate Delta Score
        delta_score = np.abs(scores_orig - scores_pert).mean()
        
        results.append({
            "user_id": ctx.get("user_id", i),
            "spearman": float(coef),
            "delta_score": float(delta_score)
        })

    avg_spearman = np.mean([r["spearman"] for r in results])
    avg_delta = np.mean([r["delta_score"] for r in results])
    
    print(f"\nResumen de 10 usuarios ({agent_type.upper()}):")
    print(f"- Promedio Spearman: {avg_spearman:.4f}")
    print(f"- Promedio Delta Score: {avg_delta:.6e}")
    
    verdict = "INSENSIBLE" if (avg_spearman > 0.999 or avg_delta < 1e-7) else "SENSIBLE"
    print(f"\n[VERDICTO] {verdict}")

    report = {
        "agent": agent_type,
        "avg_spearman": float(avg_spearman),
        "avg_delta": float(avg_delta),
        "verdict": verdict,
        "n_samples": len(results)
    }

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "sensitivity_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {out / 'sensitivity_report.json'}")
    
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="sac")
    parser.add_argument("--model-path", default="artifacts/models/benchmark_random/model.pt")
    parser.add_argument("--data-path", default="data/obd/random/all.parquet")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    
    diagnose(args.agent, args.model_path, args.data_path, args.output_dir)
