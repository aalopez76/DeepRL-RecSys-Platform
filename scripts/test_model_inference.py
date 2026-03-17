import json
import torch
from pathlib import Path
from deeprl_recsys.serving.runtime import ServingRuntime

def test_artifacts():
    base_dir = Path("artifacts/models")
    artifacts = ["dummy_model", "example_model", "latest"]
    
    context = {"user_features": {"age": 30, "gender": "F"}, "history": [101, 205]}
    candidates = [10, 25, 33, 41, 55]
    
    results = {}
    for art_id in artifacts:
        art_path = base_dir / art_id
        if not art_path.exists():
            continue
            
        runtime = ServingRuntime()
        runtime.load(art_path)
        
        preds = runtime.predict(context, candidates, k=5)
        results[art_id] = preds

    with open("inference_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    test_artifacts()
