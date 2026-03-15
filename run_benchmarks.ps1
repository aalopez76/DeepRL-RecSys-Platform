python -m deeprl_recsys.cli train --config configs/experiments/exp_benchmark_synthetic.yaml
python -m deeprl_recsys.cli evaluate --config configs/experiments/exp_benchmark_synthetic.yaml
python -m deeprl_recsys.cli train --config configs/experiments/exp_benchmark_random.yaml
python -m deeprl_recsys.cli evaluate --config configs/experiments/exp_benchmark_random.yaml
python -m deeprl_recsys.cli train --config configs/experiments/exp_benchmark_bts.yaml
python -m deeprl_recsys.cli evaluate --config configs/experiments/exp_benchmark_bts.yaml
python scripts/generate_benchmark_viz.py
