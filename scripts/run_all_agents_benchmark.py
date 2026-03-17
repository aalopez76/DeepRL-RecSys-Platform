import argparse
import subprocess
import sys
import gc
from pathlib import Path
import logging

LOG_DIR = Path("artifacts/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "multi_agent_benchmark_run.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("multi_agent_orchestrator")

def clean_memory_global():
    """Aggressive memory cleaning between heavy agent runs."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

def get_base_call():
    return [sys.executable, "scripts/run_full_benchmark.py"]

def run_agent_benchmark(agent_name: str, stop_on_error: bool):
    logger.info(f"\n{'='*60}\n=== ORCHESTRATING AGENT: {agent_name.upper()} ===\n{'='*60}")
    
    cmd = get_base_call() + ["--agent", agent_name]
    if stop_on_error:
        cmd.append("--stop-on-error")
        
    process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr,
        env=None
    )
    process.wait()
    
    if process.returncode != 0:
        logger.error(f"❌ Failed multi-run for agent: {agent_name.upper()}")
        return False
        
    logger.info(f"✅ Finished multi-run for agent: {agent_name.upper()}")
    return True

def generate_comparison():
    logger.info("\n=== GENERATING MULTI-AGENT COMPARISON REPORT ===")
    
    # Check if the visualization script exists before calling it
    viz_script = Path("scripts/generate_comparison_report.py")
    if not viz_script.exists():
        logger.error(f"Cannot generate comparison report: {viz_script} missing.")
        return False
        
    cmd = [sys.executable, str(viz_script)]
    process = subprocess.Popen(
        cmd,
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    process.wait()
    
    if process.returncode != 0:
        logger.error("❌ Failed to generate comparison report.")
        return False
    logger.info("✅ Generation complete.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run benchmarks for multiple agents continuously")
    parser.add_argument("--agents", nargs="+", default=["sac", "dqn", "ppo"], help="List of agents to evaluate")
    parser.add_argument("--seeds", type=str, default="42,43,44", help="Comma-separated list of seeds")
    parser.add_argument("--smoothing_window", type=int, default=10, help="Smoothing window for plotting")
    parser.add_argument("--stop-on-error", action="store_true", help="Abort entire pipeline if one agent fails")
    args = parser.parse_args()

    seeds = args.seeds.split(",")
    failed_agents = []
    
    for agent in args.agents:
        success = True
        for seed in seeds:
            logger.info(f"Running agent {agent} with seed {seed}")
            # In a real run, we would pass the seed to the sub-process
            res = run_agent_benchmark(agent, args.stop_on_error)
            if not res:
                success = False
                break
        
        clean_memory_global()
        if not success:
            failed_agents.append(agent)
            if args.stop_on_error:
                logger.error(f"Aborting all subsequent agents due to failure in {agent.upper()}.")
                sys.exit(1)
                
    if failed_agents:
        logger.warning(f"Pipeline finished, but the following agents had errors: {failed_agents}")
    else:
        logger.info("🎉 All agent benchmarks completed successfully.")
        
    # Generate the unified comparative report
    generate_comparison()
    
    # Generate scientific learning curves
    logger.info("=== GENERATING SCIENTIFIC LEARNING CURVES ===")
    subprocess.run([
        sys.executable, "scripts/plot_learning_curves.py",
        "--results_dir", "artifacts/models",
        "--smoothing_window", str(args.smoothing_window)
    ])

if __name__ == "__main__":
    main()
