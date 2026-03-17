import argparse
import gc
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# Configurar logging principal para el script
LOG_DIR = Path("artifacts/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "benchmark_run.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("benchmark_orchestrator")

def check_dependencies():
    """Fail-fast: verify critical dependencies."""
    logger.info("Checking dependencies...")
    missing = []
    
    try:
        import torch
        logger.info(f"PyTorch version: {torch.__version__}")
    except ImportError:
        missing.append("torch")
        
    try:
        import pandas as pd
        logger.info(f"Pandas version: {pd.__version__}")
    except ImportError:
        missing.append("pandas")
        
    try:
        import psutil
    except ImportError:
        missing.append("psutil")
        
    if missing:
        logger.error(f"Missing critical dependencies: {', '.join(missing)}")
        logger.error("Please ensure you have installed the project with its extras.")
        sys.exit(1)

def kill_zombie_processes():
    """Find and kill zombie Python processes related to training or evaluation."""
    import psutil
    logger.info("Searching for zombie processes...")
    current_pid = os.getpid()
    killed = 0
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['pid'] == current_pid:
                continue
                
            cmdline = proc.info.get('cmdline', [])
            if not cmdline:
                continue
                
            cmd_str = " ".join(cmdline).lower()
            if "deeprl_recsys.cli" in cmd_str and ("train" in cmd_str or "evaluate" in cmd_str):
                logger.warning(f"Found apparent zombie process (PID: {proc.info['pid']}): {cmd_str}")
                proc.kill()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
            
    if killed > 0:
        logger.info(f"Killed {killed} zombie processes.")
    else:
        logger.info("No zombie processes found.")

def generate_synthetic_data_if_missing():
    """Auto-generate synthetic data if missing."""
    data_path = Path("data/synthetic_demo.parquet")
    if data_path.exists():
        logger.info(f"Synthetic data found at {data_path}")
        return True
        
    logger.info(f"Generating missing synthetic data at {data_path}...")
    try:
        data_path.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np
        import pandas as pd
        n = 5000
        df = pd.DataFrame({
            'action': np.random.randint(0, 50, n),
            'reward': np.random.choice([0.0, 1.0], n, p=[0.9, 0.1]),
            'pscore': 0.02,
            'context': [json.dumps({"user_item_affinity": float(np.random.normal()), "user_id": int(i)}) for i in range(n)]
        })
        df.to_parquet(data_path)
        logger.info("Synthetic data generated successfully.")
        return True
    except Exception as e:
        logger.error(f"Failed to generate synthetic data: {e}")
        return False

def verify_datasets():
    """Ensure Parquet files exist; auto-generate or instruct the user."""
    logger.info("Verifying datasets...")
    
    if not generate_synthetic_data_if_missing():
        return False
        
    datasets = [
        Path("data/obd/random/all.parquet"),
        Path("data/obd/bts/all.parquet")
    ]
    
    for ds in datasets:
        if not ds.exists():
            logger.error(f"Missing dataset: {ds}")
            logger.error(f"Please run the OBD preparation script first: python scripts/prepare_obd.py")
            return False
            
    logger.info("All required datasets found.")
    return True

def clean_memory():
    """Free up CPU and GPU memory."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

def run_command(cmd_list, log_file):
    """Run a command, teeing output to log file and console."""
    cmd_str = " ".join(cmd_list)
    logger.info(f"Executing: {cmd_str}")
    
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"\n{'='*50}\n")
        lf.write(f"Executing: {cmd_str}\n")
        lf.write(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        lf.write(f"{'='*50}\n\n")
        lf.flush()
        
        # Popen to stream stdout/stderr in real-time
        process = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy()
        )
        
        for line in process.stdout:
            sys.stdout.write(line)
            lf.write(line)
            lf.flush()
            
        process.wait()
        
    if process.returncode != 0:
        logger.error(f"Command failed with exit code {process.returncode}: {cmd_str}")
        return False
    
    logger.info(f"Command succeeded: {cmd_str}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Run the full Benchmark Orchestration")
    parser.add_argument("--stop-on-error", action="store_true", help="Abort execution on first error")
    parser.add_argument("--agent", type=str, default="sac", choices=["sac", "dqn", "ppo"], help="Agent to benchmark")
    args = parser.parse_args()
    
    agent = args.agent
    logger.info(f"Starting Full Benchmark Orchestration for Agent: {agent.upper()}")
    check_dependencies()
    kill_zombie_processes()
    
    if not verify_datasets():
        if args.stop_on_error:
            logger.error("Stopping due to missing datasets.")
            sys.exit(1)
        else:
            logger.warning("Continuing despite missing datasets (expect failures).")
            
    # For sac, logs might be called benchmark_synthetic instead of benchmark_sac_synthetic for backward compatibility
    # but the configs we created are exp_dqn_synthetic, etc. Let's use standard prefix if agent != sac
    
    dataset_scenarios = ["synthetic", "random", "bts"]
    scenarios = []
    for s in dataset_scenarios:
        if agent == "sac":
            scenarios.append(f"benchmark_{s}")
        else:
            scenarios.append(f"{agent}_{s}")
    
    python_exec = sys.executable
    has_errors = False
    
    for scenario in scenarios:
        logger.info(f"\n--- Starting Scenario: {scenario} ---")
        config_path = f"configs/experiments/exp_{scenario}.yaml"
        
        # Verify config exists
        if not Path(config_path).exists():
            logger.error(f"Config for scenario '{scenario}' not found at {config_path}")
            has_errors = True
            if args.stop_on_error:
                sys.exit(1)
            continue
            
        # Train
        train_cmd = [python_exec, "-m", "deeprl_recsys.cli", "train", "--config", config_path]
        if not run_command(train_cmd, LOG_FILE):
            has_errors = True
            if args.stop_on_error:
                logger.error(f"Stopping on error during train: {scenario}")
                sys.exit(1)
                
        clean_memory()
        
        # Evaluate
        eval_cmd = [python_exec, "-m", "deeprl_recsys.cli", "evaluate", "--config", config_path]
        if not run_command(eval_cmd, LOG_FILE):
            has_errors = True
            if args.stop_on_error:
                logger.error(f"Stopping on error during evaluate: {scenario}")
                sys.exit(1)
                
        clean_memory()
        
    logger.info("\n--- Regenerating Benchmark Visualization ---")
    viz_cmd = [python_exec, "scripts/generate_benchmark_viz.py"]
    if not run_command(viz_cmd, LOG_FILE):
        logger.error("Failed to regenerate benchmark visualization.")
    else:
        logger.info("Successfully updated docs/benchmarks/SAC_Full_Report.md")
        
    if has_errors:
        logger.warning(f"Benchmark for {agent.upper()} completed with some errors. Check the logs.")
    else:
        logger.info(f"All benchmarks for {agent.upper()} completed successfully.")

        
if __name__ == "__main__":
    main()
