"""
============================================================================
RUN EXPERIMENT — One command: train + evaluate + register
============================================================================

Usage:
    python scripts/run_experiment.py --config configs/v1_baseline.yaml --name baseline
    python scripts/run_experiment.py --config configs/v2_arclen.yaml --name arclen --set model.k_neighbors=10
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Run a complete experiment: train + evaluate + register")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--name", type=str, required=True, help="Experiment name")
    parser.add_argument("--set", nargs="*", default=[], help="Config overrides: key=value")
    parser.add_argument("--split", type=str, default="val", help="Evaluation split (default: val)")
    parser.add_argument("--include-val", action="store_true", help="Train on train+val combined")
    args = parser.parse_args()
    
    python = sys.executable
    
    # ── Step 1: Train ────────────────────────────────────────────────────────
    print("=" * 72)
    print("  STEP 1/2: TRAINING")
    print("=" * 72)
    
    train_cmd = [python, str(ROOT_DIR / "train.py"),
                 "--config", args.config,
                 "--name", args.name]
    if args.set:
        train_cmd.extend(["--set"] + args.set)
    if args.include_val:
        train_cmd.append("--include-val")
    
    result = subprocess.run(train_cmd, cwd=str(ROOT_DIR))
    if result.returncode != 0:
        print(f"\nTraining failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    
    # ── Find the latest run directory ────────────────────────────────────────
    runs_dir = ROOT_DIR / "runs"
    if not runs_dir.exists():
        print("Error: No runs/ directory found after training.")
        sys.exit(1)
    
    # Get the most recently created run
    run_dirs = sorted(
        [d for d in runs_dir.iterdir() if d.is_dir() and d.name != ".gitkeep"],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not run_dirs:
        print("Error: No run directories found after training.")
        sys.exit(1)
    
    latest_run = run_dirs[0]
    print(f"\n  Latest run: {latest_run.name}")
    
    # ── Step 2: Evaluate ─────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  STEP 2/2: EVALUATION")
    print("=" * 72)
    
    eval_cmd = [python, str(ROOT_DIR / "evaluate.py"),
                "--run", str(latest_run),
                "--split", args.split]
    
    result = subprocess.run(eval_cmd, cwd=str(ROOT_DIR))
    if result.returncode != 0:
        print(f"\nEvaluation failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    
    # ── Print final summary ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  EXPERIMENT COMPLETE")
    print("=" * 72)
    print(f"  Run directory: {latest_run}")
    print(f"  Config:        {args.config}")
    
    # Print summary if available
    summary_file = latest_run / "results" / f"summary_{args.split}.json"
    if summary_file.exists():
        import json
        with open(summary_file) as f:
            summary = json.load(f)
        print(f"  MD ({args.split}):    {summary.get('mean_distance_mm', 'N/A')} mm")
        print(f"  Median:        {summary.get('median_distance_mm', 'N/A')} mm")
        print(f"  Worst:         {summary.get('max_distance_mm', 'N/A')} mm")
    
    # Print index.csv row
    index_csv = ROOT_DIR / "runs" / "index.csv"
    if index_csv.exists():
        print(f"\n  Leaderboard: {index_csv}")
        with open(index_csv) as f:
            lines = f.readlines()
        if len(lines) > 1:
            print(f"  {lines[0].strip()}")
            print(f"  {lines[-1].strip()}")
    
    print("=" * 72)
    print(f"\nTo promote this run:")
    print(f"  python scripts/promote.py --run {latest_run.name} --as best")
    print(f"  python scripts/promote.py --run {latest_run.name} --as submission")


if __name__ == "__main__":
    main()
