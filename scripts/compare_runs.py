"""
============================================================================
COMPARE RUNS — Print/plot a ranked table of all experiment runs
============================================================================

Usage:
    python scripts/compare_runs.py
    python scripts/compare_runs.py --top 5 --sort MD
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.runs import list_runs, INDEX_CSV


def main():
    parser = argparse.ArgumentParser(description="Compare experiment runs from index.csv")
    parser.add_argument("--top", type=int, default=0, help="Show only top N runs")
    parser.add_argument("--sort", type=str, default="MD", help="Column to sort by (default: MD)")
    args = parser.parse_args()
    
    rows = list_runs()
    
    if not rows:
        print("No runs found. Run an experiment first:")
        print("  python scripts/run_experiment.py --config configs/v1_baseline.yaml --name baseline")
        return
    
    # Sort
    try:
        rows.sort(key=lambda r: float(r.get(args.sort, 999)))
    except (ValueError, TypeError):
        pass
    
    if args.top > 0:
        rows = rows[:args.top]
    
    # Print table
    print("=" * 110)
    print("  EXPERIMENT LEADERBOARD")
    print("=" * 110)
    
    # Header
    header = f"{'#':>3}  {'Run ID':<45} {'MD':>7} {'Median':>7} {'Worst':>7} {'SR@3mm':>7} {'Split':>5} {'Git':>7}"
    print(header)
    print("-" * 110)
    
    for i, row in enumerate(rows, 1):
        run_id = row.get("run_id", "?")
        md = row.get("MD", "?")
        median = row.get("median", "?")
        worst = row.get("worst", "?")
        sr3 = row.get("SR@3mm", "?")
        split = row.get("split", "?")
        git = row.get("git_commit", "?")
        
        print(f"{i:>3}  {run_id:<45} {md:>7} {median:>7} {worst:>7} {sr3:>7} {split:>5} {git:>7}")
    
    print("=" * 110)
    print(f"  Source: {INDEX_CSV}")
    print(f"  Total runs: {len(rows)}")


if __name__ == "__main__":
    main()
