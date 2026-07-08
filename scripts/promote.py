"""
============================================================================
PROMOTE — Point models/best or models/submission at a run's weights
============================================================================

Creates a Windows junction (or symlink on Unix) from models/<target> to the
run's weights/ directory. Falls back to a .txt pointer file if junction
creation fails (e.g. insufficient privileges).

Usage:
    python scripts/promote.py --run 20260708_1530_baseline_ab12cd --as best
    python scripts/promote.py --run 20260708_1530_baseline_ab12cd --as submission
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT_DIR / "runs"
MODELS_DIR = ROOT_DIR / "models"


def main():
    parser = argparse.ArgumentParser(description="Promote a run's weights to models/best or models/submission")
    parser.add_argument("--run", type=str, required=True, help="Run ID or path to run directory")
    parser.add_argument("--as", dest="target", type=str, required=True,
                        choices=["best", "submission"],
                        help="Target pointer name: 'best' or 'submission'")
    args = parser.parse_args()
    
    # Resolve run directory
    run_input = args.run
    run_dir = RUNS_DIR / run_input
    if not run_dir.exists():
        run_dir = Path(run_input)
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_input}")
        print(f"  Tried: {RUNS_DIR / run_input}")
        sys.exit(1)
    
    weights_dir = run_dir / "weights"
    if not weights_dir.exists():
        print(f"Error: No weights/ directory in run: {run_dir}")
        sys.exit(1)
    
    # Check that weights actually exist
    predictor_pkl = weights_dir / "landmark_predictor.pkl"
    if not predictor_pkl.exists():
        print(f"Error: No landmark_predictor.pkl in {weights_dir}")
        sys.exit(1)
    
    # For submission, require test evaluation to exist
    if args.target == "submission":
        test_summary = run_dir / "results" / "summary_test.json"
        if not test_summary.exists():
            print("!" * 72)
            print("  REFUSED: Cannot promote to 'submission' without test evaluation.")
            print(f"  Missing: {test_summary}")
            print(f"\n  Run test evaluation first:")
            print(f"    python evaluate.py --run \"{run_dir}\" --split test")
            print("!" * 72)
            sys.exit(1)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pointer_path = MODELS_DIR / args.target
    txt_pointer = MODELS_DIR / f"{args.target}.txt"
    
    # Remove existing pointer
    if pointer_path.is_symlink() or pointer_path.is_dir():
        if pointer_path.is_symlink():
            pointer_path.unlink()
        elif platform.system() == "Windows":
            # Junction: remove with rmdir (doesn't delete target)
            try:
                subprocess.run(["cmd", "/c", "rmdir", str(pointer_path)],
                               capture_output=True, check=True)
            except Exception:
                shutil.rmtree(str(pointer_path), ignore_errors=True)
        else:
            pointer_path.unlink()
    
    # Try to create junction/symlink
    created_link = False
    weights_abs = weights_dir.resolve()
    
    if platform.system() == "Windows":
        # Try junction (no admin required)
        try:
            result = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(pointer_path), str(weights_abs)],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                created_link = True
                print(f"  Created junction: {pointer_path} -> {weights_abs}")
        except Exception:
            pass
    else:
        # Unix symlink
        try:
            pointer_path.symlink_to(weights_abs)
            created_link = True
            print(f"  Created symlink: {pointer_path} -> {weights_abs}")
        except OSError:
            pass
    
    # Fallback: .txt pointer file
    # Always write it as a backup, even if junction succeeded
    relative_path = os.path.relpath(weights_abs, ROOT_DIR)
    txt_pointer.write_text(relative_path + "\n")
    
    if not created_link:
        print(f"  Junction/symlink failed. Using .txt pointer: {txt_pointer}")
        print(f"  Content: {relative_path}")
    else:
        print(f"  Also wrote .txt fallback: {txt_pointer}")
    
    print(f"\n  Promoted run '{run_dir.name}' as '{args.target}'")
    
    # Print model info if available
    info_path = weights_dir / "model_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        print(f"  Config hash: {info.get('config_hash', 'N/A')}")
        print(f"  Train PIDs:  {info.get('num_subjects', 'N/A')} subjects")


if __name__ == "__main__":
    main()
