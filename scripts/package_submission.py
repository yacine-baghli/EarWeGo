"""
============================================================================
PACKAGE SUBMISSION — Build a clean submission zip for Huawei upload
============================================================================

Dereferences models/submission (symlink, junction, or .txt pointer),
copies the real weights + JSON sidecar + src/ into a clean dist/,
and zips it. The resulting zip is the upload artifact.

Usage:
    python scripts/package_submission.py
    python scripts/package_submission.py --output earwego_submission.zip
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.runs import resolve_weights_dir


def main():
    parser = argparse.ArgumentParser(description="Package a clean submission zip for Huawei upload")
    parser.add_argument("--output", type=str, default="earwego_submission.zip",
                        help="Output zip filename (default: earwego_submission.zip)")
    parser.add_argument("--dist-dir", type=str, default="dist",
                        help="Temporary directory for packaging (default: dist/)")
    args = parser.parse_args()
    
    # 1. Resolve the promoted submission weights
    print("Resolving submission weights...")
    weights_dir = resolve_weights_dir("submission")
    
    if weights_dir is None:
        print("!" * 72)
        print("  ERROR: No promoted submission found.")
        print("  Promote a run first:")
        print("    python scripts/promote.py --run <run_id> --as submission")
        print("!" * 72)
        sys.exit(1)
    
    print(f"  Weights directory: {weights_dir}")
    
    # Verify required files
    required_files = ["ear_detector.pkl", "landmark_predictor.pkl"]
    for fname in required_files:
        if not (weights_dir / fname).exists():
            print(f"  ERROR: Missing required file: {weights_dir / fname}")
            sys.exit(1)
    
    # 2. Create clean dist directory
    dist_dir = ROOT_DIR / args.dist_dir
    if dist_dir.exists():
        shutil.rmtree(dist_dir)
    dist_dir.mkdir(parents=True)
    
    print(f"\nBuilding clean distribution in: {dist_dir}")
    
    # 3. Copy models/ (real weights, not pointers)
    dist_models = dist_dir / "models"
    dist_models.mkdir()
    
    for fname in required_files + ["model_info.json"]:
        src = weights_dir / fname
        if src.exists():
            shutil.copy2(str(src), str(dist_models / fname))
            print(f"  Copied: models/{fname}")
    
    # 4. Copy src/ (all Python modules)
    dist_src = dist_dir / "src"
    shutil.copytree(
        str(ROOT_DIR / "src"),
        str(dist_src),
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )
    print(f"  Copied: src/ ({len(list(dist_src.glob('*.py')))} Python files)")
    
    # 5. Copy requirements.txt
    req_src = ROOT_DIR / "requirements.txt"
    if req_src.exists():
        shutil.copy2(str(req_src), str(dist_dir / "requirements.txt"))
        print(f"  Copied: requirements.txt")
    
    # 6. Create the zip
    output_path = ROOT_DIR / args.output
    print(f"\nCreating zip: {output_path}")
    
    with ZipFile(output_path, "w") as zf:
        for root, dirs, files in os.walk(dist_dir):
            for file in files:
                file_path = Path(root) / file
                arcname = file_path.relative_to(dist_dir)
                zf.write(file_path, arcname)
    
    # Print summary
    zip_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nSubmission packaged successfully!")
    print(f"  Output:  {output_path}")
    print(f"  Size:    {zip_size_mb:.1f} MB")
    
    # Show model info if available
    info_path = weights_dir / "model_info.json"
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        print(f"  Name:    {info.get('name', 'N/A')}")
        print(f"  Config:  {info.get('config_hash', 'N/A')}")
        print(f"  Trained: {info.get('trained_at', 'N/A')}")
    
    # Cleanup dist
    shutil.rmtree(dist_dir)
    print(f"  Cleaned up: {dist_dir}")


if __name__ == "__main__":
    main()
