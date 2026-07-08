import argparse
import sys
from pathlib import Path

# Ensure core package is in the Python search path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.splits import make_splits, SPLITS_DIR

def parse_args():
    parser = argparse.ArgumentParser(
        description="Regenerate and freeze train/validation/test splits from participant meshes."
    )
    parser.add_argument(
        "--mesh-dir",
        type=str,
        default="2026 Munich Tech Arena - Datas/2026 Munich Tech Arena - Datas/mesh",
        help="Path to directory containing PLY mesh files"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    mesh_path = Path(args.mesh_dir)
    # If relative, resolve from repo root
    if not mesh_path.is_absolute():
        mesh_path = ROOT_DIR / mesh_path
        
    print(f"Scanning mesh directory: {mesh_path}")
    if not mesh_path.exists():
        print(f"Error: Mesh directory does not exist: {mesh_path}")
        sys.exit(1)
        
    pids = sorted([f.stem for f in mesh_path.glob("*.ply")])
    if not pids:
        print(f"Error: No PLY files found in {mesh_path}")
        sys.exit(1)
        
    print(f"Found {len(pids)} subjects on disk.")
    train_pids, val_pids, test_pids = make_splits(pids)
    
    # Ensure splits folder exists
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save the splits to txt files
    for name, plist in [("train", train_pids), ("val", val_pids), ("test", test_pids)]:
        out_file = SPLITS_DIR / f"{name}_pids.txt"
        with open(out_file, "w") as f:
            for pid in plist:
                f.write(f"{pid}\n")
        print(f"  Saved {len(plist)} subjects to splits/{name}_pids.txt")
        
    # Check disjointness and exhaustiveness of output
    overlap = (set(train_pids) & set(val_pids)) | (set(train_pids) & set(test_pids)) | (set(val_pids) & set(test_pids))
    assert not overlap, f"Mutual exclusivity violated in generated splits: {overlap}"
    assert len(train_pids) + len(val_pids) + len(test_pids) == len(pids), "Coverage mismatch!"
    
    print("\nDeterministic splits generation and freezing complete!")

if __name__ == "__main__":
    main()
