import numpy as np
from pathlib import Path

# Fixed partition proportions and seed constants
SPLIT_SEED = 42
TRAIN_PROP = 0.70
VAL_PROP = 0.15
TEST_PROP = 0.15

# Directories relative to repository root
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
SPLITS_DIR = ROOT_DIR / "data" / "splits"
DEFAULT_MESH_DIR = ROOT_DIR / "2026 Munich Tech Arena - Datas" / "2026 Munich Tech Arena - Datas" / "mesh"

def make_splits(pids: list[str], seed: int = SPLIT_SEED) -> tuple[list[str], list[str], list[str]]:
    """
    Deterministically partitions participant IDs into train, validation, and test splits.
    
    1. Sorts pids first so ordering is stable and independent of filesystem listing.
    2. Randomly permutes using a seeded NumPy Generator.
    3. Partitions according to target ratios (70% / 15% / 15%).
    4. Returns sorted lists of PIDs for each split.
    
    Guarantees disjointness and coverage.
    """
    sorted_pids = sorted(list(pids))
    n = len(sorted_pids)
    
    if n == 0:
        return [], [], []
        
    # Use Generator with seed for stable permutation across numpy versions and platforms
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    
    n_train = int(np.round(n * TRAIN_PROP))
    n_val = int(np.round(n * VAL_PROP))
    
    train_idx = permutation[:n_train]
    val_idx = permutation[n_train:n_train + n_val]
    test_idx = permutation[n_train + n_val:]
    
    train_pids = sorted([sorted_pids[i] for i in train_idx])
    val_pids = sorted([sorted_pids[i] for i in val_idx])
    test_pids = sorted([sorted_pids[i] for i in test_idx])
    
    # Validation guards (mutual exclusivity and exhaustiveness)
    assert len(set(train_pids) & set(val_pids)) == 0, "Leakage: Overlap between train and validation splits."
    assert len(set(train_pids) & set(test_pids)) == 0, "Leakage: Overlap between train and test splits."
    assert len(set(val_pids) & set(test_pids)) == 0, "Leakage: Overlap between validation and test splits."
    assert len(train_pids) + len(val_pids) + len(test_pids) == n, "Incomplete: Splitting did not cover all input PIDs."
    
    return train_pids, val_pids, test_pids

def load_splits(mesh_dir: str | Path = None) -> tuple[list[str], list[str], list[str]]:
    """
    Loads frozen PID lists from disk (splits/*.txt) if they exist.
    If splits files are not found, derives them dynamically from the meshes on disk.
    
    Returns:
        (train_pids, val_pids, test_pids)
    """
    train_file = SPLITS_DIR / "train_pids.txt"
    val_file = SPLITS_DIR / "val_pids.txt"
    test_file = SPLITS_DIR / "test_pids.txt"
    
    if train_file.exists() and val_file.exists() and test_file.exists():
        with open(train_file, "r") as f:
            train_pids = sorted([line.strip() for line in f if line.strip()])
        with open(val_file, "r") as f:
            val_pids = sorted([line.strip() for line in f if line.strip()])
        with open(test_file, "r") as f:
            test_pids = sorted([line.strip() for line in f if line.strip()])
            
        # Verify file consistency
        overlap = (set(train_pids) & set(val_pids)) | (set(train_pids) & set(test_pids)) | (set(val_pids) & set(test_pids))
        if overlap:
            raise ValueError(f"CRITICAL: Persisted splits contain overlapping participant IDs: {overlap}")
        return train_pids, val_pids, test_pids
        
    # Derive dynamically
    if mesh_dir is None:
        mesh_dir = DEFAULT_MESH_DIR
        
    mesh_path = Path(mesh_dir)
    if not mesh_path.exists():
        raise FileNotFoundError(
            f"Splits text files not found, and mesh directory '{mesh_path}' does not exist to derive dynamically."
        )
        
    pids = [f.stem for f in mesh_path.glob("*.ply")]
    return make_splits(pids)

def get_split(name: str, mesh_dir: str | Path = None) -> list[str]:
    """
    Helper function to query a specific split by name.
    
    Args:
        name: One of 'train', 'val'/'validation', or 'test'.
        mesh_dir: Optional path to mesh directory (used if dynamic derivation is triggered).
        
    Returns:
        List of participant ID strings.
    """
    train_pids, val_pids, test_pids = load_splits(mesh_dir)
    name_lower = name.lower()
    
    if name_lower == "train":
        return train_pids
    elif name_lower in ("val", "validation"):
        return val_pids
    elif name_lower == "test":
        return test_pids
    else:
        raise ValueError(f"Unknown split name: '{name}'. Expected 'train', 'val', or 'test'.")
