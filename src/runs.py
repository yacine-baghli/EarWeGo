"""
============================================================================
RUN MANAGEMENT — Immutable experiment run directories and registry
============================================================================

Handles run-id creation, directory scaffolding, metadata recording,
and the runs/index.csv leaderboard registry.

run_id format: YYYYMMDD_HHMM_<name>_<gitshort>
"""

import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT_DIR / "runs"
INDEX_CSV = RUNS_DIR / "index.csv"

INDEX_COLUMNS = [
    "run_id", "name", "date", "git_commit", "split", "MD", "median",
    "std", "worst", "P90", "SR@2mm", "SR@3mm", "SR@5mm",
    "Concha_MLE", "config_hash", "weights_path",
]


def _git_short_hash() -> str:
    """Get short git commit hash, or 'nogit' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT_DIR),
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "nogit"
    except Exception:
        return "nogit"


def _git_dirty() -> bool:
    """Check if the git working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(ROOT_DIR),
            capture_output=True, text=True, timeout=5,
        )
        return bool(result.stdout.strip()) if result.returncode == 0 else True
    except Exception:
        return True


def _compute_split_hash(pids: list) -> str:
    """Compute SHA1 hash of a sorted PID list for reproducibility verification."""
    content = "\n".join(sorted(pids))
    return hashlib.sha1(content.encode()).hexdigest()[:12]


def new_run(name: str, config: dict) -> Path:
    """
    Create a new immutable run directory with scaffolding.
    
    Args:
        name: Human-readable name for this run.
        config: Resolved config dict (will be saved as config.resolved.yaml).
        
    Returns:
        Path to the created run directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    git_short = _git_short_hash()
    
    # Sanitize name for filesystem
    safe_name = name.replace(" ", "-").replace("/", "-")
    run_id = f"{timestamp}_{safe_name}_{git_short}"
    
    run_dir = RUNS_DIR / run_id
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    
    # Save resolved config (YAML)
    try:
        import yaml
        with open(run_dir / "config.resolved.yaml", "w") as f:
            # Strip internal keys
            clean_cfg = {k: v for k, v in config.items() if not k.startswith("_")}
            yaml.dump(clean_cfg, f, default_flow_style=False, sort_keys=False)
    except ImportError:
        # Fallback: save as JSON if yaml not available
        with open(run_dir / "config.resolved.json", "w") as f:
            clean_cfg = {k: v for k, v in config.items() if not k.startswith("_")}
            json.dump(clean_cfg, f, indent=2, default=str)
    
    return run_dir


def write_metadata(
    run_dir: Path,
    name: str,
    config: dict,
    train_pids: list,
    val_pids: list = None,
    test_pids: list = None,
    durations: dict = None,
    **extra,
) -> dict:
    """
    Write comprehensive metadata.json for reproducibility.
    
    Contains everything needed to verify a run was produced correctly:
    git state, seed, config hash, split hashes, environment info, timings.
    """
    run_id = run_dir.name
    
    metadata = {
        "run_id": run_id,
        "name": name,
        "created_at": datetime.now().isoformat(),
        "git_commit": _git_short_hash(),
        "git_dirty": _git_dirty(),
        "seed": config.get("seed", 42),
        "config_hash": config.get("_config_hash", ""),
        "split_hashes": {
            "train": _compute_split_hash(train_pids),
        },
        "data_fingerprint": {
            "n_train_subjects": len(train_pids),
            "train_pids_hash": _compute_split_hash(train_pids),
        },
        "python_version": sys.version,
        "platform": platform.platform(),
        "host": platform.node(),
        "key_lib_versions": _get_lib_versions(),
        "durations": durations or {},
    }
    
    if val_pids:
        metadata["split_hashes"]["val"] = _compute_split_hash(val_pids)
        metadata["data_fingerprint"]["n_val_subjects"] = len(val_pids)
    if test_pids:
        metadata["split_hashes"]["test"] = _compute_split_hash(test_pids)
        metadata["data_fingerprint"]["n_test_subjects"] = len(test_pids)
    
    # Merge extra fields
    metadata.update(extra)
    
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    return metadata


def _get_lib_versions() -> dict:
    """Collect versions of key libraries."""
    versions = {}
    for lib_name in ["numpy", "scipy", "sklearn", "trimesh"]:
        try:
            mod = __import__(lib_name)
            versions[lib_name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[lib_name] = "not installed"
    return versions


def register_run(run_dir: Path, summary: dict):
    """
    Append or update a row in runs/index.csv with headline metrics.
    
    Args:
        run_dir: Path to the run directory.
        summary: Dict with evaluation summary metrics.
    """
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name
    
    # Read metadata for extra fields
    meta_path = run_dir / "metadata.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    
    row = {
        "run_id": run_id,
        "name": meta.get("name", ""),
        "date": meta.get("created_at", "")[:10],
        "git_commit": meta.get("git_commit", ""),
        "split": summary.get("split", "val"),
        "MD": f"{summary.get('mean_distance_mm', 0):.4f}",
        "median": f"{summary.get('median_distance_mm', 0):.4f}",
        "std": f"{summary.get('std_distance_mm', 0):.4f}",
        "worst": f"{summary.get('max_distance_mm', 0):.4f}",
        "P90": f"{summary.get('P90_mm', 0):.4f}",
        "SR@2mm": f"{summary.get('SR@2mm', 0):.1f}",
        "SR@3mm": f"{summary.get('SR@3mm', 0):.1f}",
        "SR@5mm": f"{summary.get('SR@5mm', 0):.1f}",
        "Concha_MLE": f"{summary.get('Concha_MLE_mm', 0):.4f}",
        "config_hash": meta.get("config_hash", ""),
        "weights_path": str(run_dir / "weights"),
    }
    
    # Read existing index, remove any existing row for this run_id, append new
    existing_rows = []
    if INDEX_CSV.exists():
        with open(INDEX_CSV, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_rows = [r for r in reader if r.get("run_id") != run_id]
    
    existing_rows.append(row)
    
    with open(INDEX_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(existing_rows)
    
    print(f"  Registered run '{run_id}' in {INDEX_CSV}")


def resolve_weights_dir(pointer_name: str = "submission") -> Optional[Path]:
    """
    Resolve a models/<pointer_name> reference to the actual weights directory.
    
    Resolution order:
    1. Real symlink or Windows junction
    2. .txt pointer file containing the path
    3. None if nothing found
    """
    models_dir = ROOT_DIR / "models"
    pointer = models_dir / pointer_name
    
    # 1. Real symlink or junction (os.path.isdir follows junctions on Windows)
    if pointer.is_dir():
        # Resolve to real path
        return pointer.resolve()
    
    # 2. .txt pointer file
    txt_pointer = models_dir / f"{pointer_name}.txt"
    if txt_pointer.exists():
        target_str = txt_pointer.read_text().strip()
        target = Path(target_str)
        if not target.is_absolute():
            target = ROOT_DIR / target
        if target.is_dir():
            return target
    
    return None


def list_runs() -> list[dict]:
    """Read all runs from index.csv, sorted by MD ascending."""
    if not INDEX_CSV.exists():
        return []
    
    with open(INDEX_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Sort by MD (ascending = best first)
    try:
        rows.sort(key=lambda r: float(r.get("MD", 999)))
    except (ValueError, TypeError):
        pass
    
    return rows
