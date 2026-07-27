"""
============================================================================
CONFIG SYSTEM — Load, merge, and freeze experiment configurations
============================================================================

Provides YAML-based configuration management with deep merging:
    base.yaml  <-  version file  <-  CLI --set overrides

Usage:
    from src.config import load_config
    cfg = load_config("configs/v1_baseline.yaml", overrides={"model.k_neighbors": 10})
"""

import copy
import hashlib
import json
from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parent.parent


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins on conflicts)."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _apply_dotted_overrides(cfg: dict, overrides: dict) -> dict:
    """Apply dotted key overrides like {'model.k_neighbors': 10}."""
    cfg = copy.deepcopy(cfg)
    for dotted_key, value in overrides.items():
        keys = dotted_key.split(".")
        d = cfg
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]
        # Try to parse numeric/bool values from string
        if isinstance(value, str):
            value = _parse_value(value)
        d[keys[-1]] = value
    return cfg


def _parse_value(s: str):
    """Parse a CLI string value into the appropriate Python type."""
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def compute_config_hash(cfg: dict) -> str:
    """Compute a stable SHA1 hash of the config for reproducibility tracking."""
    # Remove non-deterministic keys before hashing
    hashable = {k: v for k, v in cfg.items() if k not in ("_base", "_config_hash")}
    canonical = json.dumps(hashable, sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode()).hexdigest()[:12]


def load_config(path: str | Path, overrides: dict = None) -> dict:
    """
    Load and merge a config file with its base and optional CLI overrides.
    
    Resolution order: base.yaml <- version file <- overrides dict
    
    Args:
        path: Path to a YAML config file (absolute or relative to repo root).
        overrides: Optional dict of dotted-key overrides, e.g. {"model.k_neighbors": 10}
    
    Returns:
        Fully merged config dict with a '_config_hash' key appended.
    """
    path = Path(path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, "r") as f:
        version_cfg = yaml.safe_load(f) or {}
    
    # Resolve _base chain
    if "_base" in version_cfg:
        base_path = path.parent / version_cfg["_base"]
        base_cfg = load_config(base_path)  # recursive
        # Remove meta keys from version before merging
        version_clean = {k: v for k, v in version_cfg.items() if not k.startswith("_")}
        cfg = _deep_merge(base_cfg, version_clean)
    else:
        cfg = version_cfg
    
    # Apply CLI overrides
    if overrides:
        cfg = _apply_dotted_overrides(cfg, overrides)
    
    # Stamp config hash
    cfg["_config_hash"] = compute_config_hash(cfg)
    
    return cfg


def resolve_data_paths(cfg: dict) -> dict:
    """Resolve data.root, data.splits_dir to absolute paths relative to repo root."""
    cfg = copy.deepcopy(cfg)
    data = cfg.get("data", {})
    
    root = Path(data.get("root", ""))
    if not root.is_absolute():
        root = ROOT_DIR / root
    data["root"] = str(root)
    data["mesh_dir"] = str(root / "mesh")
    data["landmarks_dir"] = str(root / "landmarks")
    
    splits_dir = Path(data.get("splits_dir", "data/splits"))
    if not splits_dir.is_absolute():
        splits_dir = ROOT_DIR / splits_dir
    data["splits_dir"] = str(splits_dir)
    
    cfg["data"] = data
    return cfg
