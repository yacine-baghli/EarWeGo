"""
Data loading utilities for the Huawei Tech Arena 2026 challenge.
Handles PLY mesh files and CSV landmark files.
"""

import re
import glob
import os
from pathlib import Path
from typing import Optional

import numpy as np
import trimesh
import pandas as pd


# ─── Constants ───────────────────────────────────────────────────────────────

DATA_ROOT = Path(__file__).resolve().parent.parent / "2026 Munich Tech Arena - Datas" / "2026 Munich Tech Arena - Datas"
MESH_DIR = DATA_ROOT / "mesh"
LANDMARK_DIR = DATA_ROOT / "landmarks"

NUM_LANDMARKS = 85


# ─── Landmark Loading ────────────────────────────────────────────────────────

def parse_landmark_line(line: str) -> tuple[int, np.ndarray]:
    """Parse a single landmark CSV line like '0,[-0.75018139 74.57184769 31.7642007 ]'."""
    line = line.strip()
    if not line:
        return None
    
    # Split index from coordinate
    idx_str, coord_str = line.split(",", 1)
    idx = int(idx_str)
    
    # Parse the coordinate array [x y z]
    coord_str = coord_str.strip().strip("[]")
    coords = np.array([float(x) for x in coord_str.split()])
    
    return idx, coords


def load_landmarks(csv_path: str | Path) -> np.ndarray:
    """
    Load landmarks from a CSV file.
    
    Returns:
        np.ndarray of shape (85, 3) — the 3D coordinates of each landmark.
    """
    landmarks = np.zeros((NUM_LANDMARKS, 3), dtype=np.float64)
    
    with open(csv_path, "r") as f:
        for line in f:
            result = parse_landmark_line(line)
            if result is not None:
                idx, coords = result
                landmarks[idx] = coords
    
    return landmarks


def load_mesh(ply_path: str | Path) -> trimesh.Trimesh:
    """
    Load a PLY mesh file using trimesh.
    
    Returns:
        trimesh.Trimesh object with vertices, faces, and vertex colors.
    """
    mesh = trimesh.load(str(ply_path), process=False)
    return mesh


# ─── Dataset Iteration ───────────────────────────────────────────────────────

def get_participant_ids() -> list[str]:
    """Get sorted list of all participant IDs (e.g., ['P0001', 'P0002', ...])."""
    ply_files = sorted(MESH_DIR.glob("*.ply"))
    return [f.stem for f in ply_files]


def load_participant(pid: str) -> dict:
    """
    Load all data for a single participant.
    
    Returns:
        dict with keys:
            - 'id': participant ID string
            - 'mesh': trimesh.Trimesh object
            - 'landmarks_left': (85, 3) array
            - 'landmarks_right': (85, 3) array
    """
    mesh_path = MESH_DIR / f"{pid}.ply"
    left_path = LANDMARK_DIR / f"{pid}_left_ear_landmarks.csv"
    right_path = LANDMARK_DIR / f"{pid}_right_ear_landmarks.csv"
    
    return {
        "id": pid,
        "mesh": load_mesh(mesh_path),
        "landmarks_left": load_landmarks(left_path),
        "landmarks_right": load_landmarks(right_path),
    }


def load_all_landmarks() -> dict[str, dict[str, np.ndarray]]:
    """
    Load all landmarks (without meshes, for speed).
    
    Returns:
        dict: pid -> {'left': (85,3), 'right': (85,3)}
    """
    result = {}
    for pid in get_participant_ids():
        left_path = LANDMARK_DIR / f"{pid}_left_ear_landmarks.csv"
        right_path = LANDMARK_DIR / f"{pid}_right_ear_landmarks.csv"
        result[pid] = {
            "left": load_landmarks(left_path),
            "right": load_landmarks(right_path),
        }
    return result


# ─── Ear Region Extraction ───────────────────────────────────────────────────

def extract_ear_region(
    mesh: trimesh.Trimesh,
    landmarks: np.ndarray,
    margin: float = 15.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract the ear region from a full head mesh using landmark bounding box.
    
    Args:
        mesh: Full head trimesh.
        landmarks: (85, 3) landmark positions.
        margin: Extra margin (mm) around the landmark bounding box.
    
    Returns:
        vertices: (N, 3) ear region vertices
        faces: (M, 3) ear region faces (re-indexed)
        vertex_mask: boolean mask over original vertices
    """
    # Compute bounding box from landmarks
    lm_min = landmarks.min(axis=0) - margin
    lm_max = landmarks.max(axis=0) + margin
    
    vertices = np.array(mesh.vertices)
    
    # Mask vertices within bounding box
    mask = np.all((vertices >= lm_min) & (vertices <= lm_max), axis=1)
    
    # Get faces where all 3 vertices are in the region
    faces = np.array(mesh.faces)
    face_mask = mask[faces].all(axis=1)
    selected_faces = faces[face_mask]
    
    # Re-index faces
    old_to_new = np.full(len(vertices), -1, dtype=np.int64)
    new_indices = np.where(mask)[0]
    old_to_new[new_indices] = np.arange(len(new_indices))
    
    new_faces = old_to_new[selected_faces]
    new_vertices = vertices[mask]
    
    return new_vertices, new_faces, mask


def extract_ear_region_auto(
    mesh: trimesh.Trimesh,
    side: str = "left",
    margin: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract ear region without landmarks, using heuristic based on Y-coordinate.
    Left ear: positive Y, Right ear: negative Y.
    Looks for the ear as a protruding region on the side of the head.
    
    Args:
        mesh: Full head trimesh.
        side: 'left' or 'right'.
        margin: Extra margin.
    
    Returns:
        vertices, faces of the ear region.
    """
    vertices = np.array(mesh.vertices)
    
    # Heuristic: ears are at extreme Y values
    y_vals = vertices[:, 1]
    
    if side == "left":
        # Left ear is at high positive Y
        y_threshold = np.percentile(y_vals, 95)
        mask = y_vals > (y_threshold - margin)
    else:
        # Right ear is at low negative Y  
        y_threshold = np.percentile(y_vals, 5)
        mask = y_vals < (y_threshold + margin)
    
    faces = np.array(mesh.faces)
    face_mask = mask[faces].all(axis=1)
    selected_faces = faces[face_mask]
    
    old_to_new = np.full(len(vertices), -1, dtype=np.int64)
    new_indices = np.where(mask)[0]
    old_to_new[new_indices] = np.arange(len(new_indices))
    
    new_faces = old_to_new[selected_faces]
    new_vertices = vertices[mask]
    
    return new_vertices, new_faces


# ─── Statistics ──────────────────────────────────────────────────────────────

def compute_landmark_statistics(all_landmarks: dict) -> pd.DataFrame:
    """
    Compute per-landmark statistics across all participants.
    
    Args:
        all_landmarks: dict from load_all_landmarks()
    
    Returns:
        DataFrame with mean, std, min, max for each landmark index.
    """
    # Combine left and right (after mirroring right Y to positive)
    all_left = np.stack([v["left"] for v in all_landmarks.values()])   # (N, 85, 3)
    all_right = np.stack([v["right"] for v in all_landmarks.values()]) # (N, 85, 3)
    
    # Mirror right ear Y to make comparable
    all_right_mirrored = all_right.copy()
    all_right_mirrored[:, :, 1] *= -1
    
    # Combined statistics
    combined = np.concatenate([all_left, all_right_mirrored], axis=0)  # (2N, 85, 3)
    
    records = []
    for i in range(NUM_LANDMARKS):
        pts = combined[:, i, :]  # (2N, 3)
        records.append({
            "landmark_idx": i,
            "mean_x": pts[:, 0].mean(),
            "mean_y": pts[:, 1].mean(),
            "mean_z": pts[:, 2].mean(),
            "std_x": pts[:, 0].std(),
            "std_y": pts[:, 1].std(),
            "std_z": pts[:, 2].std(),
            "spread": np.linalg.norm(pts.std(axis=0)),
        })
    
    return pd.DataFrame(records)


if __name__ == "__main__":
    # Quick test
    pids = get_participant_ids()
    print(f"Found {len(pids)} participants: {pids[:5]}...")
    
    # Load first participant
    data = load_participant(pids[0])
    print(f"\n{data['id']}:")
    print(f"  Mesh: {len(data['mesh'].vertices)} vertices, {len(data['mesh'].faces)} faces")
    print(f"  Left landmarks shape: {data['landmarks_left'].shape}")
    print(f"  Right landmarks shape: {data['landmarks_right'].shape}")
    print(f"  Left landmark 0: {data['landmarks_left'][0]}")
    print(f"  Right landmark 0: {data['landmarks_right'][0]}")
    
    # Ear region extraction
    ear_verts, ear_faces, _ = extract_ear_region(data["mesh"], data["landmarks_left"])
    print(f"\n  Left ear region: {len(ear_verts)} vertices, {len(ear_faces)} faces")
    
    # Stats
    print("\nLoading all landmarks...")
    all_lm = load_all_landmarks()
    stats = compute_landmark_statistics(all_lm)
    print(stats.head(10).to_string())
