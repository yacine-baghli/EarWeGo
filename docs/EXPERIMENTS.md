# EarWeGo — Experiment Workflow Guide

## Quick Start

### Run an experiment (one command)
```bash
python scripts/run_experiment.py --config configs/v1_baseline.yaml --name baseline
```

This will:
1. Train the model on the frozen train split
2. Evaluate on the val split
3. Save everything into `runs/<run_id>/`
4. Register the results in `runs/index.csv`

### Compare runs
```bash
python scripts/compare_runs.py
```

---

## Directory Structure

```
runs/<run_id>/
    config.resolved.yaml   # exact merged config used (committed)
    metadata.json           # git, seed, hashes, env (committed)
    training_summary.json   # timing info
    weights/
        ear_detector.pkl        # large (~9KB)
        landmark_predictor.pkl  # large (~20MB) — gitignored
        model_info.json         # JSON sidecar for estimator (committed)
    results/
        summary_val.json        # headline metrics (committed)
        metrics_val.csv         # per-ear detailed CSV — gitignored
        dashboard_val.png       # evaluation plots — gitignored
    logs/
        train.log               # stdout capture — gitignored
```

**Committed** (shared ledger): config, metadata, summary JSON, model_info.json
**Local-only** (gitignored): weights/*.pkl, results/*.csv, results/*.png, logs/

---

## Creating a New Experiment

### 1. Create a config file
```yaml
# configs/v2_wider_ssm.yaml
_base: base.yaml
name: v2_wider_ssm
description: "Double SSM components for better shape coverage"

model:
  n_ssm_components: 60
```

### 2. Run it
```bash
python scripts/run_experiment.py --config configs/v2_wider_ssm.yaml --name wider-ssm
```

### 3. Or use CLI overrides (no config file needed)
```bash
python scripts/run_experiment.py \
    --config configs/base.yaml \
    --name quick-test \
    --set model.k_neighbors=10 model.blend_alpha=0.5
```

---

## Promoting a Run

### Promote as best (for local dev)
```bash
python scripts/promote.py --run <run_id> --as best
```

### Promote as submission (for Huawei upload)
```bash
# Must have test evaluation first!
python evaluate.py --run runs/<run_id> --split test
python scripts/promote.py --run <run_id> --as submission
```

### Package for upload
```bash
python scripts/package_submission.py
# Creates earwego_submission.zip with real weights + src/
```

---

## Weight Resolution

`estimator.py` (the Huawei submission class) resolves weights automatically:

1. `MODEL_PATH` environment variable (if set)
2. `models/submission` symlink/junction
3. `models/best` symlink/junction
4. `models/submission.txt` pointer file
5. `models/best.txt` pointer file
6. `models/` directory (legacy fallback)

No code changes needed. Just promote a run and the estimator finds it.

---

## Rules

1. **Train on train, tune on val, report test once.**
2. Never overwrite a previous run's folder — they are immutable.
3. The `runs/index.csv` is the offline leaderboard. Commit it so the team shares results.
4. Weights are local-only (~20MB). Teammates reproduce by re-running the config on the pinned git commit.
5. The shape bank leakage guard will block training if non-train PIDs leak into the KNN blending bank.
