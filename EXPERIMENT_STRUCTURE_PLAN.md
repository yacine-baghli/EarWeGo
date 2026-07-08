# EarWeGo — Experiment & Versioning Reorganization Plan

**For:** the coding agent working in the EarWeGo repo.
**Goal:** every time we try a new version, one command should train it, save its
weights, evaluate it on the frozen validation split, save its results, and register it
so versions are reproducible and comparable. Nothing hardcoded; nothing overwritten.

Follow this plan exactly. Keep existing imports and entry points working (add shims if
you rename things). Do not break `src/estimator.py` — it is the Huawei submission class
and the platform loads it as-is.

---

## 1. Principles (do not violate)

1. **A "version" = code state + config + weights + results + metadata, captured together.**
   Each run is immutable once written; never overwrite a previous run's folder.
2. **Config-driven, not hardcoded.** Hyperparameters, seed, data paths, and the split all
   come from config files, never from constants edited in source.
3. **Reproducible.** Every run records its git commit, seed, config, split hash, and data
   fingerprint. Re-running the same config on the same commit reproduces the numbers.
4. **Single source of truth for data separation** = `data/splits/{train,val,test}_pids.txt`
   (already introduced / to be introduced via `scripts/make_splits.py`). Train on `train`,
   tune on `val`, touch `test` once.
5. **Submission never hardcodes a run path.** `estimator.py` loads whatever weights the
   `models/submission` pointer (or `MODEL_PATH` env var) resolves to.

---

## 2. Target folder structure

Create this layout. Move existing files into it (see §11 Migration). Do not delete history.

```
EarWeGo/
├── src/                          # stable library code (keep import paths)
│   ├── config.py                 # NEW: load + merge + freeze configs
│   ├── runs.py                   # NEW: run-id, run dir, metadata, registry helpers
│   ├── splits.py                 # single source of truth for train/val/test
│   ├── data_loader.py            # reads mesh/landmarks (keep the index,[x y z] parser)
│   ├── predictor.py  estimator.py  evaluation.py  geometry.py  ear_detector.py  metrics.py
│
├── configs/                      # NEW: one YAML per experiment version
│   ├── base.yaml                 # shared defaults (paths, seed, split, all hyperparams)
│   ├── v1_baseline.yaml          # current model captured as a config (see Migration)
│   └── v2_<change>.yaml          # each new idea = a new config that overrides base
│
├── data/                         # dataset + frozen split (meshes gitignored)
│   ├── mesh/  landmarks/          # large — gitignored
│   └── splits/                   # COMMITTED: train_pids.txt val_pids.txt test_pids.txt
│
├── runs/                         # NEW: every run, one immutable dir each
│   ├── index.csv                 # COMMITTED registry: one row per run, headline metrics
│   └── <run_id>/
│       ├── config.resolved.yaml  # exact merged config used
│       ├── metadata.json         # git, seed, hashes, timings, env (reproducibility contract)
│       ├── weights/predictor.pkl # trained artifact(s) — gitignored (large)
│       ├── results/
│       │   ├── metrics_val.csv    # per-ear rigorous metrics (same 36-col schema we already emit)
│       │   ├── summary_val.json   # aggregates: MD, median, std, worst, P90/95, SR@k, per-region
│       │   └── dashboard_val.png  # the evaluation dashboard
│       └── logs/{train.log,eval.log}
│
├── models/                       # pointers to promoted runs (NOT copies)
│   ├── best        -> ../runs/<run_id>/weights   # symlink; current best on val
│   └── submission  -> ../runs/<run_id>/weights   # symlink; the one shipped to Huawei
│
├── scripts/                      # NEW: orchestration CLIs
│   ├── make_splits.py            # regenerate frozen split (seeded, deterministic)
│   ├── run_experiment.py         # train + eval + register in one command
│   ├── promote.py                # point models/best or models/submission at a run
│   └── compare_runs.py           # print/plot a table across all runs
│
├── train.py                      # refactored: --config -> writes weights into a run dir
├── evaluate.py                   # refactored: --run <dir> -> writes results into that run
└── docs/EXPERIMENTS.md           # how the workflow works (write this)
```

`run_id` format: `YYYYMMDD_HHMM_<name>_<gitshort>` — e.g. `20260708_1530_similarity-align_ab12cd`.
Sortable, unique, traceable. `<name>` comes from `--name`; `<gitshort>` = `git rev-parse --short HEAD`.

---

## 3. Config system (`src/config.py`)

- Use **YAML** (add `pyyaml` to requirements). `configs/base.yaml` holds every tunable with
  sane defaults; a version file overrides only what changes.
- `load_config(path, overrides: dict = None) -> dict`: deep-merge `base.yaml` <- version file
  <- CLI `--set key=value` overrides. Return a plain dict and also compute a stable
  `config_hash` (sha1 of canonical JSON).
- `base.yaml` must include at minimum:
  ```yaml
  seed: 42
  data:
    root: "data"           # resolves data/mesh, data/landmarks
    splits_dir: "data/splits"
  eval:
    split: "val"           # train on train, evaluate on val by default
    thresholds_mm: [2, 3, 5, 10]
  model:                   # ALL predictor hyperparams live here (pull them out of source)
    n_ssm_components: 60
    k_neighbors: 5
    blend_alpha: 0.5
    use_regressors: true
    # ...every value currently hardcoded in predictor.py / ear_detector.py
  ```
- Refactor `predictor.py`, `ear_detector.py`, `data_loader.py` to accept these values from
  the config dict instead of module-level constants. Keep old defaults as fallbacks so
  nothing breaks if a key is missing.

---

## 4. Run helpers (`src/runs.py`)

Implement:
- `new_run(name, config) -> Path`: compute `run_id`, create `runs/<run_id>/{weights,results,logs}`,
  write `config.resolved.yaml` and an initial `metadata.json`. Return the run dir.
- `write_metadata(run_dir, **kw)`: metadata.json must contain
  `run_id, name, created_at, git_commit, git_dirty, seed, config_hash, split_hashes{train,val,test},
   data_fingerprint (n_subjects + sha1 of sorted pid list), python_version, key_lib_versions,
   host, durations{train_s, eval_s}`.
- `register_run(run_dir, summary)`: append/update one row in `runs/index.csv` with
  `run_id, name, date, git_commit, split, MD, median, std, worst, P90, SR@2mm, SR@3mm, SR@5mm,
   Concha_MLE, weights_path, config_hash`. Keep it human-readable; this is the offline leaderboard.

---

## 5. Refactor `train.py`

CLI: `python train.py --config configs/v2_arclen.yaml --name arclen [--set model.k_neighbors=7]`

Behavior:
1. Load + freeze config; set all RNG seeds from `config.seed`.
2. Load `train` pids from `data/splits/train_pids.txt`. **Assert** train ∩ (val ∪ test) = ∅.
   Fit the predictor **only** on train subjects. The SSM/KNN shape bank must contain train
   shapes only (no val/test landmarks embedded in the pickle).
3. `run_dir = new_run(name, config)`; save weights to `run_dir/weights/predictor.pkl`.
4. `write_metadata(..., durations.train_s=...)`. Tee stdout to `run_dir/logs/train.log`.
5. Print the run dir path so the next step can consume it.

Keep a thin back-compat path: if called with no `--config`, use `configs/base.yaml`.

---

## 6. Refactor `evaluate.py`

CLI: `python evaluate.py --run runs/<run_id> [--split val|test]`
(also accept `--config` + `--weights` to eval an arbitrary pair.)

Behavior:
1. Load the run's resolved config + weights. Evaluate on the requested split's pids
   (default `val`). **Guard:** refuse if the eval split intersects the run's train pids
   (read from metadata) — hard error, this is the leakage tripwire.
2. Reuse the existing rigorous evaluation to produce the **same 36-column per-ear CSV**
   we already generate (`pid, side, MLE, MdLE, SDLE, P90, SR@k, Bias_*, Procrustes_*,
   Scale_*, Concha_MLE, HRTF_*` …). Write to `run_dir/results/metrics_<split>.csv`.
3. Compute `summary_<split>.json`: MD (mean MLE), median, std, worst, P90/P95, SR@2/3/5/10,
   per-region MLE (outer_helix 0–24, concha 25–54, inner_helix 55–74, sup_antihelix 75–84),
   and the rigid diagnostics (mean `Procrustes_Mean`, `Centroid_Error`, `Scale_Error_Pct`).
4. Render `dashboard_<split>.png` (reuse the existing dashboard code).
5. `register_run(run_dir, summary)` → updates `runs/index.csv`.
6. Never write into `test` results unless `--split test` is passed explicitly; log a warning
   that test is a one-shot final check.

---

## 7. Orchestration scripts

- **`scripts/run_experiment.py --config … --name …`**: calls train then evaluate then prints
  the summary and the new `index.csv` row. One command per new version. Optional flag
  `--push` to send the summary to the web leaderboard via the existing
  `submit.py push --result summary_val.json ...` (only if we keep that integration).
- **`scripts/promote.py --run <run_id> --as {best|submission}`**: repoint `models/best` or
  `models/submission` symlink at that run's `weights/`. Refuse to promote to `submission`
  unless a `test`-split evaluation exists for the run (forces a final check before shipping).
- **`scripts/compare_runs.py [--top 10] [--sort MD]`**: read `runs/index.csv`, print a ranked
  table, and optionally save a small comparison plot (MD + worst per run). This is the local
  mirror of the web leaderboard.
- **`scripts/make_splits.py`**: unchanged from the split plan — regenerates the deterministic
  `data/splits/*.txt` (seed 42) and asserts disjoint + exhaustive.

---

## 8. `estimator.py` weight resolution (don't break the submission)

`LandmarkExtractor.__init__` must resolve weights in this order and **never** hardcode a run id:
1. `MODEL_PATH` env var if set (points at a `weights/` dir or a `.pkl`);
2. else `models/submission` symlink;
3. else `models/best`;
4. else a clear error telling the user to `promote.py` a run.

This keeps the Huawei submission pointed at the promoted weights while letting us evaluate
any run locally by setting `MODEL_PATH`.

---

## 9. Git & storage policy

- **Commit:** `configs/**`, `data/splits/*.txt`, `runs/index.csv`, every run's
  `config.resolved.yaml`, `metadata.json`, `summary_*.json` (small — the shared experiment ledger).
- **Gitignore:** `data/mesh`, `data/landmarks`, `runs/*/weights/**`, `runs/*/results/*.csv`,
  `runs/*/results/*.png`, `runs/*/logs/**`.
- Weights are large (~20 MB pkl). Decision needed (see below): git-lfs, a shared drive/bucket,
  or local-only with the registry as the shared record. Default to **local-only + committed
  metadata/summary** so the team shares the ledger without bloating git; teammates reproduce
  weights by re-running the config on the pinned commit.
- Add a `.gitignore` and a `runs/.gitkeep`.

---

## 10. Integration with the frozen split and the leaderboard

- `configs/base.yaml:data.splits_dir` and the leaderboard's `config.json:val_pids` must both
  derive from `data/splits/val_pids.txt` — one source of truth. If the leaderboard config takes
  a pid list, generate it from that file (don't maintain two lists).
- `run_experiment.py --push` (optional) sends `summary_val.json` to the web board so the local
  `runs/index.csv` and the hosted leaderboard agree.

---

## 11. Migration steps (do these first, in order)

1. Create the folders and `.gitignore`. Move current `models/*.pkl` and `output/*` into a new
   run `runs/20260101_0000_v1-baseline_<gitshort>/` (weights + the existing
   `rigorous_evaluation_results.csv` as `results/metrics_val.csv`; regenerate summary + dashboard).
2. Capture the **current** hyperparameters (whatever is presently hardcoded in `predictor.py`
   / `ear_detector.py`) into `configs/v1_baseline.yaml` so v1 is reproducible.
3. Point `models/best` and `models/submission` at that v1 run.
4. Seed `runs/index.csv` with the v1 row.
5. Only then refactor `train.py`/`evaluate.py` to the config-driven flow (§5–6), verifying v1
   reproduces the same MD (~3.68 mm on val) end to end.

---

## 12. Acceptance criteria

- `python scripts/run_experiment.py --config configs/v1_baseline.yaml --name baseline` produces a
  new immutable `runs/<id>/` with weights, `metrics_val.csv` (36 cols), `summary_val.json`,
  `dashboard_val.png`, `metadata.json`, and a new `runs/index.csv` row — with **no manual steps**.
- Re-running the same config on the same commit yields identical val MD (± floating-point noise)
  and identical split hashes in metadata.
- Training on `train` and evaluating on `val` never touches `test`; the leakage guard raises if
  train ∩ eval ≠ ∅.
- `estimator.py` loads the promoted weights with no code edits; setting `MODEL_PATH` overrides it.
- `compare_runs.py` lists all runs ranked by MD.
- Existing entry points still run (back-compat shims where needed).
- Add `tests/test_runs.py`: run-id uniqueness/sortability, metadata completeness, registry
  append, and the leakage guard.

## 13. Out of scope (do not do)

- No change to the modeling/algorithm itself (alignment fixes, arc-length resampling) — this task
  is only the experiment/versioning scaffolding.
- No cloud infra changes; the web leaderboard stays as-is (optional `--push` only).
