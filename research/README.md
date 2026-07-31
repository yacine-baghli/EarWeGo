# Research experiments — reproducible record

Executable code, frozen fold assignments, exact commands and machine-readable aggregate
results for the diagnostics and correction experiments behind the numbers quoted in
[`../deep_model/README.md`](../deep_model/README.md).

**No challenge landmark coordinates are published here.** `results/` contains only fold
indices and aggregate statistics. Every experiment reproduces from the private dataset
placed at the usual path (see the root README), because each script derives its inputs
from the committed splits.

## Frozen fold assignments

`results/folds.json` — the exact subject-grouped 5-fold assignment used for every
out-of-fold (OOF) number. Both ears of a subject always share a fold; the 30 lockbox test
subjects are absent from the file entirely.

```
subject_group = ear_index // 2      # ears stored as consecutive (left, right) pairs
folds         = array_split(RandomState(12345).permutation(unique(subject_group)), 5)
```

The exporter asserts that this rule reproduces the folds the models were actually trained
with, so the file cannot silently drift from the trained artefacts.

## Pipeline order (each step consumes the previous one)

```bash
# 0. base training data (canonical 2048-pt clouds + coarse + GT + train-only SSM)
python research/code/build_corr_data.py        # dense-correspondence data
# 1. 5-fold subject-grouped CV of the base model            -> gpu_cv_f{0..4}.npz
bash  research/code/run_cv.sh                  # FOLD/NFOLD env in gpu_train.py
python research/code/cv_assemble.py            # -> cv_oof.npz (single-sample OOF)
# 2. fresh-sample TTA clouds, then OOF predictions through the FINAL pipeline
python research/code/build_multisample_all.py  # M independent surface samples
python research/code/gpu_oof_tta.py            # -> oof_tta.npz  (4-sample TTA)
python research/code/proj_all.py               # + exact surface projection
                                                      # -> oof_final.npz  (baseline 1.3144)
```

`oof_final.npz` is the frozen baseline every diagnostic below is measured against.

## Diagnostics

```bash
python research/code/decomp_valid.py      # exact directional decomposition +
                                                 # oracle-displacement identity (cross-term)
python research/code/oracles_v2.py  scratch/oof_final.npz   # scalar/affine/monotone oracles
python research/code/oracle_audit.py      # fit (a,b) on half the landmarks,
                                                 # score the other half (is the oracle real?)
python research/code/brief_diagnostics.py # coarse shift, Procrustes, geodesic
python research/code/fold_contamination.py# attention-window fold contamination
```

Oracle conventions (identical in all oracles, and in the differentiable warp used for
training): piecewise-linear interpolation along the **predicted** polyline, linear
extrapolation past the ends, parameters clipped to ±3 mm beyond the curve, monotonicity
enforced (the per-point oracle is solved by dynamic programming over a non-decreasing
parameter sequence). Ground truth only ever selects parameters — never supplies geometry.

## Correction experiments (all rejected; see the results JSON)

```bash
CONTOUR=inner python research/code/build_contour_seq.py   # surface sequence features
CONTOUR=inner python research/code/train_phase_cnn.py     # -> affine (offset,stretch)
CONTOUR=inner python research/code/train_endpoint_heat.py # -> endpoint localisation
python research/code/build_ortho_feats.py                 # per-landmark frames+patches
LOSS=proxy  ALLOW_T=0.0 EPOCHS=200 python research/code/train_ortho.py
LOSS=metric ALLOW_T=0.0 EPOCHS=200 python research/code/train_ortho.py
ALLOW_T=0.2 python research/code/train_ortho.py           # bounded-tangential ablation
```

## Inference ablations (all rejected)

```bash
SAMPLER=area   python research/code/build_clouds_sampler.py   # also: norepl | fps | repl
bash research/code/run_ablate.sh    # sampler x GK x K x temperature, on OOF
```
`gpu_ablate.py` patches the graph size (`GK_OV`), landmark window (`K_OV`) and soft-argmax
temperature (`TEMP`) at inference only, and saves **per-sample** predictions so aggregation
rules (mean / coordinate-median / geometric-median / trimmed) can be compared offline.

## Base-model screening (one representative fold, one change per run)

```bash
python research/code/build_screen_data.py    # -> screen_data_2048.npz (repack; data unchanged)
bash   research/code/run_screen.sh           # base s0, base s1, untied4, untied6, fusion2
python research/code/screen_compare.py       # -> results/screening.json + decision table
```

`gpu_screen.py` takes `VARIANT` / `SEED` / `FOLD` / `EPOCHS` from the environment and
writes one JSON report plus its validation predictions per run. Variants:

| variant | single change vs `base` |
| --- | --- |
| `base` | none — the shipped architecture (813,232 params), run twice to measure training noise |
| `untied4` | the 4 refinement passes stop sharing weights |
| `untied6` | 6 untied coarse-to-fine passes, fixed-radius neighbourhoods 11→3 mm, per-pass offset bounds 7→0.7 mm |
| `pts4096` | 4096 points, with `K`/`GK` rescaled to hold the physical window (96 / 40) |
| `normals` | XYZ plus consistently oriented triangle-derived normals |
| `fusion2` | two independent surface samples fused per forward pass, with a consistency term |
| `chamfer` | ordered loss plus a per-contour curve-Chamfer term |

`K`/`GK` are **not** held fixed when the point count changes: at 2048 points the spacing is
0.995 mm, so `K=48` is a 7.35 mm window and `GK=20` a 4.94 mm graph. Keeping them fixed at
4096 points shrinks both windows by √2, which is why the earlier 8192-point test was
confounded and is not treated as decisive.

`screen_compare.py` refuses to rank variants until both `base` seeds exist, then requires a
variant to beat the seed-to-seed delta **and** have a paired per-subject bootstrap interval
excluding zero. Only the best one or two go to 5-fold CV. Intervals from one fold cover
val-ear sampling, not fold choice.

## Results

| file | contents |
| --- | --- |
| `results/folds.json` | frozen subject-grouped fold assignment (verified against trained models) |
| `results/aggregate_results.json` | baseline + orthogonal-corrector MLEs, per-contour, directional decomposition, oracle identity |
| `results/correlation_analyses.json` | left/right vs geometry-matched correlations, **with the confounding caveat recorded in the file** |

## Interpretation caveat (recorded in the data files as well)

The directional shares (77.7 % along-contour, 20.2 % across-contour, 2.1 % normal) are
**energy shares of the error**, not causal attributions. The left/right correlations show a
**shared per-subject factor that ear geometry does not explain**; they do not isolate its
cause, because with one paired observation per subject **subject, scan session,
preprocessing and annotation session are mutually confounded**. A shared scan or alignment
effect produces the same signature. Separating them requires repeat annotations, or
features that distinguish scan/alignment from annotation.

## GPU driver

`tc.py` drives the remote GPU (paramiko; `put` / `run` / `get`). Known traps documented in
its docstring: SFTP is chrooted to `$HOME`, and `pkill -f <script>` matches its own shell —
use `pkill -f "[s]cript"`.
