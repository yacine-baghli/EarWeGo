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

## Model families

`train_family.py` is the single trainer for every architecture. A family is one module
exposing `MODEL` with three contract points: `cls.NEEDS` (a **class** attribute, read
before instantiation), `forward(batch) -> {'pred': (B,85,3)}`, and `loss(out, tg[, batch])`.

```bash
FAMILY=kpconv FOLD=0 SEED=0 python research/code/train_family.py
FAMILY=local  FOLD=0 SEED=0 LOCAL_CROPS=scratch/local_crops.npz \
              python research/code/train_family.py
ACCUM=4 CFG_BS=16 DATA=scratch/screen_data_32768nrm.npz ...      # exact, see below
```

| family | what it is | status |
| --- | --- | --- |
| `kpconv` / `pointnext` / `ptv3` | Family A — dense hierarchical point cascades, 8192 pts | trained; KPConv 1.2516, PTv3 1.2417, both in the shipped ensemble |
| `diffusionnet` / `vheat` | Family B — intrinsic mesh; spectral, and a spectrum-free vertex heatmap | built and verified, not yet trained |
| `template` | Family C — dense template correspondence, landmarks as fixed barycentrics | 1.83 mm, rejected |
| `bilateral` | Family E — shared subject token; single / bilateral / bilateral+head | built and verified, not yet trained |
| `phase` | Family F — explicit curve + structurally monotone phase | 1.81 mm, rejected — but see the note in the root README |
| `profile` / `endpoint` | arc-length placement and contour-endpoint specialists | rejected |
| `local` | per-landmark cascade refiner on 0.21 mm patches | built, verified end to end, not yet trained |

`ACCUM` splits only the **forward** pass, never the batch construction: the batch is built
and augmented whole, so every random draw is byte-for-byte what `ACCUM=1` would have drawn,
and each slice's loss carries its true share of the batch. Both properties are asserted
entry-wise by `res_sweep_prep.py`, including a negative control that drops the reweight and
must fail. It is a memory workaround, not a change of the training run — which matters,
because it exists so the 8k/16k/32k resolution sweep is a controlled comparison.

## Training-free diagnostics

Cheap measurements that decide whether an expensive experiment is worth running. Six of the
last eight ideas here were null, which is what makes them worth their own section.

```bash
python research/code/info_limit.py       # -> results/info_limit.json   (~25 min, CPU)
python research/code/curv_probe.py       # does a curvature channel make landmarks
                                         #    more identifiable? retrieval, no training
python research/code/oracle_ladder.py    # -> results/oracle_ladder.json
python research/code/curve_floor.py      # representation cost of a smooth curve
python research/code/arc_profile.py      # is the landmark spacing shared across subjects?
```

`info_limit.py` is the load-bearing one: leave-one-**subject**-out descriptor matching on
the native undecimated crop, resampled at 0.22 mm. Nothing is fitted, so nothing can
overfit; the one honesty requirement — that a descriptor from the ear under test never
enters its own reference set — is asserted. Its seven caveats are stored in the output
file, and the first is the important one: **annotation noise is not measured and is not
measurable here**, because the dataset carries no repeat annotations.

## Results

| file | contents |
| --- | --- |
| `results/folds.json` | frozen subject-grouped fold assignment (verified against trained models) |
| `results/best_current.json` | the current best pipeline, stage by stage, with its bootstrap interval |
| `results/oracle_ladder.json` | what each class of correction would buy, and at how many dof |
| `results/info_limit.json` | information content of the local surface, training-free |
| `results/locality_limit.json` | error vs local density and sharpness, by direction |
| `results/multiseed_*.json` | fold × seed matrices and the two-way variance decomposition |
| `results/ensemble*.json` | member diversity, nested weights, and the equal-weight comparison |
| `results/full_pipeline_*.json` | the complete inference pipeline measured out-of-fold |
| `results/arc_profile.json`, `profile_*.json`, `anchor_deployable.json` | the arc-length-profile line of attack, oracle and deployable |
| `results/curve_floor.json`, `family_F.json` | why Family F failed, and the falsification of our first explanation |
| `results/screening.json`, `cv_normals.json` | the 7-variant base-model screen |
| `results/aggregate_results.json` | baseline + orthogonal-corrector MLEs, per-contour, directional decomposition, oracle identity |
| `results/correlation_analyses.json` | left/right vs geometry-matched correlations, **with the confounding caveat recorded in the file** |

Figures in the root README are regenerated from these files plus the frozen OOF prediction
array by `python research/code/make_figures.py`, which asserts that each plotted headline
matches its source JSON — so a figure cannot silently drift from the result it illustrates.

## Interpretation caveat (recorded in the data files as well)

The directional shares (80.4 % along-contour, 17.6 % across-contour, 2.0 % normal for the
current best prediction; 77.7 / 20.2 / 2.1 for the earlier baseline the file was written
against) are **energy shares of the error**, not causal attributions. The left/right correlations show a
**shared per-subject factor that ear geometry does not explain**; they do not isolate its
cause, because with one paired observation per subject **subject, scan session,
preprocessing and annotation session are mutually confounded**. A shared scan or alignment
effect produces the same signature. Separating them requires repeat annotations, or
features that distinguish scan/alignment from annotation.

## GPU driver

`tc.py` drives the remote GPU (paramiko; `put` / `run` / `get`). Known traps documented in
its docstring: SFTP is chrooted to `$HOME`, and `pkill -f <script>` matches its own shell —
use `pkill -f "[s]cript"`.
