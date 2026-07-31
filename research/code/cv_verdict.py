"""
POOLED-OOF VERDICT for a promoted screening variant vs the base model.

Both models were trained on the SAME five folds, so every one of the 340 ears is held
out exactly once by each. That makes the comparison paired at the ear level over the
whole dataset, which is the only comparison with enough subjects to resolve the ~0.03 mm
effects the single-fold screen cannot.

Reports: per-fold deltas, pooled OOF MLE for both models, a paired per-subject bootstrap
over all 170 subjects, a fold-level sign test, per-contour deltas, and the
tangent/across/normal decomposition on the pooled predictions.

    VARIANT=normals python research/code/cv_verdict.py
Writes research/results/cv_<variant>.json
"""
import os, json
import numpy as np

WORK = "scratch"
VAR = os.environ.get("VARIANT", "normals")
NB = 20000
CONT = [(0, 24, "outer_helix"), (25, 54, "concha"),
        (55, 74, "inner_helix"), (75, 84, "sup._antihelix")]

of = np.load(f"{WORK}/ortho_feats.npz")
GT, T, B, N, SUBJ = of["gt"], of["t"], of["b"], of["n"], of["subj"]
NE = len(GT)


def load(tag):
    """Assemble out-of-fold predictions for one model from its five fold runs."""
    P = np.full((NE, 85, 3), np.nan)
    seen = np.zeros(NE, bool)
    folds, per_fold = np.full(NE, -1), {}
    for f in range(5):
        j = json.load(open(f"{WORK}/screen_{tag}_s0_f{f}.json"))
        idx = np.array(j["val_ear_index"])
        assert not seen[idx].any(), f"{tag}: ear appears in two folds"
        seen[idx] = True
        P[idx] = np.load(f"{WORK}/screen_{tag}_s0_f{f}.npy")
        folds[idx] = f
        per_fold[f] = j["ordered_MLE_mm"]
    assert seen.all(), f"{tag}: {(~seen).sum()} ears never held out"
    return P, folds, per_fold


PB, foldB, pfB = load("base")
PV, foldV, pfV = load(VAR)
assert np.array_equal(foldB, foldV), "the two models used different fold assignments"

EB, EV = PB - GT, PV - GT
dB, dV = np.linalg.norm(EB, axis=2), np.linalg.norm(EV, axis=2)
mB, mV = dB.mean(1), dV.mean(1)          # per-ear MLE

print(f"POOLED OOF over all {NE} ears (each held out exactly once by both models)\n")
print(f"{'fold':>5s} {'n':>4s} {'base':>8s} {VAR:>9s} {'delta':>9s}")
signs = []
for f in range(5):
    m = foldB == f
    d = mV[m].mean() - mB[m].mean()
    signs.append(d)
    print(f"{f:5d} {m.sum():4d} {mB[m].mean():8.4f} {mV[m].mean():9.4f} {d:+9.4f}")
better = sum(1 for d in signs if d < 0)
print(f"\npooled  {NE:4d} {mB.mean():8.4f} {mV.mean():9.4f} {mV.mean()-mB.mean():+9.4f}")
print(f"folds favouring {VAR}: {better}/5")

# paired per-subject bootstrap over every subject
us = np.unique(SUBJ)
per = {s: np.where(SUBJ == s)[0] for s in us}
diff = mV - mB
rng = np.random.RandomState(3)
draws = np.empty(NB)
for k in range(NB):
    pick = np.concatenate([per[s] for s in rng.choice(us, len(us), replace=True)])
    draws[k] = diff[pick].mean()
lo, hi = np.percentile(draws, [2.5, 97.5])
print(f"\npaired per-subject bootstrap ({len(us)} subjects, {NB} draws)")
print(f"  delta {diff.mean():+.4f} mm   95% CI [{lo:+.4f}, {hi:+.4f}]")
print(f"  P(delta < 0) = {(draws < 0).mean():.3f}")

print(f"\n{'contour':16s} {'base':>8s} {VAR:>9s} {'delta':>9s}")
pc = {}
for lo_, hi_, nm in CONT:
    a, b = dB[:, lo_:hi_ + 1].mean(), dV[:, lo_:hi_ + 1].mean()
    pc[nm] = [round(float(a), 4), round(float(b), 4), round(float(b - a), 4)]
    print(f"{nm:16s} {a:8.4f} {b:9.4f} {b-a:+9.4f}")

print(f"\n{'component':12s} {'base':>8s} {VAR:>9s} {'delta':>9s}")
dirs = {}
for nm, V in (("tangent_t", T), ("across_b", B), ("normal_n", N)):
    a = float(np.sqrt((((EB * V).sum(-1)) ** 2).mean()))
    b = float(np.sqrt((((EV * V).sum(-1)) ** 2).mean()))
    dirs[nm] = [round(a, 4), round(b, 4), round(b - a, 4)]
    print(f"{nm:12s} {a:8.4f} {b:9.4f} {b-a:+9.4f}")

verdict = ("ADOPT" if hi < 0 else "REJECT" if lo > 0 else "INDISTINGUISHABLE")
print(f"\nVERDICT: {verdict}")
json.dump({"variant": VAR, "n_ears": int(NE), "n_subjects": int(len(us)),
           "per_fold_base": pfB, "per_fold_variant": pfV,
           "per_fold_delta": [round(float(x), 4) for x in signs],
           "folds_favouring_variant": int(better),
           "pooled_oof_MLE_base": round(float(mB.mean()), 4),
           "pooled_oof_MLE_variant": round(float(mV.mean()), 4),
           "delta_mm": round(float(diff.mean()), 4),
           "ci95": [round(float(lo), 4), round(float(hi), 4)],
           "p_delta_negative": round(float((draws < 0).mean()), 4),
           "per_contour": pc, "directional_rmse": dirs,
           "n_bootstrap": NB, "verdict": verdict,
           "note": ("Raw-network OOF, evaluated with 4 fresh samples; no surface "
                    "projection and no dense-SSM blend, so these are NOT comparable "
                    "to the shipped 1.273/1.3144 full-pipeline numbers.")},
          open(f"research/results/cv_{VAR}.json", "w"), indent=1)
print(f"wrote research/results/cv_{VAR}.json")
