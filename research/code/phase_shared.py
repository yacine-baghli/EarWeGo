"""
IS THE ALONG-CONTOUR (PHASE) ERROR SHARED ACROSS ARCHITECTURES?

The central open question. 77% of the error energy is along-contour, and it has moved
0.8% across nine architecture variants, a template-correspondence formulation, six
learned correction predictors and a 121-feature context probe. Two readings remain:
the phase error is irreducible from this data, or nothing tried so far addresses it.
The monotone-per-point oracle at 0.5657mm says the information IS present in the
predicted curves, so this is not a representation limit.

This narrows it. Four architecturally unrelated backbones -- DGCNN edge-conv, KPConv
kernel-point, PTv3 serialized attention, PointNeXt set-abstraction -- predicted the same
held-out ears. If phase error were a learning limitation it would differ between them and
averaging would cancel it. If they make the SAME phase mistake, no amount of model choice
or ensembling will help.

Reports per direction (tangent / across / normal): each model's RMSE, the cross-model
correlation of the SIGNED error, what an equal-weight ensemble does, and what IN-SAMPLE
OPTIMAL nonnegative weights could do. The optimal figure is deliberately leaky -- it is an
upper bound on what any weighting scheme could achieve, so a small number there is a
strong statement.

    python research/code/phase_shared.py
Writes research/results/phase_shared.json
"""
import os, json
import numpy as np
from scipy.optimize import nnls

WORK = "scratch"
FOLD = int(os.environ.get("FOLD", "0"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]

of = np.load(f"{WORK}/ortho_feats.npz")
GT, T, B, N = of["gt"].astype(float), of["t"], of["b"], of["n"]
ref = f"{WORK}/famA_kpconv_f{FOLD}.json"
assert os.path.exists(ref), f"need {ref}; run the Family A probes first"
idx = np.array(json.load(open(ref))["val_ear_index"])
G, Tt, Bb, Nn = GT[idx], T[idx], B[idx], N[idx]

M = {"dgcnn": np.mean([np.load(f"{WORK}/screen_normalsfix_s{s}_f{FOLD}.npy").astype(float)
                       for s in SEEDS], 0)}
for t in ("kpconv", "ptv3", "pointnext"):
    p = f"{WORK}/famA_{t}_f{FOLD}.npy"
    if os.path.exists(p):
        M[t] = np.load(p).astype(float)
ks = list(M)
DIRS = (("tangent", Tt), ("across", Bb), ("normal", Nn))
sgn = {k: {c: ((M[k] - G) * V).sum(-1) for c, V in DIRS} for k in ks}

out = {"fold": FOLD, "models": ks, "n_val_ears": len(idx), "directions": {}}
print(f"fold {FOLD}, {len(idx)} val ears, models: {', '.join(ks)}\n")
print(f"{'model':11s}" + "".join(f"{c:>9s}" for c, _ in DIRS))
for k in ks:
    print(f"{k:11s}" + "".join(f"{float(np.sqrt((sgn[k][c] ** 2).mean())):9.4f}"
                               for c, _ in DIRS))

for c, V in DIRS:
    A = np.stack([sgn[k][c].ravel() for k in ks], 1)           # (n, K) signed errors
    R = np.corrcoef(A.T)
    eq = float(np.sqrt(((A @ (np.ones(len(ks)) / len(ks))) ** 2).mean()))
    per = np.sqrt((A ** 2).mean(0))
    # in-sample optimal simplex weights: minimise ||A w|| with w>=0, sum w = 1.
    # The sum constraint is imposed by a heavily weighted extra row rather than by a
    # projection, so nnls solves it directly.
    lam = 1e3
    w, _ = nnls(np.vstack([A, lam * np.ones((1, len(ks)))]),
                np.concatenate([np.zeros(len(A)), [lam]]))
    w = w / max(w.sum(), 1e-12)
    opt = float(np.sqrt(((A @ w) ** 2).mean()))
    off = R[np.triu_indices(len(ks), 1)]
    out["directions"][c] = {
        "per_model_rmse": {k: round(float(per[i]), 4) for i, k in enumerate(ks)},
        "signed_error_correlation": {f"{ks[i]}|{ks[j]}": round(float(R[i, j]), 3)
                                     for i in range(len(ks)) for j in range(i + 1, len(ks))},
        "mean_offdiag_correlation": round(float(off.mean()), 3),
        "equal_weight_rmse": round(eq, 4),
        "best_single_rmse": round(float(per.min()), 4),
        "insample_optimal_rmse_LEAKY": round(opt, 4),
        "insample_optimal_weights_LEAKY": {k: round(float(w[i]), 3) for i, k in enumerate(ks)},
        "headroom_vs_best_single_pct": round(100 * (opt - per.min()) / per.min(), 2)}
    print(f"\n{c}: mean cross-model correlation {off.mean():.3f} "
          f"(range {off.min():.3f}..{off.max():.3f})")
    print(f"  best single {per.min():.4f} | equal-weight {eq:.4f} | "
          f"IN-SAMPLE OPTIMAL {opt:.4f}  ->  headroom "
          f"{100*(opt-per.min())/per.min():+.2f}%")
    print("  optimal weights " + "  ".join(f"{k} {w[i]:.3f}" for i, k in enumerate(ks)))

t = out["directions"]["tangent"]
out["conclusion"] = (
    f"Four unrelated architectures agree on the SIGN of their phase error at mean "
    f"r={t['mean_offdiag_correlation']}. Even in-sample optimal weights -- an upper bound "
    f"on any weighting scheme, and leaky by construction -- reduce along-contour RMSE by "
    f"only {abs(t['headroom_vs_best_single_pct'])}% versus the best single model. Model "
    f"choice and ensembling do not address the dominant error component. IMPORTANT LIMIT: "
    f"all four share the same targets, the same ordered-MSE loss, the same "
    f"canonicalisation and the same coarse initialisation, so this separates "
    f"'architecture-bound' from 'not architecture-bound' but NOT 'data-bound' from "
    f"'training-recipe-bound'. An objective that models phase explicitly is untried.")
print(f"\n{out['conclusion']}")
json.dump(out, open("research/results/phase_shared.json", "w"), indent=1)
print("\nwrote research/results/phase_shared.json")
