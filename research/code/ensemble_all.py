"""
THE FULL ENSEMBLE: every trained network, not just the seven that happened to ship.

THE REFRAME THIS FILE RESTS ON. Every measured gain in this repository came from variance
reduction and cross-family diversity; no single model ever got better at geometry. So the
question asked of a new arm has been wrong. I was testing whether LOSSFN=dist and
LOSSFN=phuber BEAT the mse member -- a replacement test that dist failed by 0.0008mm and
phuber failed on its confirmation seed. But an ensemble member does not have to be better,
only DIFFERENT: dist's 3-seed pooled OOF is 1.2189 and phuber's 1.2212, both already below
the mse member's 1.2292, and their errors are produced by a different objective. Thirteen
complete networks were sitting unused for that reason alone.

MEMBERS. Twenty complete pooled-OOF networks, grouped by (architecture, objective) because
that is the axis diversity actually lives on:

    dgcnn_mse      seeds 0-2      dgcnn_dist   seeds 0-3     dgcnn_phuber  seeds 0-2
    kpconv         seeds 0-1      ptv3         seeds 0-1
    famE_single    seeds 0-2      famE_bilat   seeds 0-2

famE_bilat is included even though bilateral context was measured null as a REPLACEMENT
(+0.0052mm, CI [-0.0067, +0.0170]): null-as-a-replacement and useless-as-a-member are
different claims, and its seed sd is nearly half single's, which is exactly the kind of
differently-shaped error an ensemble can use.

SELECTION OPTIMISM IS THE WHOLE DANGER HERE, so the headline rule performs NO selection.

    RULE A (headline)  equal weight per GROUP, every group included, no fitting, no
                       choosing. Nothing is selected on the data it is scored on, so there
                       is no optimism to correct.
    RULE B             equal weight per NETWORK. Reported because it is the obvious
                       alternative, but it hands dgcnn 10 of 20 votes purely because that
                       backbone was cheapest to train -- a compute artefact, not a belief.
    RULE C             nested-CV nonnegative weights per group, fitted on the outer fold's
                       TRAINING ears only. Previously measured to gain nothing over equal
                       weights across 16 granularities; reported to confirm that still
                       holds at 7 groups.
    LEAVE-ONE-GROUP-OUT is reported per group as a diagnostic ONLY. Dropping the group
                       whose removal helps most would be selection on the scored data, and
                       that number is labelled accordingly.

Surface projection is applied last, exactly as in ensemble_final.py, with the same
frame assertion (worst GT-to-vertex median < 2mm) that caught a 73.5mm frame bug once.

    python research/code/ensemble_all.py
Writes research/results/ensemble_all.json
"""
import json
import os
import sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import nnls

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "deep_model"))
from surfproj import SurfaceProjector  # noqa: E402

W = "scratch"
NB = 20000
CONT = [(0, 24, "outer_helix"), (25, 54, "concha"),
        (55, 74, "inner_helix"), (75, 84, "sup._antihelix")]

of = np.load(f"{W}/ortho_feats.npz")
GT = of["gt"].astype(np.float64)
SUBJ, FOLD = of["subj"].astype(int), of["fold"].astype(int)
T = of["t"].astype(np.float64)
NE = len(GT)

FJ = json.load(open("research/results/folds.json"))["assignments"]
for a in FJ:
    i = a["ear_index"]
    assert a["fold"] == FOLD[i] and a["subject_group"] == SUBJ[i], f"ear {i}: fold drift"

# ------------------------------------------------------------------- member discovery
SPECS = {
    "dgcnn_mse":    (f"{W}/screen_normalsfix_s{{s}}_f{{f}}", range(6)),
    "dgcnn_dist":   (f"{W}/screen_loss_dist_s{{s}}_f{{f}}_s{{s}}_f{{f}}", range(6)),
    "dgcnn_phuber": (f"{W}/screen_loss_phuber_s{{s}}_f{{f}}_s{{s}}_f{{f}}", range(6)),
    "kpconv":       ([f"{W}/famA_kpconv_f{{f}}", f"{W}/famA_kpconv_s1_f{{f}}",
                      f"{W}/famA_kpconv_s2_f{{f}}", f"{W}/famA_kpconv_s3_f{{f}}"], None),
    "ptv3":         ([f"{W}/famA_ptv3_f{{f}}", f"{W}/famA_ptv3_s1_f{{f}}",
                      f"{W}/famA_ptv3_s2_f{{f}}", f"{W}/famA_ptv3_s3_f{{f}}"], None),
    "famE_single":  (f"{W}/famE_single_s{{s}}_f{{f}}", range(6)),
    "famE_bilat":   (f"{W}/famE_bilat_s{{s}}_f{{f}}", range(6)),
}


def load_net(pat):
    """Assemble one network's pooled OOF, verifying each fold IS the frozen fold."""
    P = np.full((NE, 85, 3), np.nan)
    for f in range(5):
        base = pat.format(f=f)
        if not (os.path.exists(base + ".npy") and os.path.exists(base + ".json")):
            return None
        idx = np.asarray(json.load(open(base + ".json"))["val_ear_index"], int)
        assert set(idx.tolist()) == set(np.where(FOLD == f)[0].tolist()), \
            f"{base}: val_ear_index is not fold {f}"
        P[idx] = np.load(base + ".npy").astype(np.float64)
    assert not np.isnan(P).any(), f"{pat}: an ear was never held out"
    return P


GROUPS, NETS = {}, {}
for g, (spec, seeds) in SPECS.items():
    got = []
    for pat in (spec if isinstance(spec, list) else [spec.format(s=s, f="{f}") for s in seeds]):
        P = load_net(pat)
        if P is not None:
            got.append(P)
    if got:
        GROUPS[g] = np.mean(got, 0)
        NETS[g] = got

mle = lambda P: float(np.linalg.norm(P - GT, axis=2).mean())
print(f"{'group':14s} {'nets':>5s} {'pooled OOF':>11s}   per-net")
for g in GROUPS:
    print(f"{g:14s} {len(NETS[g]):5d} {mle(GROUPS[g]):11.4f}   "
          f"{[round(mle(p), 4) for p in NETS[g]]}")
n_nets = sum(len(v) for v in NETS.values())
print(f"\n{len(GROUPS)} groups, {n_nets} networks")

# ------------------------------------------------------------------ diversity structure
names = list(GROUPS)
sig = {g: np.einsum("elk,elk->el", GROUPS[g] - GT, T) for g in names}   # signed tangent
R = np.zeros((len(names), len(names)))
for i, a in enumerate(names):
    for j, b in enumerate(names):
        R[i, j] = np.corrcoef(sig[a].ravel(), sig[b].ravel())[0, 1]
print("\nsigned along-contour error correlation between groups "
      "(lower = more useful diversity):")
print("               " + " ".join(f"{n[:7]:>8s}" for n in names))
for i, a in enumerate(names):
    print(f"{a:14s} " + " ".join(f"{R[i, j]:8.3f}" for j in range(len(names))))
off = R[~np.eye(len(names), dtype=bool)]
print(f"mean off-diagonal {off.mean():.3f}  (the shipped 3-group set averaged ~0.80)")


def project(ENS, tag):
    md = np.load(f"{W}/mesh_data.npz")
    V, F, VP, FP, Rm, C0 = (md["verts"], md["faces"], md["v_ptr"], md["f_ptr"],
                            md["R"].astype(float), md["c0"].astype(float))
    P = ENS.copy()
    worst = 0.0
    for i in range(NE):
        v = V[VP[i]:VP[i + 1]].astype(float) @ Rm[i] + C0[i]
        f_ = F[FP[i]:FP[i + 1]].astype(np.int64) - VP[i]
        worst = max(worst, float(np.median(cKDTree(v).query(GT[i])[0])))
        P[i] = SurfaceProjector(v, f_).project(ENS[i])[0]
    assert worst < 2.0, f"frame mismatch: worst GT-to-vertex median {worst:.2f}mm"
    print(f"  projected {tag}: {mle(ENS):.4f} -> {mle(P):.4f} mm")
    return P


def paired(A, B, seed=5):
    """B - A, resampling SUBJECTS so a subject's two ears move together."""
    diff = (np.linalg.norm(B - GT, axis=2).mean(1) - np.linalg.norm(A - GT, axis=2).mean(1))
    uu = np.unique(SUBJ)
    bys = np.array([diff[SUBJ == s].mean() for s in uu])
    rng = np.random.RandomState(seed)
    bs = bys[rng.randint(0, len(uu), (NB, len(uu)))].mean(1)
    ci = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
    v = "ADOPT" if ci[1] < 0 else "HARMFUL" if ci[0] > 0 else "INDISTINGUISHABLE"
    return round(float(diff.mean()), 4), ci, round(float((bs < 0).mean()), 4), v


# ------------------------------------------------------------------------- the rules
SHIPPED = np.mean([GROUPS[g] for g in ("dgcnn_mse", "kpconv", "ptv3")], 0)
RULE_A = np.mean([GROUPS[g] for g in names], 0)
RULE_B = np.mean([p for g in names for p in NETS[g]], 0)

# Rule C: nested nonnegative least squares per group, fitted on training ears of each fold
Wt = np.zeros((5, len(names)))
RULE_C = np.zeros_like(GT)
for f in range(5):
    tr, te = FOLD != f, FOLD == f
    assert not (set(SUBJ[tr].tolist()) & set(SUBJ[te].tolist())), "subject leak"
    A = np.stack([GROUPS[g][tr].reshape(-1) for g in names], 1)
    w, _ = nnls(A, GT[tr].reshape(-1))
    w = w / max(w.sum(), 1e-9)
    Wt[f] = w
    RULE_C[te] = np.tensordot(w, np.stack([GROUPS[g][te] for g in names]), axes=(0, 0))

# RULE D: merge NEAR-DUPLICATE groups before weighting. The 0.95 threshold is not chosen
# here -- it is the dup_r already used by ensemble_oof.py, predating this analysis -- and
# the criterion is CORRELATION, never score, so this is not selection on the scored data.
# It matters because equal-weighting near-duplicates silently triples dgcnn's vote: mse,
# dist and phuber correlate 0.971-0.979 with each other, and famE single/bilat 0.971.
DUP_R = 0.95
parent = {g: g for g in names}


def find(x):
    while parent[x] != x:
        x = parent[x]
    return x


for i, a in enumerate(names):
    for j, b in enumerate(names):
        if i < j and R[i, j] >= DUP_R:
            parent[find(b)] = find(a)
clusters = {}
for g in names:
    clusters.setdefault(find(g), []).append(g)
CL = {"+".join(v): np.mean([GROUPS[x] for x in v], 0) for v in clusters.values()}
print(f"\nRule D clusters at dup_r={DUP_R} (correlation-based, score-blind):")
for k, v in clusters.items():
    print(f"  {'+'.join(v)}")
RULE_D = np.mean(list(CL.values()), 0)

print()
out = {"n_groups": len(GROUPS), "n_networks": n_nets, "n_ears": NE, "n_bootstrap": NB,
       "ruleD_clusters": {"dup_r": DUP_R,
                          "source_of_threshold": "ensemble_oof.py dup_r, predates this file",
                          "clusters": ["+".join(v) for v in clusters.values()]},
       "reframe": ("an ensemble member does not need to BEAT the baseline, only to differ "
                   "from it; dist and phuber failed replacement tests but are useful "
                   "members"),
       "groups": {g: {"n_nets": len(NETS[g]), "pooled_oof_mm": round(mle(GROUPS[g]), 4),
                      "per_net_mm": [round(mle(p), 4) for p in NETS[g]]} for g in names},
       "diversity": {"signed_tangent_corr_matrix": {a: {b: round(float(R[i, j]), 3)
                                                        for j, b in enumerate(names)}
                                                    for i, a in enumerate(names)},
                     "mean_off_diagonal": round(float(off.mean()), 3)},
       "rules": {}}

PROJ = {}
for tag, ENS in (("shipped_3group", SHIPPED), ("ruleA_group_equal", RULE_A),
                 ("ruleB_network_equal", RULE_B), ("ruleC_nested_nnls", RULE_C),
                 ("ruleD_dedup_cluster_equal", RULE_D)):
    PROJ[tag] = project(ENS, tag)
    out["rules"][tag] = {"raw_mm": round(mle(ENS), 4), "projected_mm": round(mle(PROJ[tag]), 4)}

BEST_OLD = np.load(f"{W}/ensemble5_proj.npy").astype(np.float64)
out["previous_best_mm"] = round(mle(BEST_OLD), 4)
print(f"\nprevious best (shipped 1.1776mm artefact): {mle(BEST_OLD):.4f} mm")
print("\nvs the previous best, paired subject bootstrap:")
for tag in out["rules"]:
    d, ci, p, v = paired(BEST_OLD, PROJ[tag])
    out["rules"][tag].update({"delta_vs_previous_best_mm": d, "ci95": ci,
                              "p_negative": p, "verdict": v})
    print(f"  {tag:22s} {mle(PROJ[tag]):.4f} mm  delta {d:+.4f}  CI {ci}  {v}")

out["ruleC_weights"] = {"per_fold": [{g: round(float(Wt[f, i]), 4)
                                      for i, g in enumerate(names)} for f in range(5)],
                        "mean": {g: round(float(Wt[:, i].mean()), 4)
                                 for i, g in enumerate(names)}}

# leave-one-group-out, DIAGNOSTIC ONLY
print("\nleave-one-group-out (DIAGNOSTIC -- dropping the best one would be selection):")
out["leave_one_group_out_DIAGNOSTIC"] = {}
for g in names:
    keep = [x for x in names if x != g]
    m = mle(np.mean([GROUPS[x] for x in keep], 0))
    out["leave_one_group_out_DIAGNOSTIC"][g] = round(m - mle(RULE_A), 4)
    print(f"  without {g:14s} {m:.4f}  ({m - mle(RULE_A):+.4f} vs Rule A, pre-projection)")

best_rule = min(out["rules"], key=lambda k: out["rules"][k]["projected_mm"])
out["headline_rule"] = "ruleA_group_equal"
out["best_rule_by_score"] = best_rule
A = out["rules"]["ruleA_group_equal"]
out["conclusion"] = (
    f"Rule A -- equal weight per group, every group included, nothing selected or fitted -- "
    f"reaches {A['projected_mm']} mm against the previous best {out['previous_best_mm']} mm "
    f"({A['delta_vs_previous_best_mm']:+}, CI {A['ci95']}, {A['verdict']}). "
    f"The thirteen previously-unused networks were unused because they lost REPLACEMENT "
    f"tests, which is the wrong test for an ensemble member.")
out["multiplicity_caveat"] = (
    "Five weighting rules are now reported. Quoting whichever scores best is selection over "
    "five correlated choices and would carry optimism. Rule A selects nothing and is the "
    "headline for that reason; Rule D is motivated by a correlation threshold that predates "
    "this file and is score-blind, which makes it defensible a priori but does not exempt it "
    "from the same caution. No rule here has a paired interval excluding zero.")
out["caveats"] = [
    "Rule A is the headline precisely because it selects nothing. Reporting the best of "
    "four rules as the result would carry selection optimism over four correlated choices.",
    "leave_one_group_out is a diagnostic. Dropping the group whose removal helps most is "
    "selection on the scored data and would need nested selection to be honest.",
    "Groups are unequal in size (2-4 networks) and in internal diversity, so equal group "
    "weighting is a choice, not a neutral default -- it is simply a choice made a priori "
    "rather than from the scores.",
    "Members are raw-network OOF; TTA is not applied here. Surface projection is.",
    "The lockbox remains untouched: every number is out-of-fold over the 340 development "
    "ears under the frozen subject-grouped split."]
json.dump(out, open("research/results/ensemble_all.json", "w"), indent=1)
np.save(f"{W}/ensemble_all_ruleA_proj.npy", PROJ["ruleA_group_equal"])
print(f"\n{out['conclusion']}")
print("\nwrote research/results/ensemble_all.json and scratch/ensemble_all_ruleA_proj.npy")
