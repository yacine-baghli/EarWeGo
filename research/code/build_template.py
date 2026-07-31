"""
FAMILY C step 1 (LOCAL, fold-aware): canonical ear TEMPLATE + dense pseudo-correspondence.

The measured error is 77% ordered correspondence (phase along the contour) and only 2%
surface-normal. Regressing 85 free XYZ points cannot fix phase, because nothing in that
parameterisation knows that landmark 31 must sit "one step" past landmark 30 on the SAME
material curve. This script builds the artefact that makes that structural: ONE canonical
template mesh whose 85 landmarks are FIXED barycentric points (face index + weights), so
any deformation of the template moves all 85 landmarks together and in order.

Template choice: REUSED, not rebuilt. scratch/corr_data.npz already carries the template
that deep_model/dense_ssm.npz was built on -- the training ear closest to the GPA mean of
the training landmark shapes (P0143 right, ear index 149), cropped to 23252 verts / 46200
faces (manifold, 302 boundary edges), with a barycentric landmark map accurate to 0.0095mm.
Rebuilding that from the raw meshes would reproduce the same object. What is NOT reused is
dense_ssm.npz's mean/comps: that PCA was fitted over all 280 ears of the ORIGINAL
train split, which contains validation ears of every CV fold, so it is leaky here. The
mean template and the dense PCA basis are rebuilt per fold from that fold's TRAINING ears
only. Ear 149 is itself a fold-3 VALIDATION ear, so for FOLD=3 the seed template is
rebuilt from the raw mesh of the fold-3 training ear closest to the fold-3 training GPA
mean (needs trimesh + the challenge data; local only).

Pseudo-correspondence: landmark-anchored Laplacian-regularised non-rigid ICP of the
template onto each TRAINING-fold ear. Init is the exact 85-point TPS warp (tps.py), then
    ( I + lam L'L + mu A'A ) V = C + lam L'L V_rest + mu A' Lm_target
with C = point-to-plane targets on the target cloud, A = the sparse barycentric transport
operator, lam annealed stiff -> flexible. Vertices whose match is farther than MAX_DIST
get C_i = V_i, which damps them instead of dropping them; that keeps the system matrix
CONSTANT per lam, so it is prefactorised once and reused for every ear (the whole reason
this runs in minutes instead of hours). A damped vertex is EXTRAPOLATED, not solved, so the
fraction of ears that actually matched each vertex ships as `valid_frac` and is reported --
the prebuilt template's crop overhangs the 16384-point target cloud on some ears, and those
boundary vertices are the whole p2plane_p99 tail.

Everything downstream needs no mesh library: the edge list, cotangent ARAP neighbourhoods,
farthest-point control set and skinning weights are all computed here and shipped as plain
arrays, per constraint 1.

    FOLD=0 python research/code/build_template.py
    FOLD=0 LIMIT=3 ROUNDS=2 python research/code/build_template.py     # smoke test
"""
import os, sys, time, json
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.spatial import cKDTree

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path[:0] = [_HERE, _ROOT, os.path.join(_ROOT, "deep_model")]   # nicp.py needs surfproj
from nicp import edges_from_faces, uniform_laplacian, barycentric_of, transport
from tps import TPS3D

FOLD = int(os.environ.get("FOLD", "0"))
NFOLD = int(os.environ.get("NFOLD", "5"))
LIMIT = int(os.environ.get("LIMIT", "0"))            # 0 = every training ear of the fold
CORR = os.environ.get("CORR", "scratch/corr_data.npz")
SCREEN = os.environ.get("SCREEN", "scratch/screen_data_2048.npz")
OUT = os.environ.get("OUT", f"scratch/template_f{FOLD}.npz")
FOLDS_JSON = os.environ.get("FOLDS_JSON", "research/results/folds.json")
LAMBDAS = [float(x) for x in os.environ.get("LAMBDAS", "60,25,10,4,1.5,0.6").split(",")]
ITERS_PER = int(os.environ.get("ITERS_PER", "3"))
MU = float(os.environ.get("MU_ANCHOR", "25.0"))      # landmark anchor weight in the solve
MAX_DIST = float(os.environ.get("MAX_DIST", "6.0"))  # match rejection radius (mm)
KNRM = int(os.environ.get("KNRM", "16"))             # kNN for target-cloud normals
NCTRL = int(os.environ.get("NCTRL", "512"))          # farthest-point control vertices
CTRL_K = int(os.environ.get("CTRL_K", "8"))          # skinning neighbours per vertex
NBASIS = int(os.environ.get("NBASIS", "120"))        # dense PCA components to keep
ROUNDS = int(os.environ.get("ROUNDS", "1"))          # 2 = re-register from the fold mean
STORE_CORR = int(os.environ.get("STORE_CORR", "1"))
ALLOW_REBUILD = int(os.environ.get("ALLOW_REBUILD", "1"))
MESH = os.environ.get("MESH", "2026 Munich Tech Arena - Datas/"
                              "2026 Munich Tech Arena - Datas/mesh")
CROP = float(os.environ.get("CROP", "12.0"))   # seed-rebuild crop margin around the landmarks
NL = 85


# ------------------------------------------------------------------ mesh operators
# All of this is connectivity work: it happens HERE and ships as arrays, because the GPU
# box has torch/numpy/scipy and no mesh library.
def cotan_adjacency(V, F):
    """symmetric cotangent edge weights (n,n), clamped positive so ARAP stays PSD"""
    n = len(V)
    i, j, k = F[:, 0], F[:, 1], F[:, 2]

    def cot(a, b, c):                                   # cot of the angle at a
        u, v = V[b] - V[a], V[c] - V[a]
        cr = np.linalg.norm(np.cross(u, v), axis=1)
        return (u * v).sum(1) / np.maximum(cr, 1e-12)

    wij, wjk, wki = 0.5 * cot(k, i, j), 0.5 * cot(i, j, k), 0.5 * cot(j, k, i)
    rows = np.r_[i, j, j, k, k, i]
    cols = np.r_[j, i, k, j, i, k]
    data = np.r_[wij, wij, wjk, wjk, wki, wki]
    W = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    W.data = np.clip(W.data, 1e-3, None)
    return W


def padded_neighbours(W):
    """csr adjacency -> (nbr (n,D) int32, weight (n,D) float32, mask (n,D) bool)"""
    n = W.shape[0]
    deg = np.diff(W.indptr)
    D = int(deg.max())
    nbr = np.zeros((n, D), np.int32)
    wt = np.zeros((n, D), np.float32)
    msk = np.zeros((n, D), bool)
    for v in range(n):
        a, b = W.indptr[v], W.indptr[v + 1]
        d = b - a
        nbr[v, :d] = W.indices[a:b]
        wt[v, :d] = W.data[a:b]
        msk[v, :d] = True
        nbr[v, d:] = v                                  # self-padding: zero-length edge
    return nbr, wt, msk


def farthest_point(V, m):
    """deterministic farthest-point subset, seeded at the vertex nearest the centroid"""
    idx = np.empty(m, np.int64)
    idx[0] = int(np.argmin(((V - V.mean(0)) ** 2).sum(1)))
    d = np.linalg.norm(V - V[idx[0]], axis=1)
    for t in range(1, m):
        idx[t] = int(np.argmax(d))
        d = np.minimum(d, np.linalg.norm(V - V[idx[t]], axis=1))
    return idx


def skinning(V, ctrl_idx, k):
    """fixed linear-blend weights: every vertex is driven by its k nearest control verts"""
    tree = cKDTree(V[ctrl_idx])
    d, nn = tree.query(V, k=k)
    sig = np.median(d[:, -1]) / 2.0
    w = np.exp(-(d ** 2) / (2 * sig ** 2)) + 1e-6
    w = w / w.sum(1, keepdims=True)
    return nn.astype(np.int32), w.astype(np.float32), float(sig)


def anchor_operator(n, tri, bw):
    """sparse (85,n) barycentric transport operator A, so A V = the 85 landmarks"""
    rows = np.repeat(np.arange(NL), 3)
    return sp.coo_matrix((bw.ravel(), (rows, tri.ravel())), shape=(NL, n)).tocsr()


def local_normals(P, k):
    """unoriented per-point normals of a cloud by local PCA (sign is irrelevant to p2plane)"""
    _, nn = cKDTree(P).query(P, k=k)
    Q = P[nn] - P[nn].mean(1, keepdims=True)
    C = np.einsum("nki,nkj->nij", Q, Q)
    return np.linalg.eigh(C)[1][:, :, 0]


def similarity(A, B):
    """least-squares similarity A -> B (scale, rotation, translation)"""
    ca, cb = A.mean(0), B.mean(0)
    A0, B0 = A - ca, B - cb
    U, _, Vt = np.linalg.svd(A0.T @ B0)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U = U.copy(); U[:, -1] *= -1; R = U @ Vt
    s = (B0 * (A0 @ R)).sum() / max((A0 * A0).sum(), 1e-9)
    return s, R, cb - s * (ca @ R)


# ------------------------------------------------------------------ registration
def register(V_init, cloud, factors, LtL_rest, AtL, tree, nrm):
    """landmark-anchored NICP of the template onto ONE ear. Returns V and residuals."""
    V = V_init.copy()
    for (_, fac), rhs_rest in zip(factors, LtL_rest):
        for _ in range(ITERS_PER):
            d, j = tree.query(V)
            q, nq = cloud[j], nrm[j]
            C = V + (((q - V) * nq).sum(1, keepdims=True)) * nq      # point-to-plane target
            far = d > MAX_DIST
            C[far] = V[far]                       # damping, not a hard zero weight (see docstring)
            V = fac.solve(C + rhs_rest + MU * AtL)
    d, j = tree.query(V)
    p2p = np.abs(((cloud[j] - V) * nrm[j]).sum(1))
    # d is returned so main() can accumulate per-VERTEX cloud support. A vertex the target
    # cloud does not cover is damped, never solved, and its residual here is meaningless --
    # see the valid_frac report.
    return V, dict(nn_mean=float(d.mean()), nn_p90=float(np.percentile(d, 90)),
                   p2plane_mean=float(p2p.mean()), p2plane_p90=float(np.percentile(p2p, 90)),
                   p2plane_p99=float(np.percentile(p2p, 99)),
                   rejected_frac=float((d > MAX_DIST).mean())), d


def prefactor(n, LtL, AtA):
    t0 = time.time()
    fac = [(lam, spl.splu((sp.eye(n, format="csr") + lam * LtL + MU * AtA).tocsc()))
           for lam in LAMBDAS]
    print(f"prefactorised {len(LAMBDAS)} systems ({n} verts) in {time.time()-t0:.1f}s", flush=True)
    return fac


# ------------------------------------------------------------------ fold-3 seed rebuild
def rebuild_seed(ear, gt_world_of_ear):
    """template mesh from the raw mesh of ONE training ear (trimesh; local only)"""
    from pathlib import Path
    import trimesh
    from src.splits import get_split
    tr = get_split("train", mesh_dir=Path(MESH)); va = get_split("val", mesh_dir=Path(MESH))
    order = [(p, s) for p in tr for s in ("left", "right")] + \
            [(p, s) for p in va for s in ("left", "right")]
    pid, side = order[ear]
    m = trimesh.load(Path(MESH) / f"{pid}.ply")
    V, F = np.asarray(m.vertices, float), np.asarray(m.faces, int)
    if side == "right":                                  # mirrored-left world frame
        V = V * np.array([1., -1., 1.]); F = F[:, ::-1]
    lo, hi = gt_world_of_ear.min(0) - CROP, gt_world_of_ear.max(0) + CROP
    vin = np.all((V >= lo) & (V <= hi), axis=1)
    Fs = F[vin[F].all(axis=1)]
    keep = np.unique(Fs)
    remap = -np.ones(len(V), int); remap[keep] = np.arange(len(keep))
    V, F = V[keep], remap[Fs]
    bf, bw = barycentric_of(gt_world_of_ear, V, F)
    err = np.linalg.norm(transport(F[bf], bw, V) - gt_world_of_ear, axis=1)
    print(f"rebuilt seed from {pid} {side}: {len(V)} verts {len(F)} faces, "
          f"barycentric landmark err mean {err.mean():.4f} max {err.max():.4f} mm", flush=True)
    return V, F, bf.astype(np.int32), bw.astype(np.float32), f"{pid}_{side}"


# ------------------------------------------------------------------ main
def main():
    t_all = time.time()
    d = np.load(CORR, allow_pickle=True)
    s = np.load(SCREEN, allow_pickle=True)
    assert (d["split"] == s["split"]).all(), "corr_data and screen_data are not the same ear order"
    NE = len(d["split"])
    R = s["R"].astype(np.float64); c0 = s["c0"].astype(np.float64)

    # ---- FROZEN subject-grouped folds (research/results/folds.json) ----
    subj = np.arange(NE) // 2
    val_s = set(np.array_split(np.random.RandomState(12345).permutation(np.unique(subj)),
                               NFOLD)[FOLD].tolist())
    tr_idx = np.array([i for i in range(NE) if subj[i] not in val_s])
    va_idx = np.array([i for i in range(NE) if subj[i] in val_s])
    assert not (set(tr_idx) & set(va_idx))
    assert not (set(subj[tr_idx]) & set(subj[va_idx])), "a subject spans both sides of the split"
    if os.path.exists(FOLDS_JSON):
        ref = json.load(open(FOLDS_JSON))["assignments"]
        assert [a["ear_index"] for a in ref] == list(range(NE))
        assert np.array_equal(np.array([a["fold"] for a in ref]) == FOLD,
                              np.isin(np.arange(NE), va_idx)), "folds drifted from folds.json"

    # ---- SLICE TO TRAINING EARS AND DROP THE REST. Nothing below can see a val ear. ----
    gt_tr = d["gt_lms"][tr_idx].astype(np.float64)
    cl_tr = d["clouds"][tr_idx].astype(np.float64)
    TV_pre = d["template_V"].astype(np.float64); TF_pre = d["template_F"].astype(np.int64)
    bf_pre, bw_pre = d["bary_f"], d["bary_w"].astype(np.float64)
    del d, s

    # ---- seed template, fold-aware ----
    # The prebuilt template IS one ear's cropped mesh, identified by matching its barycentric
    # landmarks. The search runs over TRAINING ears ONLY: the single fact needed is "is it a
    # training ear?", and taking an argmin over validation GT to answer it is exactly the use
    # constraint 2 forbids. No training match => it is a validation ear => rebuild.
    L_pre = transport(TF_pre[bf_pre], bw_pre, TV_pre)
    res_tr = np.linalg.norm(gt_tr - L_pre[None], axis=2).mean(1)
    pick0 = int(np.argmin(res_tr))
    if res_tr[pick0] < 0.05:
        seed = int(tr_idx[pick0])
        TV, TF, bf, bw = TV_pre, TF_pre, bf_pre, bw_pre
        seed_tag = f"prebuilt(ear{seed})"
        print(f"seed template: reusing the dense-SSM template, ear {seed} (subject "
              f"{subj[seed]}), which IS a fold-{FOLD} TRAINING ear (landmark match "
              f"{res_tr[pick0]:.4f}mm)", flush=True)
    else:
        assert ALLOW_REBUILD, (
            f"the prebuilt template matches no fold-{FOLD} TRAINING ear (best "
            f"{res_tr[pick0]:.2f}mm), so it is a validation ear and reusing it would leak; "
            "set ALLOW_REBUILD=1 (needs the raw meshes)")
        mu_tr = gt_tr.mean(0)                      # fold-train GPA-mean landmark shape
        al = []
        for g in gt_tr:
            sc_, R_, t_ = similarity(g, mu_tr)
            al.append(sc_ * (g @ R_) + t_)
        pick = int(np.argmin(np.linalg.norm(np.stack(al) - mu_tr, axis=2).mean(1)))
        new_seed = int(tr_idx[pick])
        print(f"the prebuilt template matches no fold-{FOLD} TRAINING ear (best "
              f"{res_tr[pick0]:.2f}mm) -> rebuilding the seed from training ear {new_seed}",
              flush=True)
        TV, TF, bf, bw, who = rebuild_seed(new_seed, gt_tr[pick])
        bw = bw.astype(np.float64)
        seed, seed_tag = new_seed, f"rebuilt(ear{new_seed},{who})"
    assert seed in set(tr_idx.tolist()), "LEAKAGE: seed template is not a training ear"

    # canonical frame of the seed ear, then similarity-aligned to the fold-TRAIN mean shape
    V0 = (TV - c0[seed]) @ R[seed].T
    tri = TF[bf]
    gt_can = np.stack([(gt_tr[t] - c0[i]) @ R[i].T for t, i in enumerate(tr_idx)])
    mean_lm = gt_can.mean(0)
    sc, Rr, tt = similarity(transport(tri, bw, V0), mean_lm)
    V0 = sc * (V0 @ Rr) + tt
    n = len(V0)
    print(f"template {n} verts {len(TF)} faces | aligned to the fold-train mean landmark "
          f"shape (scale {sc:.4f}) | landmark map err "
          f"{np.linalg.norm(transport(tri, bw, V0) - mean_lm, axis=1).mean():.3f}mm to that mean",
          flush=True)

    # ---- connectivity artefacts (shipped, so the GPU needs no mesh library) ----
    E = edges_from_faces(TF)
    Lap = uniform_laplacian(n, E).tocsr()
    LtL = (Lap.T @ Lap).tocsr()
    Wc = cotan_adjacency(V0, TF)
    nbr, nbw, nbm = padded_neighbours(Wc)
    ctrl = farthest_point(V0, min(NCTRL, n))
    sk_i, sk_w, sk_sig = skinning(V0, ctrl, min(CTRL_K, len(ctrl)))
    A = anchor_operator(n, tri, bw)
    AtA = (A.T @ A).tocsr()
    print(f"edges {len(E)} | ARAP max degree {nbr.shape[1]} | ctrl {len(ctrl)} "
          f"| skinning sigma {sk_sig:.2f}mm", flush=True)

    use = tr_idx if LIMIT <= 0 else tr_idx[:LIMIT]
    assert set(use.tolist()) <= set(tr_idx.tolist()), "LEAKAGE: registering a non-training ear"
    if LIMIT > 0:
        print(f"!! LIMIT={LIMIT}: SMOKE TEST ONLY, the mean template and PCA are meaningless",
              flush=True)

    # ---- registration rounds ----
    factors = prefactor(n, LtL, AtA)
    V_rest = V0
    corr = np.zeros((len(use), n, 3), np.float64)
    stats = []
    for rd in range(ROUNDS):
        LtL_rest = [lam * (LtL @ V_rest) for lam in LAMBDAS]
        t0 = time.time()
        nsup = np.zeros(n, np.int64)              # per-vertex count of ears that MATCHED it
        for t, ear in enumerate(use):
            pos = int(np.where(tr_idx == ear)[0][0])
            cloud = (cl_tr[pos] - c0[ear]) @ R[ear].T
            lms = gt_can[pos]                     # train-fold GT: allowed as a registration anchor
            nrm = local_normals(cloud, KNRM)
            tree = cKDTree(cloud)
            init = (TPS3D().fit(transport(tri, bw, V_rest), lms).transform(V_rest)
                    if rd == 0 else corr[t])
            V, st, dnn = register(init, cloud, factors, LtL_rest, A.T @ lms, tree, nrm)
            nsup += dnn <= MAX_DIST
            st["anchor_mean"] = float(np.linalg.norm(transport(tri, bw, V) - lms, axis=1).mean())
            st["anchor_max"] = float(np.linalg.norm(transport(tri, bw, V) - lms, axis=1).max())
            st["ear"] = int(ear)
            corr[t] = V
            if rd == ROUNDS - 1:
                stats.append(st)
            if (t + 1) % 20 == 0 or t + 1 == len(use):
                print(f"  round{rd} {t+1}/{len(use)} ears ({time.time()-t0:.0f}s)", flush=True)
        V_rest = corr.mean(0)                     # fold-safe mean template for the next round

    mean_V = corr.mean(0)
    spread = np.linalg.norm(corr - mean_V, axis=2)

    # ---- fold-safe dense PCA over the training-fold registrations ----
    X = corr.reshape(len(use), -1) - mean_V.reshape(-1)
    K = int(min(NBASIS, max(len(use) - 1, 1)))
    ev, U = np.linalg.eigh(X @ X.T / max(len(use) - 1, 1))
    o = np.argsort(ev)[::-1][:K]
    ev, U = np.maximum(ev[o], 1e-12), U[:, o]
    comps = U.T @ X
    comps = comps / np.maximum(np.linalg.norm(comps, axis=1, keepdims=True), 1e-12)
    var = float((X ** 2).sum() / max(len(use) - 1, 1))

    # ---- report ----
    print(f"\nregistration residuals over {len(use)} training ears "
          f"(point-to-plane |.| unless noted)")
    for k in ("p2plane_mean", "p2plane_p90", "p2plane_p99", "nn_mean", "anchor_mean",
              "anchor_max", "rejected_frac"):
        v = np.array([st[k] for st in stats])
        print(f"  {k:14s} mean {v.mean():7.4f}  median {np.median(v):7.4f}  "
              f"p90 {np.percentile(v, 90):7.4f}  max {v.max():7.4f}")
    worst = np.argsort([st["p2plane_p90"] for st in stats])[::-1][:5]
    print("  worst ears by p2plane_p90: " +
          ", ".join(f"ear{stats[i]['ear']}:{stats[i]['p2plane_p90']:.2f}" for i in worst))
    # A vertex the target cloud does not cover is DAMPED (C_i = V_i), never solved, so its
    # correspondence is a smooth extrapolation and it drags the mean template and the PCA
    # basis with it. Shipped as valid_frac so a consumer can down-weight or crop, and printed
    # because it is the honest explanation of the p2plane_p99 tail above.
    vf = nsup / len(use)
    print(f"vertex cloud support: {float((vf < 1).mean())*100:.2f}% of the {n} vertices "
          f"unmatched on >=1 ear, {float((vf < 0.5).mean())*100:.2f}% on most ears "
          f"(min {vf.min():.2f}) -- their correspondence is DAMPED, not solved (valid_frac)")
    print(f"correspondence spread around the fold mean: mean {spread.mean():.3f}mm  "
          f"p90 {np.percentile(spread, 90):.3f}mm  "
          f"(fully-supported vertices only: mean {spread[:, vf == 1].mean():.3f}mm)")
    print(f"dense PCA: {K} comps, total vertex variance {var:.1f}mm^2, "
          f"leading eigenvalues {np.round(ev[:min(5,K)], 2).tolist()}")

    # train_ear_mask is the fold-safety PROOF the consumers assert on (train_family.py
    # refuses any artefact that cannot show it): True exactly for the ears that built this
    # file, so a validation ear leaking in is detectable downstream, not just here.
    mask = np.zeros(NE, bool); mask[use] = True
    assert not mask[va_idx].any(), "LEAKAGE: a validation ear contributed to the template"
    out = dict(fold=np.int64(FOLD), train_ear_mask=mask,
               n_ears_train=len(tr_idx), n_ears_used=len(use),
               train_ear=tr_idx.astype(np.int32), val_ear=va_idx.astype(np.int32),
               used_ear=use.astype(np.int32), seed_ear=np.int32(seed), seed_tag=seed_tag,
               template_V=V0.astype(np.float32), template_F=TF.astype(np.int32),
               bary_f=np.asarray(bf).astype(np.int32), bary_w=bw.astype(np.float32),
               bary_tri=tri.astype(np.int32),
               valid_frac=vf.astype(np.float32),
               mean_V=mean_V.astype(np.float32), comps=comps.astype(np.float32),
               eig=ev.astype(np.float32), edges=E.astype(np.int32),
               nbr=nbr, nbr_w=nbw, nbr_mask=nbm,
               ctrl_idx=ctrl.astype(np.int32), skin_idx=sk_i, skin_w=sk_w,
               skin_sigma=np.float32(sk_sig),
               resid=json.dumps(stats),
               config=json.dumps(dict(lambdas=LAMBDAS, iters_per=ITERS_PER, mu=MU,
                                      max_dist=MAX_DIST, knrm=KNRM, nctrl=NCTRL,
                                      ctrl_k=CTRL_K, rounds=ROUNDS, limit=LIMIT)))
    if STORE_CORR:
        out["corr_V"] = corr.astype(np.float32)
    np.savez_compressed(OUT, **out)
    print(f"\nsaved {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB) in {time.time()-t_all:.0f}s")


if __name__ == "__main__":
    main()
