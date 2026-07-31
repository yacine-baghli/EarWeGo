"""
DENSE-CORRESPONDENCE / DENSE-SSM architecture (GPU).

STAGE 1 (corr):  landmark-anchored non-rigid ICP of ONE template onto every TRAIN
                 ear -> dense anatomical correspondence. GT landmarks are used as
                 anchors here (train only), so correspondence is anatomically right.
STAGE 2 (ssm):   GPA-align the deformed instances, PCA -> dense-vertex shape model
                 in which the 85 landmarks are FIXED barycentric points.
STAGE 3 (fit):   fit that model to each VAL ear's surface with NO landmarks
                 (similarity + shape coefficients). Thousands of surface points
                 over-determine each landmark's along-contour position -- the
                 mechanism that plain per-point detection lacks.

    STAGE=corr|ssm|fit python3 gpu_dense.py
"""
import os, time
import numpy as np
import torch

DATA = os.environ.get("DATA", "/home/ubuntu/ear/corr_data.npz")
WORK = os.environ.get("WORK", "/home/ubuntu/ear")
STAGE = os.environ.get("STAGE", "corr")
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)

d = np.load(DATA, allow_pickle=True)
TV = torch.tensor(d["template_V"]).float().to(dev)          # (n,3)
TFa = torch.tensor(d["template_F"].astype(np.int64)).to(dev)
BF = torch.tensor(d["bary_f"].astype(np.int64)).to(dev)     # (85,)
BW = torch.tensor(d["bary_w"]).float().to(dev)              # (85,3)
clouds = d["clouds"]; gt_lms = d["gt_lms"]; init_lms = d["init_lms"]; split = d["split"]
N, P, _ = clouds.shape
n = TV.shape[0]
tr_idx = np.where(split == "train")[0]; va_idx = np.where(split == "val")[0]
_lim = int(os.environ.get("LIMIT", "0"))            # >0 = smoke-test subset
if _lim:
    tr_idx = tr_idx[:_lim]; va_idx = va_idx[:_lim]
print(f"dev {dev} | template {n} verts | {len(tr_idx)} train / {len(va_idx)} val ears | {P} pts/ear")


# ---------------------------------------------------------------- helpers
def build_laplacian(F, n):
    e = torch.cat([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 0)
    e = torch.cat([e, e.flip(1)], 0)
    e = torch.unique(e, dim=0)
    i, j = e[:, 0], e[:, 1]
    vals = torch.ones(len(i), device=F.device)
    A = torch.sparse_coo_tensor(torch.stack([i, j]), vals, (n, n)).coalesce()
    deg = torch.sparse.sum(A, 1).to_dense()
    idx = torch.arange(n, device=F.device)
    D = torch.sparse_coo_tensor(torch.stack([idx, idx]), deg, (n, n))
    return (D - A).coalesce()


L = build_laplacian(TFa, n)
TRI = TFa[BF]                                              # (85,3) landmark triangles


def transport(V):
    """barycentric landmark positions on (B,n,3) or (n,3)"""
    if V.dim() == 2:
        return (BW[..., None] * V[TRI]).sum(1)
    return (BW[None, ..., None] * V[:, TRI]).sum(2)


def nearest(V, C, chunk=4096):
    """for each row of V (B,n,3) the closest point in C (B,m,3) -> (B,n,3)"""
    out = torch.empty_like(V)
    for s in range(0, V.shape[1], chunk):
        v = V[:, s:s + chunk]
        dist = torch.cdist(v, C)                            # (B,c,m)
        idx = dist.argmin(-1)
        out[:, s:s + chunk] = torch.gather(C, 1, idx[..., None].expand(-1, -1, 3))
    return out


def lap_of(V):
    if V.dim() == 2:
        return torch.sparse.mm(L, V)
    return torch.stack([torch.sparse.mm(L, V[b]) for b in range(V.shape[0])])


def similarity_init(src_lms, dst_lms, Vsrc):
    """rigid+scale transform mapping src landmarks onto dst landmarks, applied to Vsrc"""
    out = []
    for b in range(dst_lms.shape[0]):
        A = src_lms - src_lms.mean(0); B = dst_lms[b] - dst_lms[b].mean(0)
        U, S, Vt = torch.linalg.svd(A.T @ B)
        R = U @ Vt
        if torch.det(R) < 0:
            U[:, -1] *= -1; R = U @ Vt
        s = (B * (A @ R)).sum() / (A * A).sum()
        out.append(s * ((Vsrc - src_lms.mean(0)) @ R) + dst_lms[b].mean(0))
    return torch.stack(out)


# ================================================================ STAGE 1
if STAGE == "corr":
    B = int(os.environ.get("BATCH", "4"))
    OUTER = int(os.environ.get("OUTER", "26"))     # correspondence updates
    INNER = int(os.environ.get("INNER", "22"))     # Adam steps per update
    LAM = np.geomspace(60.0, 0.6, OUTER)           # Laplacian stiffness anneal
    W_LM = float(os.environ.get("W_LM", "12.0"))   # landmark anchor weight
    deformed = np.zeros((len(tr_idx), n, 3), np.float32)
    t0 = time.time()
    for bs in range(0, len(tr_idx), B):
        ids = tr_idx[bs:bs + B]
        C = torch.tensor(clouds[ids]).float().to(dev)
        A = torch.tensor(gt_lms[ids]).float().to(dev)               # anchors (train only)
        I = torch.tensor(init_lms[ids]).float().to(dev)
        V0 = similarity_init(transport(TV), I, TV)                  # (b,n,3) rigid init
        V = V0.clone().requires_grad_(True)
        L0 = lap_of(V0).detach()
        opt = torch.optim.Adam([V], lr=0.05)
        for o in range(OUTER):
            with torch.no_grad():
                tgt = nearest(V.detach(), C)
            lam = float(LAM[o])
            for _ in range(INNER):
                opt.zero_grad()
                e_data = ((V - tgt) ** 2).sum(-1).mean()
                e_lap = ((lap_of(V) - L0) ** 2).sum(-1).mean()
                e_lm = ((transport(V) - A) ** 2).sum(-1).mean()
                (e_data + lam * e_lap + W_LM * e_lm).backward()
                opt.step()
        with torch.no_grad():
            fin = nearest(V.detach(), C)
            sd = (V.detach() - fin).norm(dim=-1).mean().item()
            lm = (transport(V.detach()) - A).norm(dim=-1).mean().item()
        deformed[bs:bs + len(ids)] = V.detach().cpu().numpy()
        print(f"  ears {bs:3d}-{bs+len(ids)-1:3d}  surf {sd:.3f}mm  anchor {lm:.3f}mm  "
              f"({time.time()-t0:.0f}s)", flush=True)
    np.save(f"{WORK}/deformed_train.npy", deformed)
    print(f"saved deformed_train.npy {deformed.shape}")

# ================================================================ STAGE 2
elif STAGE == "ssm":
    Vs = torch.tensor(np.load(f"{WORK}/deformed_train.npy")).float().to(dev)   # (M,n,3)
    M = Vs.shape[0]
    ref = TV.clone()
    for it in range(3):                                    # GPA to a running mean
        al = []
        for b in range(M):
            A = Vs[b] - Vs[b].mean(0); Bb = ref - ref.mean(0)
            U, S, Vt = torch.linalg.svd(A.T @ Bb)
            R = U @ Vt
            if torch.det(R) < 0:
                U[:, -1] *= -1; R = U @ Vt
            s = (Bb * (A @ R)).sum() / (A * A).sum()
            al.append(s * A @ R + ref.mean(0))
        Al = torch.stack(al)
        ref = Al.mean(0)
        print(f"  GPA iter {it}: mean drift {(Al.mean(0)-ref).norm(dim=-1).mean():.4f}")
    X = Al.reshape(M, -1)
    mu = X.mean(0)
    Xc = X - mu
    # PCA via the Gram trick (M x M)
    G = (Xc @ Xc.T) / (M - 1)
    ev, U = torch.linalg.eigh(G.double())
    ev = ev.flip(0).float(); U = U.flip(1).float()
    K = int(os.environ.get("K", "120"))
    K = min(K, M - 1)
    comps = (U[:, :K].T @ Xc)                              # (K, n*3)
    comps = comps / comps.norm(dim=1, keepdim=True)
    lam = ev[:K].clamp(min=1e-8)
    var = (lam / ev.clamp(min=0).sum()).cumsum(0)
    print(f"  SSM: K={K}, variance explained {var[-1]*100:.2f}% "
          f"(K=30 {var[min(29,K-1)]*100:.1f}%, K=60 {var[min(59,K-1)]*100:.1f}%)")
    np.savez(f"{WORK}/dense_ssm.npz", mean=mu.cpu().numpy(), comps=comps.cpu().numpy(),
             eig=lam.cpu().numpy(), ref=ref.cpu().numpy())
    # reconstruction check on train
    coef = Xc @ comps.T
    rec = mu + coef @ comps
    err = (rec - X).reshape(M, n, 3).norm(dim=-1).mean().item()
    print(f"  train dense reconstruction err {err:.4f}mm")
    lm_rec = transport(rec.reshape(M, n, 3))
    lm_true = transport(Al)
    print(f"  train LANDMARK reconstruction err "
          f"{(lm_rec-lm_true).norm(dim=-1).mean().item():.4f}mm")

# ================================================================ STAGE 3
elif STAGE == "fit3":
    # HYBRID closed-form fit: the dense SSM is fitted to BOTH the target surface
    # (dense, sub-mm, but correspondence-ambiguous) AND the deep model's predicted
    # landmarks (good correspondence, noisy/off-surface). Each fixes the other's
    # weakness. Exact normal equations in the K-dim coefficient space.
    S = np.load(f"{WORK}/dense_ssm.npz")
    mu = torch.tensor(S["mean"]).float().to(dev)
    comps = torch.tensor(S["comps"]).float().to(dev)
    eig = torch.tensor(S["eig"]).float().to(dev)
    KUSE = int(os.environ.get("KUSE", "120"))
    comps = comps[:KUSE].contiguous(); eig = eig[:KUSE]
    SIG = [float(x) for x in os.environ.get("SIG", "9,4,1,0.36,0.16,0.09").split(",")]
    ITERS = int(os.environ.get("ITERS", "4"))
    CUT = float(os.environ.get("CUT", "4.0"))
    W_LM = float(os.environ.get("W_LM", "1.0"))       # trust in the deep landmarks
    mean_V = mu.reshape(n, 3)
    # landmark response of each component: Clm[k] = transported landmarks of comp k
    Clm = torch.stack([transport(comps[k].reshape(n, 3)).reshape(-1) for k in range(KUSE)])
    lm_mean = transport(mean_V).reshape(-1)                       # (255,)
    G_lm = Clm @ Clm.T                                            # (K,K)
    Iden = torch.eye(KUSE, device=dev)
    invlam = 1.0 / eig

    def wprocrustes(A, B, w):
        wn = w / w.sum().clamp(min=1e-9)
        ca = (wn[:, None] * A).sum(0); cb = (wn[:, None] * B).sum(0)
        A0, B0 = A - ca, B - cb
        H = (wn[:, None] * A0).T @ B0
        U_, S_, Vt_ = torch.linalg.svd(H)
        R = U_ @ Vt_
        if torch.det(R) < 0:
            U_ = U_.clone(); U_[:, -1] *= -1; R = U_ @ Vt_
        s = (wn[:, None] * B0 * (A0 @ R)).sum() / (wn[:, None] * A0 * A0).sum().clamp(min=1e-9)
        return s, R, cb - s * (ca @ R)

    ids = va_idx
    got = np.zeros((len(ids), 85, 3), np.float32)
    t0 = time.time()
    for k, i in enumerate(ids):
        C = torch.tensor(clouds[i:i+1]).float().to(dev)
        Ld = torch.tensor(init_lms[i]).float().to(dev)            # deep predictions
        c = torch.zeros(KUSE, device=dev)
        s, R, t = wprocrustes(transport(mean_V), Ld, torch.ones(85, device=dev))
        for sig2 in SIG:
            for _ in range(ITERS):
                V = (mu + c @ comps).reshape(n, 3)
                Vw = s * (V @ R) + t
                tgt = nearest(Vw[None], C)[0]
                dist = (Vw - tgt).norm(dim=-1)
                w = (dist < CUT).float()
                if w.sum() < 100:
                    w = torch.ones_like(w)
                # similarity from BOTH surface correspondences and the deep landmarks
                Asrc = torch.cat([V, transport(V)], 0)
                Bdst = torch.cat([tgt, Ld], 0)
                wcat = torch.cat([w, W_LM * 20.0 * torch.ones(85, device=dev)], 0)
                s, R, t = wprocrustes(Asrc, Bdst, wcat)
                # coefficients: exact normal equations (surface + landmark + prior)
                tgt_m = ((tgt - t) @ R.T) / s
                Ld_m = ((Ld - t) @ R.T) / s
                A = Iden / sig2 + W_LM * G_lm + torch.diag(invlam)
                b = (comps @ (tgt_m.reshape(-1) - mu)) / sig2 \
                    + W_LM * (Clm @ (Ld_m.reshape(-1) - lm_mean))
                c = torch.linalg.solve(A, b)
        with torch.no_grad():
            V = (mu + c @ comps).reshape(n, 3)
            Vw = s * (V @ R) + t
            got[k] = transport(Vw).cpu().numpy()
            sd = (Vw - nearest(Vw[None], C)[0]).norm(dim=-1).mean().item()
        e = np.linalg.norm(got[k] - gt_lms[i], axis=1).mean()
        e0 = np.linalg.norm(init_lms[i] - gt_lms[i], axis=1).mean()
        print(f"  val {k:2d}: surf {sd:.3f} | hybrid {e:.3f} | deep {e0:.3f}mm "
              f"({time.time()-t0:.0f}s)", flush=True)
    np.savez(f"{WORK}/val_hybrid_fit.npz", pred=got, gt=gt_lms[ids], init=init_lms[ids])
    E = np.linalg.norm(got - gt_lms[ids], axis=2)
    E0 = np.linalg.norm(init_lms[ids] - gt_lms[ids], axis=2)
    print(f"\nW_LM={W_LM} KUSE={KUSE} SIGend={SIG[-1]}")
    print(f"OVERALL  hybrid {E.mean():.4f}mm   deep {E0.mean():.4f}mm")
    for lo, hi, nm in [(0, 24, "Helix"), (25, 54, "Antihelix"), (55, 74, "Concha"), (75, 84, "Lobe")]:
        print(f"  {nm:10s} {E[:, lo:hi+1].mean():.3f}  (deep {E0[:, lo:hi+1].mean():.3f})")

elif STAGE == "fit2":
    # Closed-form alternating fit (NO Adam): the model is LINEAR in the shape
    # coefficients and the PCA components are orthonormal, so the MAP coefficients
    # are exact:  c_k = <v_k, T-mean> / (1 + sigma^2/lambda_k).
    # Alternate: correspondences -> weighted similarity (Procrustes) -> coefficients,
    # annealing sigma^2 from stiff (near mean shape) to flexible.
    S = np.load(f"{WORK}/dense_ssm.npz")
    mu = torch.tensor(S["mean"]).float().to(dev)
    comps = torch.tensor(S["comps"]).float().to(dev)
    eig = torch.tensor(S["eig"]).float().to(dev)
    KUSE = int(os.environ.get("KUSE", "120"))
    comps = comps[:KUSE].contiguous(); eig = eig[:KUSE]
    SIG = [float(x) for x in os.environ.get("SIG", "9,4,1,0.36,0.16,0.09,0.06").split(",")]
    ITERS = int(os.environ.get("ITERS", "4"))
    CUT = float(os.environ.get("CUT", "4.0"))
    mean_V = mu.reshape(n, 3)

    def wprocrustes(A, B, w):
        """similarity (s,R,t) mapping A -> B with weights w"""
        wn = w / w.sum().clamp(min=1e-9)
        ca = (wn[:, None] * A).sum(0); cb = (wn[:, None] * B).sum(0)
        A0, B0 = A - ca, B - cb
        H = (wn[:, None] * A0).T @ B0
        U_, S_, Vt_ = torch.linalg.svd(H)
        R = U_ @ Vt_
        if torch.det(R) < 0:
            U_ = U_.clone(); U_[:, -1] *= -1; R = U_ @ Vt_
        s = (wn[:, None] * B0 * (A0 @ R)).sum() / (wn[:, None] * A0 * A0).sum().clamp(min=1e-9)
        return s, R, cb - s * (ca @ R)

    ids = va_idx
    got = np.zeros((len(ids), 85, 3), np.float32)
    t0 = time.time()
    for k, i in enumerate(ids):
        C = torch.tensor(clouds[i:i+1]).float().to(dev)
        I = torch.tensor(init_lms[i]).float().to(dev)
        c = torch.zeros(KUSE, device=dev)
        # init similarity: model landmarks -> deep-model predicted landmarks
        s, R, t = wprocrustes(transport(mean_V), I, torch.ones(85, device=dev))
        for sig2 in SIG:
            for _ in range(ITERS):
                V = (mu + c @ comps).reshape(n, 3)
                Vw = s * (V @ R) + t
                tgt = nearest(Vw[None], C)[0]
                dist = (Vw - tgt).norm(dim=-1)
                w = (dist < CUT).float()
                if w.sum() < 100:
                    w = torch.ones_like(w)
                s, R, t = wprocrustes(V, tgt, w)
                tgt_m = ((tgt - t) @ R.T) / s                 # targets in model space
                proj = comps @ (tgt_m.reshape(-1) - mu)
                c = proj / (1.0 + sig2 / eig)
        with torch.no_grad():
            V = (mu + c @ comps).reshape(n, 3)
            Vw = s * (V @ R) + t
            got[k] = transport(Vw).cpu().numpy()
            sd = (Vw - nearest(Vw[None], C)[0]).norm(dim=-1).mean().item()
        e = np.linalg.norm(got[k] - gt_lms[i], axis=1).mean()
        e0 = np.linalg.norm(init_lms[i] - gt_lms[i], axis=1).mean()
        print(f"  val {k:2d}: surf {sd:.3f}mm | SSM-fit {e:.3f}mm | deep {e0:.3f}mm "
              f"({time.time()-t0:.0f}s)", flush=True)
    np.savez(f"{WORK}/val_ssm_fit.npz", pred=got, gt=gt_lms[ids], init=init_lms[ids])
    E = np.linalg.norm(got - gt_lms[ids], axis=2)
    E0 = np.linalg.norm(init_lms[ids] - gt_lms[ids], axis=2)
    print(f"\nOVERALL  SSM-fit {E.mean():.4f}mm   deep {E0.mean():.4f}mm")
    for lo, hi, nm in [(0, 24, "Helix"), (25, 54, "Antihelix"), (55, 74, "Concha"), (75, 84, "Lobe")]:
        print(f"  {nm:10s} {E[:, lo:hi+1].mean():.3f}  (deep {E0[:, lo:hi+1].mean():.3f})")

elif STAGE == "fit":
    S = np.load(f"{WORK}/dense_ssm.npz")
    mu = torch.tensor(S["mean"]).float().to(dev)
    comps = torch.tensor(S["comps"]).float().to(dev)
    eig = torch.tensor(S["eig"]).float().to(dev)
    Kuse = int(os.environ.get("KUSE", "60"))
    comps = comps[:Kuse]; eig = eig[:Kuse]
    PRIOR = float(os.environ.get("PRIOR", "0.02"))
    SCALE_W = float(os.environ.get("SCALE_W", "2.0"))     # keeps scale near the init
    COVER_W = float(os.environ.get("COVER_W", "1.0"))     # target->model coverage
    COVER_CUT = float(os.environ.get("COVER_CUT", "3.0")) # mm; drops head surface in margin
    OUTER = int(os.environ.get("OUTER", "30")); INNER = int(os.environ.get("INNER", "25"))
    ids = va_idx
    got = np.zeros((len(ids), 85, 3), np.float32)
    t0 = time.time()
    for k, i in enumerate(ids):
        C = torch.tensor(clouds[i:i+1]).float().to(dev)
        I = torch.tensor(init_lms[i:i+1]).float().to(dev)
        mean_V = mu.reshape(n, 3)
        # init similarity from the model's landmarks onto the deep-model prediction
        V_init = similarity_init(transport(mean_V), I, mean_V)[0]
        # parametrize: coeffs + similarity (axis-angle, scale, translation)
        c = torch.zeros(Kuse, device=dev, requires_grad=True)
        w = torch.zeros(3, device=dev, requires_grad=True)
        ls = torch.zeros(1, device=dev, requires_grad=True)
        t = (V_init.mean(0) - mean_V.mean(0)).clone().requires_grad_(True)
        # initial rotation/scale from the similarity init (absorbed via a fixed R0,s0)
        A0 = mean_V - mean_V.mean(0); B0 = V_init - V_init.mean(0)
        U_, S_, Vt_ = torch.linalg.svd(A0.T @ B0)
        R0 = (U_ @ Vt_).detach()
        if torch.det(R0) < 0:
            U_[:, -1] *= -1; R0 = (U_ @ Vt_).detach()
        s0 = ((B0 * (A0 @ R0)).sum() / (A0 * A0).sum()).detach()
        opt = torch.optim.Adam([c, w, ls, t], lr=0.02)

        def model():
            V = (mu + c @ comps).reshape(n, 3)
            th = w.norm() + 1e-9
            K_ = torch.zeros(3, 3, device=dev)
            kx, ky, kz = w / th
            K_ = torch.stack([torch.stack([torch.zeros_like(kx), -kz, ky]),
                              torch.stack([kz, torch.zeros_like(kx), -kx]),
                              torch.stack([-ky, kx, torch.zeros_like(kx)])])
            Rr = torch.eye(3, device=dev) + torch.sin(th) * K_ + (1 - torch.cos(th)) * (K_ @ K_)
            cen = V.mean(0)
            return s0 * torch.exp(ls) * ((V - cen) @ (R0 @ Rr)) + cen + t

        for o in range(OUTER):
            with torch.no_grad():
                Vc = model()[None]
                tgt = nearest(Vc, C)[0]
            with torch.no_grad():
                # coverage: target points the model should explain (robust cutoff drops
                # the extraneous head surface inside the crop margin)
                Vc2 = model()[None]
                dsub = torch.cdist(C[:, ::4], Vc2)                  # (1,m/4,n)
                dmin, amin = dsub.min(-1)
                keep = (dmin[0] < COVER_CUT)
                cov_t = C[0, ::4][keep]                             # target pts to cover
            for _ in range(INNER):
                opt.zero_grad()
                V = model()
                e_data = ((V - tgt) ** 2).sum(-1).mean()
                e_prior = (c ** 2 / eig).sum()
                e_scale = (ls ** 2).sum()                           # prevent scale collapse
                loss = e_data + PRIOR * e_prior + SCALE_W * e_scale
                if COVER_W > 0 and len(cov_t) > 0:
                    dc = torch.cdist(cov_t[None], V[None]).min(-1).values   # (1,k)
                    loss = loss + COVER_W * (dc ** 2).mean()
                loss.backward()
                opt.step()
        with torch.no_grad():
            V = model()
            got[k] = transport(V).cpu().numpy()
            sd = (V[None] - nearest(V[None], C)).norm(dim=-1).mean().item()
        e = np.linalg.norm(got[k] - gt_lms[i], axis=1).mean()
        e0 = np.linalg.norm(init_lms[i] - gt_lms[i], axis=1).mean()
        print(f"  val {k:2d}: surf {sd:.3f}mm | SSM-fit {e:.3f}mm | deep-model {e0:.3f}mm "
              f"({time.time()-t0:.0f}s)", flush=True)
    np.savez(f"{WORK}/val_ssm_fit.npz", pred=got, gt=gt_lms[ids], init=init_lms[ids])
    E = np.linalg.norm(got - gt_lms[ids], axis=2)
    E0 = np.linalg.norm(init_lms[ids] - gt_lms[ids], axis=2)
    print(f"\nOVERALL  SSM-fit {E.mean():.4f}mm   deep-model {E0.mean():.4f}mm")
    for lo, hi, nm in [(0, 24, "Helix"), (25, 54, "Antihelix"), (55, 74, "Concha"), (75, 84, "Lobe")]:
        print(f"  {nm:10s} {E[:, lo:hi+1].mean():.3f}  (deep {E0[:, lo:hi+1].mean():.3f})")
