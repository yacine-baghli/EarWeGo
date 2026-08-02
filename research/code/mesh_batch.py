"""
RAGGED MESH -> PADDED BATCH, on the GPU, with no mesh library and no per-step numpy.

Why this file exists. build_mesh_data.py emits ONE flat block-diagonal concatenation of
340 ears (v_ptr/deg_ptr offsets, GLOBAL indices) because padding 128 eigenvectors to
Vmax would store ~1 GB of zeros. fam_diffusionnet.py, on the other hand, was written
against a PADDED (B,Vmax,...) contract with different key names and two tensors
(`vfan`, `vfan_mask`) that build_mesh_data.py never emitted at all. Something has to
bridge the two, per training step, for a random subset of ears, fast enough that it is
not the bottleneck. That is this module, and both mesh families share it so the bridge
is written and tested once.

HOW. The whole ragged artefact is uploaded ONCE as flat GPU tensors (~0.7 GB without
eigenpairs, +0.96 GB of float16 eigenvectors with them). A batch is then pure index
arithmetic on the device:

    g[b, v]      = v_ptr[e_b] + v                      global vertex id
    fi[b, v, d]  = deg_ptr[g] + d      (d < deg)       global one-ring slot
    xi[b, v, t]  = fan_ptr[g] + t      (t < fdeg)      global incident-triangle slot

so there is no host->device copy and no numpy in the training loop. Cost measured in the
smoke test below.

PADDING CONVENTION (identical to build_mesh_data.pad_batch, which this replaces):
  * `vmask` is authoritative. Every per-vertex float array is multiplied by it, so
    `mass` and `evecs` are EXACTLY 0 on padding -- that is the single invariant the
    spectral transform's correctness rests on, and it is asserted, not assumed.
  * a padded one-ring slot points at the vertex ITSELF and carries weight 0, so every
    operator stays a pure gather-sum with no scatter (there is no torch_scatter here).
  * a padded incident-triangle slot is the degenerate (i,i,i) and `fan_mask` is False.
  * neighbour/fan indices are made LOCAL to the padded row (global - v_ptr[e]).

THE FAN. `fan1`/`fan2` give, for each vertex i, the OTHER TWO corners (in winding order)
of each triangle incident to i. It is a CSR group-by of the 3*Ftot corner records and is
built once at load (~2 s, ~180 MB). It is what makes a heatmap head land EXACTLY on the
surface: the decoded point is a convex barycentric combination of a real triangle.

NO (B,V,T,2) OR (B,V,2,3) TENSORS -- deliberately. train_family._flatten_samples squeezes
axis 1 of EVERY batch tensor with ndim >= 4 when cls.SAMPLES == 1, which is every family
here. A (B,Vmax,T,2) fan would come out of it as (B,T,2) with no error and no shape
assert anywhere downstream. So the fan ships as two (B,V,T) tensors and the tangent basis
as basis_x/basis_y, never as a stacked axis. This is not style; it is the bug.

AUGMENTATION. `mesh_augment` is a per-ear ROTATION (plus optional isotropic scale) that
moves the mesh, the tangent frames, the cloud, the coarse init and the target TOGETHER.
train_family.default_augment cannot be used: it would rotate `pc`, `coarse` and the
target and leave `verts` alone, which every shape check passes and which trains the model
against an inconsistent frame. Scale is exact rather than approximate -- under p -> s p,
  mass -> s^2 mass,  grad_* -> grad_*/s,  evals -> evals/s^2,  evecs -> evecs/s,
  lap_w / lap_diag  UNCHANGED (cotangents are scale-invariant)
-- and the smoke test verifies the eigenvalue and gradient scaling numerically. It is
still OFF by default (aug_scale=0) because a scaled ear is not an ear.

ENV
  MESH_DATA  scratch/mesh_data.npz   ragged artefact from build_mesh_data.py
  MESH_SPEC  scratch/mesh_spec.npz   eigenpairs; only read when keig > 0
  MESH_KEIG  0                       eigenpairs to load (0 = do not touch MESH_SPEC)

    python research/code/mesh_batch.py           # CPU smoke test on a synthetic artefact
"""
import os, time
import numpy as np
import torch

MESH_DATA = os.environ.get("MESH_DATA", "scratch/mesh_data.npz")
MESH_SPEC = os.environ.get("MESH_SPEC", "scratch/mesh_spec.npz")
MESH_KEIG = int(os.environ.get("MESH_KEIG", "0"))

VKEYS = ("verts", "nrm", "basis_x", "basis_y")          # (V,3) per-vertex vectors
SKEYS = ("mass", "lap_diag", "grad_xd", "grad_yd")      # (V,)  per-vertex scalars
EKEYS = ("lap_w", "grad_x", "grad_y")                   # (nnz,) per-one-ring-slot
ROT_V = ("verts", "nrm", "basis_x", "basis_y")          # rotate as 3-vectors
_STORE = {}


def build_fan(faces, v_ptr):
    """CSR group-by: for each vertex, the other two corners of each incident triangle.

    faces are GLOBAL, so the grouping is global too and a per-ear slice needs no remap.
    Winding is preserved -- (i, fan1, fan2) is the incident triangle in its original
    orientation -- which keeps a barycentric decode consistent with the surface normal.
    """
    a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
    src = np.concatenate([a, b, c])
    p1 = np.concatenate([b, c, a]).astype(np.int32)
    p2 = np.concatenate([c, a, b]).astype(np.int32)
    o = np.argsort(src, kind="stable")
    nv = int(v_ptr[-1])
    ptr = np.concatenate([[0], np.cumsum(np.bincount(src, minlength=nv))]).astype(np.int64)
    return ptr, p1[o], p2[o]


class MeshStore:
    """The whole ragged artefact resident on `dev`, plus device-side padding."""

    def __init__(self, path=None, dev="cpu", keig=0, spec=None):
        self.path, self.dev, self.keig = path or MESH_DATA, dev, keig
        z = np.load(self.path)
        T = lambda a: torch.as_tensor(a).to(dev)
        self.v_ptr = T(z["v_ptr"].astype(np.int64))
        self.deg_ptr = T(z["deg_ptr"].astype(np.int64))
        self.ne = len(self.v_ptr) - 1
        self.nv = self.v_ptr[1:] - self.v_ptr[:-1]
        self.deg = (self.deg_ptr[1:] - self.deg_ptr[:-1]).int()
        for k in VKEYS + SKEYS + EKEYS:
            setattr(self, k, T(z[k]))
        self.nbr = T(z["nbr"].astype(np.int32))
        fptr, f1, f2 = build_fan(z["faces"].astype(np.int64), z["v_ptr"])
        self.fan_ptr, self.fan1, self.fan2 = T(fptr), T(f1), T(f2)
        self.fdeg = (self.fan_ptr[1:] - self.fan_ptr[:-1]).int()
        assert int(self.fdeg.min()) > 0, \
            "a vertex has no incident triangle: barycentric refinement has nothing to " \
            "project onto there. build_mesh_data.py keeps only face-referenced vertices, " \
            "so this artefact is not one of its outputs."
        self.coarse_np = z["coarse"] if "coarse" in z.files else None
        self.evals = self.evecs = None
        if keig:
            s = np.load(spec or MESH_SPEC)
            assert (s["v_ptr"] == z["v_ptr"]).all(), "mesh_spec/mesh_data disagree on v_ptr"
            k = min(keig, s["evals"].shape[1])
            assert k == keig, f"MESH_SPEC has {s['evals'].shape[1]} eigenpairs, need {keig}"
            self.evals = T(s["evals"][:, :k].astype(np.float32))
            self.evecs = T(np.ascontiguousarray(s["evecs"][:, :k]))     # float16
        self.bytes = sum(t.numel() * t.element_size() for t in vars(self).values()
                         if torch.is_tensor(t))

    # ---------------------------------------------------------------- padding
    def pad(self, ears, want_spec=False):
        """ears (B,) -> dict of padded (B,Vmax,...) tensors. Pure device index arithmetic."""
        dev = self.dev
        e = torch.as_tensor(np.asarray(ears), dtype=torch.long, device=dev)
        nv = self.nv[e]
        B, P = len(e), int(nv.max())
        ar = torch.arange(P, device=dev)
        vmask = ar[None] < nv[:, None]                                   # (B,P)
        g = self.v_ptr[e][:, None] + ar[None] * vmask                    # pad -> ear's v0
        fm = vmask[..., None].float()
        out = {"vmask": vmask, "nv": nv, "ear_row": e}
        for k in VKEYS:
            out[k] = getattr(self, k)[g] * fm
        for k in SKEYS:
            out[k] = getattr(self, k)[g] * vmask                         # mass EXACTLY 0
        # one-ring axis
        deg = self.deg[g]
        D = int((deg * vmask).max())
        col = torch.arange(D, device=dev)
        nmask = (col[None, None] < deg[..., None]) & vmask[..., None]    # (B,P,D)
        fi = self.deg_ptr[g][..., None] + col[None, None] * nmask
        nb = self.nbr[fi].long() - self.v_ptr[e][:, None, None]
        out["nbr"] = torch.where(nmask, nb, ar[None, :, None].expand_as(nb))
        out["nbr_mask"] = nmask
        for k in EKEYS:
            out[k] = getattr(self, k)[fi] * nmask
        # incident-triangle fan axis
        fdeg = self.fdeg[g]
        Tt = int((fdeg * vmask).max())
        fcol = torch.arange(Tt, device=dev)
        fmask = (fcol[None, None] < fdeg[..., None]) & vmask[..., None]  # (B,P,T)
        xi = self.fan_ptr[g][..., None] + fcol[None, None] * fmask
        sf = ar[None, :, None].expand(B, P, Tt)
        for k, src in (("fan1", self.fan1), ("fan2", self.fan2)):
            v = src[xi].long() - self.v_ptr[e][:, None, None]
            out[k] = torch.where(fmask, v, sf)                           # pad -> (i,i,i)
        out["fan_mask"] = fmask
        if want_spec:
            assert self.evecs is not None, \
                "spectral tensors requested but MESH_KEIG=0 -- set MESH_KEIG before import"
            out["evals"] = self.evals[e]
            out["evecs"] = self.evecs[g].float() * fm                    # EXACTLY 0 on pad
        assert float((out["mass"] * ~vmask).abs().max()) == 0.0
        return out


def store(dev, keig=None, path=None, spec=None):
    """Process-wide singleton per (path, device, keig). Built on first use, never rebuilt."""
    keig = MESH_KEIG if keig is None else keig
    key = (path or MESH_DATA, str(dev), int(keig))
    if key not in _STORE:
        t = time.time()
        _STORE[key] = MeshStore(key[0], dev, int(keig), spec)
        print(f"[mesh_batch] {key[0]} -> {_STORE[key].ne} ears, "
              f"{int(_STORE[key].nv.sum())} vertices, keig={keig}, "
              f"{_STORE[key].bytes/1e6:.0f} MB on {dev} ({time.time()-t:.1f}s)", flush=True)
    return _STORE[key]


# ---------------------------------------------------------------------- augmentation
def rand_rot(B, maxang, gen, dev):
    ax = torch.randn(B, 3, device=dev, generator=gen)
    ax = ax / ax.norm(dim=1, keepdim=True)
    ang = (torch.rand(B, device=dev, generator=gen) - .5) * maxang
    c, s = ang.cos(), ang.sin(); x, y, z = ax[:, 0], ax[:, 1], ax[:, 2]; C = 1 - c
    return torch.stack([
        torch.stack([c + x*x*C, x*y*C - z*s, x*z*C + y*s], -1),
        torch.stack([y*x*C + z*s, c + y*y*C, y*z*C - x*s], -1),
        torch.stack([z*x*C - y*s, z*y*C + x*s, c + z*z*C], -1)], 1)


def mesh_augment(b, tg, cfg, rotates, gen):
    """Per-ear similarity applied to the MESH as well as the cloud/coarse/target.

    Rotation is free: the cotangent Laplacian, the one-ring weights and the eigenpairs are
    all rotation-invariant, and rotating (nrm, basis_x, basis_y) with the vertices keeps
    grad_x/grad_y -- which are expressed in that basis -- valid. Scale is exact but off by
    default; see the module docstring for the five tensors it touches.
    """
    dev = b["verts"].device
    B = b["verts"].shape[0]
    R = rand_rot(B, cfg.get("aug_rot", 0.0), gen, dev)
    s = 1 + (torch.rand(B, 1, 1, device=dev, generator=gen) - .5) * cfg.get("aug_scale", 0.0)
    out = dict(b)
    rot = lambda t: torch.einsum("b...j,bij->b...i", t, R)
    for k in ROT_V:
        out[k] = rot(b[k]) * (s if k == "verts" else 1.0)
    if "pc" in b:
        out["pc"] = rot(b["pc"]) * s[:, None] if b["pc"].dim() == 4 else rot(b["pc"]) * s
    out["coarse"] = rot(b["coarse"]) * s + \
        torch.randn(b["coarse"].shape, device=dev, generator=gen) * cfg.get("aug_qjit", 0.0)
    if float(cfg.get("aug_scale", 0.0)) != 0.0:
        s1 = s[:, :, 0]                                                  # (B,1)
        out["mass"] = b["mass"] * s1 ** 2
        for k in ("grad_x", "grad_y"):
            out[k] = b[k] / s1[..., None]
        for k in ("grad_xd", "grad_yd"):
            out[k] = b[k] / s1
        if "evals" in b:
            out["evals"] = b["evals"] / s1 ** 2
            out["evecs"] = b["evecs"] / s1[..., None]
    assert not rotates or set(rotates) <= set(out), \
        f"cls.ROTATES names {sorted(set(rotates) - set(out))}, absent from the mesh batch"
    return out, rot(tg) * s


# ---------------------------------------------------------------------- operators
def ring_gather(x, nbr):
    """x (B,V,C), nbr (B,V,D) -> (B,V,D,C). The one memory-heavy op in every mesh model."""
    B, V, C = x.shape
    D = nbr.shape[-1]
    return torch.gather(x, 1, nbr.reshape(B, V * D, 1).expand(-1, -1, C)).view(B, V, D, C)


def mesh_ops(x, b, xj=None):
    """(laplacian, d/de1, d/de2, ring mean, ring max) from ONE gather of x.

    build_mesh_data.py's convention, verbatim:
        (L f)_i     = lap_diag_i f_i - sum_k lap_w_ik f_k
        (grad f)_i  = grad_xd_i f_i + sum_k grad_x_ik f_k   (and the same for y)
    with grad_xd = -sum_k grad_x, so the gradient is the usual sum_k w_ik (f_k - f_i).
    Padded slots carry weight 0 and point at the vertex itself, so they contribute nothing.
    """
    xj = ring_gather(x, b["nbr"]) if xj is None else xj
    lap = b["lap_diag"][..., None] * x - (b["lap_w"][..., None] * xj).sum(2)
    gx = b["grad_xd"][..., None] * x + (b["grad_x"][..., None] * xj).sum(2)
    gy = b["grad_yd"][..., None] * x + (b["grad_y"][..., None] * xj).sum(2)
    m = b["nbr_mask"][..., None]
    mean = (xj * m).sum(2) / m.sum(2).clamp(min=1)
    mx = xj.masked_fill(~m, -1e30).max(2).values
    return lap, gx, gy, mean, torch.where(m.any(2), mx, torch.zeros_like(mx))


def ring_smooth(x, b, rounds):
    """uniform one-ring averaging, unconditionally stable, no parameters, no spectrum."""
    for _ in range(rounds):
        m = b["nbr_mask"][..., None]
        xj = ring_gather(x, b["nbr"]) * m
        x = 0.5 * x + 0.5 * xj.sum(2) / m.sum(2).clamp(min=1)
    return x * b["vmask"][..., None]


def curvature(b):
    """(k1, k2, H, K, H_lap, |Hn|) per vertex, 1/mm, from the shipped tangent operator.

    The shape operator is the tangent gradient of the unit normal read back in the same
    (basis_x, basis_y) frame the gradient weights were fitted in; it is symmetrised
    because a first-order least-squares estimate on an irregular one-ring is not exactly
    so. H_lap is the independent cotangent-Laplacian estimate (L p)/(2 area) . n, kept
    because it fails differently from the gradient estimate.
    """
    n = b["nrm"]
    xj = ring_gather(torch.cat([n, b["verts"]], -1), b["nbr"])
    _, gn, gnl, _, _ = mesh_ops(torch.cat([n, b["verts"]], -1), b, xj)
    # Weingarten map S = -d n. With an OUTWARD normal this reproduces the sign of the
    # classical graph mean curvature ((1+zy^2)zxx - ... )/(2(1+|grad z|^2)^{3/2}), i.e.
    # NEGATIVE on an outward bulge. Both signs are verified against that closed form and
    # against each other in the smoke test; getting them opposite would make the two
    # curvature features cancel in the first linear layer.
    dnx, dny = -gn[..., :3], -gnl[..., :3]
    ex, ey = b["basis_x"], b["basis_y"]
    S11 = (dnx * ex).sum(-1); S12 = (dnx * ey).sum(-1)
    S21 = (dny * ex).sum(-1); S22 = (dny * ey).sum(-1)
    Sm = 0.5 * (S12 + S21)
    H = 0.5 * (S11 + S22)
    K = S11 * S22 - Sm ** 2
    d = (H ** 2 - K).clamp(min=0).sqrt()
    lap, _, _, _, _ = mesh_ops(b["verts"], b, xj[..., 3:])
    Hn = -lap / (2 * b["mass"].clamp(min=1e-6))[..., None]      # (L p)_i = sum w (p_i - p_k)
    return H + d, H - d, H, K, (Hn * n).sum(-1), Hn.norm(dim=-1)


# ---------------------------------------------------------------------- surface decode
def bary_closest(p, A, B, C, eps=1e-9):
    """Clamped barycentric weights of the point of triangle (A,B,C) closest to p.

    Ericson's region algorithm, in the same evaluation order as
    deep_model/surfproj.closest_on_triangles, but returning the WEIGHTS: the decoded point
    is then w0 A + w1 B + w2 C with w >= 0 and sum w = 1, i.e. exactly on a real triangle,
    and the weights carry gradient back to p. A degenerate padded triangle (A=B=C) falls
    into the vertex-A region and returns (1,0,0).
    """
    ab, ac, ap = B - A, C - A, p - A
    d1 = (ab * ap).sum(-1); d2 = (ac * ap).sum(-1)
    bp = p - B
    d3 = (ab * bp).sum(-1); d4 = (ac * bp).sum(-1)
    cp = p - C
    d5 = (ab * cp).sum(-1); d6 = (ac * cp).sum(-1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    sdiv = lambda n, d: n / torch.where(d.abs() > eps, d, torch.full_like(d, eps))
    v, w = sdiv(vb, va + vb + vc), sdiv(vc, va + vb + vc)
    W = torch.stack([1 - v - w, v, w], -1)
    z, o = torch.zeros_like(v), torch.ones_like(v)
    t = sdiv(d4 - d3, (d4 - d3) + (d5 - d6)).clamp(0, 1)
    W = torch.where(((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0))[..., None],
                    torch.stack([z, 1 - t, t], -1), W)
    t = sdiv(d2, d2 - d6).clamp(0, 1)
    W = torch.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None],
                    torch.stack([1 - t, z, t], -1), W)
    t = sdiv(d1, d1 - d3).clamp(0, 1)
    W = torch.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None],
                    torch.stack([1 - t, t, z], -1), W)
    W = torch.where(((d6 >= 0) & (d5 <= d6))[..., None], torch.stack([z, z, o], -1), W)
    W = torch.where(((d3 >= 0) & (d4 <= d3))[..., None], torch.stack([z, o, z], -1), W)
    W = torch.where(((d1 <= 0) & (d2 <= 0))[..., None], torch.stack([o, z, z], -1), W)
    return W


def gather_pts(x, idx):
    """x (B,V,D), idx (B,...) long -> (B,...,D)"""
    B, _, D = x.shape
    return torch.gather(x, 1, idx.reshape(B, -1, 1).expand(-1, -1, D)).view(*idx.shape, D)


def fan_project(target, vstar, b, coords=None, out_coords=None):
    """Closest point to `target` on the incident-triangle fan of `vstar`.

    Returns (point (B,L,3), tri (B,L,3) long, weights (B,L,3), residual (B,L) mm). With
    coords = out_coords = verts this is an EXACT point-to-surface projection restricted to
    one vertex fan, so the output is a genuine barycentric point of a real triangle and
    the head is on-surface by construction, not by a post-hoc snap.
    """
    coords = b["verts"] if coords is None else coords
    out_coords = coords if out_coords is None else out_coords
    B, L = vstar.shape
    Tt = b["fan1"].shape[2]
    f1 = torch.gather(b["fan1"], 1, vstar[..., None].expand(-1, -1, Tt))
    f2 = torch.gather(b["fan2"], 1, vstar[..., None].expand(-1, -1, Tt))
    msk = torch.gather(b["fan_mask"], 1, vstar[..., None].expand(-1, -1, Tt))
    tri = torch.stack([vstar[..., None].expand(-1, -1, Tt), f1, f2], -1)      # (B,L,T,3)
    cA, cB, cC = (gather_pts(coords, tri[..., k]) for k in range(3))
    W = bary_closest(target[:, :, None, :], cA, cB, cC)
    P = W[..., 0:1] * cA + W[..., 1:2] * cB + W[..., 2:3] * cC
    d = (P - target[:, :, None, :]).norm(dim=-1).masked_fill(~msk, 1e9)
    j = d.argmin(-1)
    pick = lambda X: torch.gather(
        X, 2, j[..., None, None].expand(-1, -1, 1, X.shape[-1])).squeeze(2)
    Wb, trib = pick(W), pick(tri)
    oA, oB, oC = (gather_pts(out_coords, trib[..., k]) for k in range(3))
    pt = Wb[..., 0:1] * oA + Wb[..., 1:2] * oB + Wb[..., 2:3] * oC
    return pt, trib, Wb, d.gather(-1, j[..., None]).squeeze(-1)


# ====================================================================== synthetic artefact
def _grid(nx, ny, sx, sy, zf, seed):
    rs = np.random.RandomState(seed)
    x, y = np.linspace(0, sx, nx), np.linspace(0, sy, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    V = np.stack([X, Y, zf(X, Y) + 0.02 * rs.randn(nx, ny)], -1).reshape(-1, 3)
    i = (np.arange(nx - 1)[:, None] * ny + np.arange(ny - 1)[None, :]).ravel()
    F = np.concatenate([np.stack([i, i + ny, i + ny + 1], 1),
                        np.stack([i, i + ny + 1, i + 1], 1)])
    return V.astype(np.float64), F.astype(np.int64)


def synth_artefact(path, sizes=((22, 19), (26, 23), (20, 24)), seed=0):
    """A miniature mesh_data.npz with REAL connectivity and REAL operators.

    Same key layout, same GLOBAL-index / offset convention, same landmark-target encoding
    as build_mesh_data.py, so a smoke test against this file exercises the actual loader
    rather than a mock of it.
    """
    import scipy.sparse as sp
    acc = {k: [] for k in VKEYS + SKEYS + EKEYS + ("faces", "nbr", "deg")}
    v_ptr, f_ptr = [0], [0]
    lm_face, lm_bary, lm_vert, lm_dist, coarse = [], [], [], [], []
    rs = np.random.RandomState(seed)
    for si, (nx, ny) in enumerate(sizes):
        V, F = _grid(nx, ny, 30.0, 26.0,
                     lambda X, Y: 3.0 * np.sin(X / 7.0) * np.cos(Y / 6.0), seed + si)
        n = len(V)
        A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        cr = np.cross(B - A, C - A)
        a2 = np.linalg.norm(cr, axis=1)
        cA = np.einsum("ij,ij->i", B - A, C - A) / a2
        cB = np.einsum("ij,ij->i", A - B, C - B) / a2
        cC = np.einsum("ij,ij->i", A - C, B - C) / a2
        w = np.concatenate([cA, cA, cB, cB, cC, cC]) / 2.0
        r = np.concatenate([F[:, 1], F[:, 2], F[:, 2], F[:, 0], F[:, 0], F[:, 1]])
        c = np.concatenate([F[:, 2], F[:, 1], F[:, 0], F[:, 2], F[:, 1], F[:, 0]])
        W = sp.coo_matrix((w, (r, c)), shape=(n, n)).tocsr(); W.sort_indices()
        diag = np.asarray(W.sum(1)).ravel()
        mass = np.bincount(F.ravel(), np.repeat(a2 / 6.0, 3), minlength=n)
        N = np.zeros_like(V)
        for k in range(3):
            np.add.at(N, F[:, k], cr)
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        ax = np.zeros_like(N); ax[np.arange(n), np.argmin(np.abs(N), 1)] = 1.0
        X = np.cross(N, ax); X /= np.linalg.norm(X, axis=1, keepdims=True)
        Y = np.cross(N, X)
        row = np.repeat(np.arange(n), np.diff(W.indptr))
        e = V[W.indices] - V[row]
        ex = np.einsum("ij,ij->i", e, X[row]); ey = np.einsum("ij,ij->i", e, Y[row])
        w2 = 1.0 / np.maximum(np.einsum("ij,ij->i", e, e), 1e-18)
        gxx = np.bincount(row, w2 * ex * ex, minlength=n) + 1e-5
        gxy = np.bincount(row, w2 * ex * ey, minlength=n)
        gyy = np.bincount(row, w2 * ey * ey, minlength=n) + 1e-5
        det = gxx * gyy - gxy * gxy
        axc, ayc = w2 * ex, w2 * ey
        gx = (gyy / det)[row] * axc + (-gxy / det)[row] * ayc
        gy = (-gxy / det)[row] * axc + (gxx / det)[row] * ayc
        for k, v in (("verts", V), ("nrm", N), ("basis_x", X), ("basis_y", Y),
                     ("mass", mass), ("lap_diag", diag),
                     ("grad_xd", -np.bincount(row, gx, minlength=n)),
                     ("grad_yd", -np.bincount(row, gy, minlength=n)),
                     ("lap_w", W.data), ("grad_x", gx), ("grad_y", gy)):
            acc[k].append(np.asarray(v, np.float32))
        acc["faces"].append((F + v_ptr[-1]).astype(np.int32))
        acc["nbr"].append((W.indices + v_ptr[-1]).astype(np.int32))
        acc["deg"].append(np.diff(W.indptr).astype(np.int32))
        fi = rs.choice(len(F), 85, replace=False)
        bw = rs.dirichlet(np.ones(3) * 4.0, 85)
        lm_face.append(fi + f_ptr[-1]); lm_bary.append(bw)
        p = np.einsum("ij,ijk->ik", bw, V[F[fi]])
        lm_vert.append(F[fi][np.arange(85), np.abs(bw).argmax(1)] + v_ptr[-1])
        lm_dist.append(np.zeros(85))
        coarse.append(p + rs.randn(85, 3) * 0.7)
        v_ptr.append(v_ptr[-1] + n); f_ptr.append(f_ptr[-1] + len(F))
    out = {k: np.concatenate(v) for k, v in acc.items() if k != "deg"}
    out["v_ptr"] = np.asarray(v_ptr, np.int64)
    out["f_ptr"] = np.asarray(f_ptr, np.int64)
    out["deg_ptr"] = np.concatenate([[0], np.cumsum(np.concatenate(acc["deg"]))]).astype(np.int64)
    out["lm_face"] = np.stack(lm_face).astype(np.int32)
    out["lm_bary"] = np.stack(lm_bary).astype(np.float32)
    out["lm_vert"] = np.stack(lm_vert).astype(np.int32)
    out["lm_dist"] = np.stack(lm_dist).astype(np.float32)
    out["coarse"] = np.stack(coarse).astype(np.float32)
    out["n_vert"] = np.diff(out["v_ptr"]).astype(np.int32)
    np.savez(path, **out)
    return path


# ====================================================================== smoke test
if __name__ == "__main__":
    t0 = time.time()
    tmp = os.environ.get("SMOKE_DIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "meshb"))
    os.makedirs(tmp, exist_ok=True)
    p = synth_artefact(os.path.join(tmp, "synth_mesh.npz"))
    st = store("cpu", keig=0, path=p)
    ns = [int(x) for x in st.nv]
    print(f"synthetic artefact: {st.ne} ears with {ns} vertices (RAGGED, all different)")

    b = st.pad([0, 1, 2])
    print("padded: " + "  ".join(f"{k} {tuple(v.shape)}" for k, v in b.items()
                                 if k in ("verts", "nbr", "lap_w", "fan1", "fan_mask")))
    assert all(v.dim() <= 3 for v in b.values() if torch.is_tensor(v)), \
        "a batch tensor has ndim >= 4: train_family._flatten_samples would squeeze axis 1"
    P = b["verts"].shape[1]
    for i, n in enumerate(ns):
        assert bool(b["vmask"][i, :n].all()) and not bool(b["vmask"][i, n:].any())
        assert float(b["mass"][i, n:].abs().sum()) == 0.0
        assert float(b["lap_w"][i, n:].abs().sum()) == 0.0
        assert bool((b["nbr"][i, n:] == torch.arange(n, P)[:, None]).all())
    print(f"  padding: mass/lap_w EXACTLY 0 beyond nv, padded one-ring points at self, "
          f"no ndim>=4 tensor")

    # 1. the padded operators must reproduce the exact ragged ones
    f = torch.randn(3, P, 1, generator=torch.Generator().manual_seed(0)) * b["vmask"][..., None]
    lap, gx, gy, _, _ = mesh_ops(f, b)
    z = np.load(p)
    for i in range(3):
        n = ns[i]
        g0 = int(st.v_ptr[i]); d0, d1 = int(st.deg_ptr[g0]), int(st.deg_ptr[g0 + n])
        idx = z["nbr"][d0:d1] - g0
        rowr = np.repeat(np.arange(n), np.diff(z["deg_ptr"][g0:g0 + n + 1]))
        fv = f[i, :n, 0].numpy()
        lr = z["lap_diag"][g0:g0+n] * fv - np.bincount(rowr, z["lap_w"][d0:d1] * fv[idx], minlength=n)
        assert np.abs(lr - lap[i, :n, 0].numpy()).max() < 2e-4, np.abs(lr - lap[i, :n, 0].numpy()).max()
    print(f"  padded Laplacian == ragged Laplacian to {np.abs(lr - lap[2,:n,0].numpy()).max():.2e}")

    # 2. the gradient operator is exact on a linear field over a planar-ish one-ring
    lin = (b["verts"] @ torch.tensor([0.7, -1.3, 0.4])).unsqueeze(-1)
    _, gx, gy, _, _ = mesh_ops(lin, b)
    ref_x = (b["basis_x"] @ torch.tensor([0.7, -1.3, 0.4]))
    err = float(((gx[..., 0] - ref_x).abs() * b["vmask"]).max())
    print(f"  tangent gradient of a linear field, max err {err:.2e} (curved mesh -> O(h))")
    assert err < 0.35

    # 3. curvature against the analytic mean curvature of the synthetic height field
    k1, k2, H, K, Hlap, Hn = curvature(b)
    v = b["verts"][0][b["vmask"][0]]
    zxx = -3.0 / 49 * torch.sin(v[:, 0] / 7) * torch.cos(v[:, 1] / 6)
    zyy = -3.0 / 36 * torch.sin(v[:, 0] / 7) * torch.cos(v[:, 1] / 6)
    zxy = -3.0 / 42 * torch.cos(v[:, 0] / 7) * torch.sin(v[:, 1] / 6)
    zx = 3.0 / 7 * torch.cos(v[:, 0] / 7) * torch.cos(v[:, 1] / 6)
    zy = -3.0 / 6 * torch.sin(v[:, 0] / 7) * torch.sin(v[:, 1] / 6)
    den = (1 + zx ** 2 + zy ** 2)
    Hex = ((1 + zy ** 2) * zxx - 2 * zx * zy * zxy + (1 + zx ** 2) * zyy) / (2 * den ** 1.5)
    hh, hl = H[0][b["vmask"][0]], Hlap[0][b["vmask"][0]]
    cg = np.corrcoef(hh.numpy(), Hex.numpy())[0, 1]
    cl = np.corrcoef(hl.numpy(), Hex.numpy())[0, 1]
    print(f"  mean curvature 1/mm: shape-operator {hh.mean():+.4f}+-{hh.std():.4f} "
          f"(corr {cg:+.3f})  cotan-Laplacian {hl.mean():+.4f}+-{hl.std():.4f} (corr {cl:+.3f})"
          f"  vs analytic {Hex.mean():+.4f}+-{Hex.std():.4f}")
    assert cg > 0.9 and cl > 0.75, (cg, cl)
    k1v, k2v = k1[0][b["vmask"][0]], k2[0][b["vmask"][0]]
    assert bool((k1v >= k2v).all()), "principal curvatures out of order"

    # 4. the fan decode lands exactly on a real triangle and recovers a point placed on one
    gtb = torch.tensor(z["lm_bary"]).float()
    gtf = torch.tensor(z["lm_face"].astype(np.int64))
    tri = torch.tensor(z["faces"].astype(np.int64))[gtf] - st.v_ptr[:3, None, None]
    gt = (gtb[..., None] * gather_pts(b["verts"], tri.reshape(3, -1)).view(3, 85, 3, 3)).sum(2)
    dv = torch.cdist(gt, b["verts"]).masked_fill(~b["vmask"][:, None], 1e9)
    vstar = dv.argmin(-1)
    pt, tb, wb, res = fan_project(gt, vstar, b)
    rec = (wb[..., None] * gather_pts(b["verts"], tb.reshape(3, -1)).view(3, 85, 3, 3)).sum(2)
    assert float(wb.min()) >= -1e-6 and float((wb.sum(-1) - 1).abs().max()) < 1e-5
    assert float((rec - pt).abs().max()) < 1e-4
    print(f"  fan decode: nearest-vertex {float(dv.min(-1).values.mean()):.4f} mm -> "
          f"barycentric {float((pt-gt).norm(dim=-1).mean()):.5f} mm, weights in the simplex, "
          f"reconstruction exact")

    # 5. augmentation moves mesh, coarse and target as ONE rigid body
    b["coarse"] = torch.tensor(z["coarse"]).float()
    cfg = dict(aug_rot=1.2, aug_scale=0.0, aug_qjit=0.0)
    g = torch.Generator(); g.manual_seed(3)
    b2, gt2 = mesh_augment(b, gt, cfg, (), g)
    for i in range(3):
        n = ns[i]
        a = torch.cdist(gt[i], b["verts"][i, :n]); c = torch.cdist(gt2[i], b2["verts"][i, :n])
        assert float((a - c).abs().max()) < 2e-3, float((a - c).abs().max())
        fr = torch.stack([b2["basis_x"][i, :n], b2["basis_y"][i, :n], b2["nrm"][i, :n]], 1)
        assert float((fr @ fr.transpose(1, 2) - torch.eye(3)).abs().max()) < 1e-4
    print("  rotation: target-to-mesh distances invariant, tangent frames still orthonormal")
    cfg["aug_scale"] = 0.4
    b3, gt3 = mesh_augment(b, gt, cfg, (), g)
    sc = (b3["verts"][:, :ns[0]].norm(dim=-1).sum(1) / b["verts"][:, :ns[0]].norm(dim=-1).sum(1))
    ma = b3["mass"].sum(1) / b["mass"].sum(1)
    print(f"  scale: s {[round(float(x),4) for x in sc]}  area ratio "
          f"{[round(float(x),4) for x in ma]}  (must be s^2)")
    assert float((ma - sc ** 2).abs().max()) < 2e-2

    # 6. cost of one padded batch
    for _ in range(3):
        st.pad([0, 1, 2])
    t = time.time()
    for _ in range(20):
        st.pad([0, 1, 2])
    print(f"\nstore {st.bytes/1e6:.1f} MB | pad(B=3) {1000*(time.time()-t)/20:.2f} ms/batch "
          f"(device index arithmetic only)")
    print(f"SMOKE PASS in {time.time()-t0:.1f}s")
