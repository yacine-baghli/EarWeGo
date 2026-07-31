"""
FAMILY B -- intrinsic DiffusionNet on the ear MESH, with the two heads that attack
ORDERED CORRESPONDENCE rather than local XYZ accuracy.

Why this family exists. The shipped 2048-point static-DGCNN refinement family is at
1.273 mm (val) / 1.3144 mm (pooled OOF) and its error decomposes as along-contour RMSE
1.4456 (77 % of the energy), across-contour 0.7233 (20 %), surface-normal 0.2822 (2 %).
That decomposition is INVARIANT across all seven screened variants, six learned
correction predictors returned ~zero OOF R^2, and correspondence oracles on the
predicted polyline recover 1.0345 (scalar shift) / 0.7848 (affine) / 0.5657
(monotone-per-point). So the residual is PHASE ALONG THE CONTOUR, and any further
generic local-XYZ refinement stage is measuring noise. Both heads below make phase a
structural property of the model instead of something a loss has to discover:

  HEAD=heatmap    85 per-vertex heatmaps -> masked softmax over vertices -> expected
                  position (this is what carries the gradient), then a bounded tangent
                  offset at the argmax vertex which is projected EXACTLY onto that
                  vertex's incident-triangle fan. The output is a genuine barycentric
                  point of a real triangle, so it lies ON the surface by construction
                  (the GT lies 0.006 mm off the surface; raw predictions sit ~0.17 mm
                  off it, and snapping alone was worth 1.329 -> 1.309 mm).
  HEAD=coordfield a dense per-vertex CANONICAL TEMPLATE COORDINATE field u: V -> R^3.
                  Landmarks are FIXED points of the template's canonical space, so the
                  85 landmarks are transferred by inverting u: find the target vertex
                  whose predicted canonical coordinate is nearest the template landmark,
                  then solve for that landmark's clamped barycentric coordinates inside
                  the fan triangle MEASURED IN CANONICAL SPACE and apply those same
                  weights to the vertices' TARGET-space positions. Phase is then decided
                  by a dense field fitted over the whole surface, never by a local
                  detector, and the output is again exactly on the surface.

The backbone is genuinely intrinsic: learned per-channel diffusion time applied
spectrally, plus direction-aware spatial-gradient features (inner products of per-vertex
tangent gradients under a learned complex channel mixing, i.e. DiffusionNet's
rotation-equivariant gradient features), plus a pointwise MLP. INFEAT=hks is pure heat
kernel signature; the default xyzhks adds the canonical-frame xyz/normal, which is
legitimately available and informative -- but xyz is never the sole signal, and
INFEAT=xyz exists only as the ablation that proves the intrinsic channels carry signal.

LEAKAGE. Nothing here touches ground truth outside `loss()`. The only external artefact
is `tmpl_lm` (HEAD=coordfield): the 85 landmarks in the template's canonical space. That
is a FOLD-SCOPED artefact -- build_mesh_data.py must build it from the current fold's
TRAINING ears only and pass the fold index through. `canon_uv` (optional dense
correspondence supervision for the coordinate field) is likewise train-ears-only and is
read only inside `loss()`.

    HEAD=heatmap    WIDTH=128 BLOCKS=4 KEIG=128 python3 research/code/fam_diffusionnet.py
    HEAD=coordfield python3 research/code/fam_diffusionnet.py      # CPU smoke test
"""
import os, math
import numpy as np
import torch
import torch.nn as nn

NL = 85
SCALE = 30.0                      # mm normalizer, same as gpu_screen.py
CONTOURS = [(0, 24), (25, 54), (55, 74), (75, 84)]

WIDTH = int(os.environ.get("WIDTH", "128"))        # channels per DiffusionNet block
BLOCKS = int(os.environ.get("BLOCKS", "4"))        # number of blocks
KEIG = int(os.environ.get("KEIG", "128"))          # eigenpairs USED (sliced from the batch)
DROPOUT = float(os.environ.get("DROPOUT", "0.1"))
HEAD = os.environ.get("HEAD", "heatmap")           # heatmap | coordfield
INFEAT = os.environ.get("INFEAT", "xyzhks")        # hks | xyz | xyzhks
NHKS = int(os.environ.get("NHKS", "16"))           # heat-kernel-signature scales
TLO = float(os.environ.get("TLO", "1.0"))          # mm^2; diffusion distance 1 mm
THI = float(os.environ.get("THI", "400.0"))        # mm^2; diffusion distance 20 mm
TAU = float(os.environ.get("TAU", "1.0"))          # coordfield affinity temperature, mm
W_SOFT = float(os.environ.get("W_SOFT", "1.0"))    # loss: soft-expected-position term
W_CE = float(os.environ.get("W_CE", "0.3"))        # loss: heatmap cross-entropy term
W_FIELD = float(os.environ.get("W_FIELD", "1.0"))  # loss: dense canonical-coordinate term
STRICT = int(os.environ.get("STRICT", "1"))         # re-check the mass-padding invariant
                                                    # every forward. Costs one device sync
                                                    # per step; set 0 once a driver has
                                                    # verified its packed tensors.

# --------------------------------------------------------------------------------------
# TENSOR CONTRACT required from build_mesh_data.py (which does ALL mesh/connectivity work
# LOCALLY with numpy/scipy and ships plain padded .npz tensors -- the GPU box has torch
# and nothing else). One batch is a dict of these; ears in a batch have DIFFERENT vertex
# counts, so everything is padded to Vmax/Fmax and `vmask` is authoritative.
# `reference_mesh_tensors()` at the bottom of this file is the executable specification:
# build_mesh_data.py must reproduce it per ear.
# --------------------------------------------------------------------------------------
CONTRACT = dict(
    verts=("(B,Vmax,3) f32", "vertex positions in the per-ear canonical frame, mm; "
                             "padding rows arbitrary (never read for real vertices)"),
    vmask=("(B,Vmax) bool", "True for real vertices; authoritative everywhere"),
    faces=("(B,Fmax,3) i64", "triangles, for diagnostics/regularizers; the model itself "
                             "only uses vfan"),
    fmask=("(B,Fmax) bool", "True for real faces"),
    mass=("(B,Vmax) f32", "lumped (barycentric) vertex areas, mm^2. MUST be EXACTLY 0 on "
                          "padding -- this is what keeps padding out of the spectral "
                          "transform"),
    evals=("(B,K) f32", "smallest K generalized eigenvalues of the cotan Laplacian "
                        "(L phi = lambda M phi), ascending, >= 0, units 1/mm^2"),
    evecs=("(B,Vmax,K) f32", "matching eigenvectors, M-orthonormal (evecs^T M evecs = I). "
                             "MUST be EXACTLY 0 on padding rows"),
    nrm=("(B,Vmax,3) f32", "unit vertex normals (area-weighted), consistently oriented"),
    tanb=("(B,Vmax,2,3) f32", "orthonormal tangent basis rows (e1,e2) with e1 x e2 = nrm. "
                              "grad_wx/grad_wy are expressed in THIS basis"),
    grad_nbr=("(B,Vmax,R) i64", "1-ring neighbour vertex ids; padding entries = the vertex "
                                "itself, so their difference is 0"),
    grad_wx=("(B,Vmax,R) f32", "least-squares tangent-gradient weights, e1 component: "
                               "grad_e1 u [i] = sum_j wx[i,j] (u[j] - u[i]). 0 on padding"),
    grad_wy=("(B,Vmax,R) f32", "same, e2 component"),
    vfan=("(B,Vmax,T,2) i64", "for each vertex, the OTHER TWO vertices of each incident "
                              "triangle (consistent winding); padded with the vertex "
                              "itself (degenerate, and masked out anyway)"),
    vfan_mask=("(B,Vmax,T) bool", "True for real incident triangles"),
    tmpl_lm=("(85,3) f32", "HEAD=coordfield ONLY: the 85 landmarks in the TEMPLATE's "
                           "canonical space (barycentric transport of bary_f/bary_w onto "
                           "the template vertices). FOLD-SCOPED: built from the current "
                           "fold's TRAINING ears only"),
    canon_uv=("(B,Vmax,3) f32", "OPTIONAL, TRAIN EARS ONLY, read only inside loss(): the "
                                "ear's ground-truth canonical coordinate per vertex, from "
                                "landmark-anchored NICP of the fold's template"),
    canon_ok=("(B,Vmax) bool", "which vertices have a trustworthy canon_uv"),
)


# ------------------------------------------------------------------ tensor plumbing
def _gather_pts(x, idx):
    """x (B,V,D), idx (B,...) int64 -> (B,...,D)"""
    B, _, D = x.shape
    return torch.gather(x, 1, idx.reshape(B, -1, 1).expand(-1, -1, D)).view(*idx.shape, D)


def bary_closest(p, A, B, C, eps=1e-9):
    """Clamped barycentric coords of the point of triangle (A,B,C) closest to p.

    Ericson's barycentric region algorithm (same regions and the same evaluation order as
    deep_model/surfproj.closest_on_triangles), but returning the WEIGHTS, which is what
    makes both heads exactly-on-surface AND differentiable: the returned point is always
    w0 A + w1 B + w2 C with w >= 0 and sum w = 1, and the weights carry gradient to
    whichever of p / A / B / C is learned. Degenerate padded triangles (A=B=C) fall into
    the vertex-A region and return (1,0,0).
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
    den = va + vb + vc

    def sdiv(n, d):                                   # never divides by ~0, so no inf/nan
        return n / torch.where(d.abs() > eps, d, torch.full_like(d, eps))

    v, w = sdiv(vb, den), sdiv(vc, den)
    W = torch.stack([1 - v - w, v, w], -1)            # region 0, interior
    z, o = torch.zeros_like(v), torch.ones_like(v)
    t = sdiv(d4 - d3, (d4 - d3) + (d5 - d6)).clamp(0, 1)
    W = torch.where((((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)))[..., None],
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


def fan_solve(target, vstar, b, coords, out_coords):
    """Project `target` onto the incident-triangle fan of vertex `vstar`, in `coords`,
    and read the answer off `out_coords`.

    HEAD=heatmap  : coords = out_coords = verts  -> exact point-to-surface projection.
    HEAD=coordfield: coords = predicted canonical field u, out_coords = verts -> exact
                     barycentric interpolation of the INVERSE canonical map.
    Returns (point (B,L,3), tri (B,L,3) i64, weights (B,L,3), residual (B,L) in `coords`).
    """
    B, L = vstar.shape
    T = b["vfan"].shape[2]
    fan = torch.gather(b["vfan"].reshape(B, -1, T * 2), 1,
                       vstar[..., None].expand(-1, -1, T * 2)).view(B, L, T, 2)
    msk = torch.gather(b["vfan_mask"], 1, vstar[..., None].expand(-1, -1, T))
    tri = torch.cat([vstar[..., None, None].expand(-1, -1, T, 1), fan], -1)     # (B,L,T,3)
    cA, cB, cC = (_gather_pts(coords, tri[..., k]) for k in range(3))
    W = bary_closest(target[:, :, None, :], cA, cB, cC)                          # (B,L,T,3)
    P = W[..., 0:1] * cA + W[..., 1:2] * cB + W[..., 2:3] * cC
    d = (P - target[:, :, None, :]).norm(dim=-1).masked_fill(~msk, 1e9)
    j = d.argmin(-1)                                                             # (B,L)

    def pick(X):
        return torch.gather(X, 2, j[..., None, None].expand(-1, -1, 1, X.shape[-1])).squeeze(2)

    Wb, trib = pick(W), pick(tri)
    oA, oB, oC = (_gather_pts(out_coords, trib[..., k]) for k in range(3))
    pt = Wb[..., 0:1] * oA + Wb[..., 1:2] * oB + Wb[..., 2:3] * oC
    return pt, trib, Wb, d.gather(-1, j[..., None]).squeeze(-1)


# ------------------------------------------------------------------ intrinsic operators
def spectral_diffuse(x, evals, evecs, mass, log_t):
    """evecs @ (exp(-lambda t) * (evecs^T (M x))), with a learned POSITIVE per-channel t.

    mass is exactly 0 on padded vertices, so padding cannot enter the spectral transform.
    """
    t = torch.exp(log_t)                                                # (C,) > 0
    xs = torch.einsum("bvk,bvc->bkc", evecs, mass[..., None] * x)       # (B,K,C)
    xs = xs * torch.exp(-evals[..., None] * t[None, None, :])
    return torch.einsum("bvk,bkc->bvc", evecs, xs)


def tangent_grad(x, nbr, wx, wy):
    """per-vertex tangent-plane gradient of every channel, in the (e1,e2) basis"""
    B, V, C = x.shape
    R = nbr.shape[-1]
    xj = torch.gather(x, 1, nbr.reshape(B, V * R, 1).expand(-1, -1, C)).view(B, V, R, C)
    dx = xj - x[:, :, None, :]
    return torch.einsum("bvr,bvrc->bvc", wx, dx), torch.einsum("bvr,bvrc->bvc", wy, dx)


def hks(evals, evecs, ts):
    """heat kernel signature, trace-normalized then logged -> (B,V,S). Fully intrinsic."""
    ex = torch.exp(-evals[..., None] * ts[None, None, :])               # (B,K,S)
    h = torch.einsum("bvk,bks->bvs", evecs ** 2, ex)
    return torch.log((h / ex.sum(1)[:, None, :].clamp(min=1e-12)).clamp(min=1e-12))


class GradFeat(nn.Module):
    """direction-aware features: Re(conj(g) . (A g)) with A a learned complex CxC mixing.

    Writing the tangent gradient as the complex number g = gx + i gy, this inner product
    is invariant to a per-vertex ROTATION of the tangent basis, so it does not depend on
    the arbitrary choice of e1 -- which is the whole point of using it instead of raw
    (gx, gy) channels.
    """
    def __init__(self, C):
        super().__init__()
        self.Wr = nn.Linear(C, C, bias=False)
        self.Wi = nn.Linear(C, C, bias=False)

    def forward(self, gr, gi):
        hr = self.Wr(gr) - self.Wi(gi)
        hi = self.Wr(gi) + self.Wi(gr)
        return torch.tanh(gr * hr + gi * hi)


class DiffusionBlock(nn.Module):
    def __init__(self, C, dropout, tlo, thi):
        super().__init__()
        # one diffusion time per channel, spread log-uniformly over the PHYSICAL range
        # [tlo, thi] mm^2 (diffusion distance sqrt(t)); positive by construction via exp.
        self.log_t = nn.Parameter(torch.linspace(math.log(tlo), math.log(thi), C))
        self.gradfeat = GradFeat(C)
        self.mlp = nn.Sequential(nn.Linear(3 * C, C), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(C, C))
        self.norm = nn.LayerNorm(C)

    def forward(self, x, b):
        d = spectral_diffuse(x, b["evals"], b["evecs"], b["mass"], self.log_t)
        gr, gi = tangent_grad(d, b["grad_nbr"], b["grad_wx"], b["grad_wy"])
        y = self.mlp(torch.cat([x, d, self.gradfeat(gr, gi)], -1))
        return self.norm(x + y) * b["vmask"][..., None]


class Backbone(nn.Module):
    def __init__(self, C=WIDTH, blocks=BLOCKS, dropout=DROPOUT, infeat=INFEAT,
                 nhks=NHKS, tlo=TLO, thi=THI):
        super().__init__()
        self.infeat, self.nhks = infeat, nhks
        self.register_buffer("hks_t", torch.exp(torch.linspace(math.log(tlo), math.log(thi),
                                                               nhks)))
        cin = (6 if "xyz" in infeat else 0) + (nhks if "hks" in infeat else 0)
        assert cin > 0, f"INFEAT={infeat} selects no input channels"
        self.lift = nn.Sequential(nn.Linear(cin, C), nn.ReLU(), nn.Linear(C, C))
        self.inorm = nn.LayerNorm(C)
        self.blocks = nn.ModuleList([DiffusionBlock(C, dropout, tlo, thi)
                                     for _ in range(blocks)])

    def forward(self, b):
        f = []
        if "xyz" in self.infeat:
            f += [b["verts"] / SCALE, b["nrm"]]
        if "hks" in self.infeat:
            f += [hks(b["evals"], b["evecs"], self.hks_t)]
        x = self.inorm(self.lift(torch.cat(f, -1))) * b["vmask"][..., None]
        for blk in self.blocks:
            x = blk(x, b)
        return x


# ------------------------------------------------------------------ HEAD 1: heatmaps
class HeatmapHead(nn.Module):
    """85 per-vertex heatmaps -> masked softmax over vertices -> expected position, then a
    bounded tangent offset at the argmax vertex projected exactly onto its triangle fan.

    KNOWN REACH LIMIT, stated because it decides what this head can and cannot fix: the
    refinement is bounded by the argmax vertex's 1-ring radius, so the head recovers
    SUB-VERTEX-SPACING accuracy and nothing more. If the heatmap peak is a millimetre off
    in PHASE along the contour, this head cannot move it -- the peak has to be right. The
    soft expected position is unbounded but off-surface, which is why it is used as the
    gradient carrier rather than as the output, and why HEAD=coordfield exists as the
    variant whose phase is decided globally instead of by a local peak.
    """
    def __init__(self, C, dropout=DROPOUT):
        super().__init__()
        self.logit = nn.Sequential(nn.Linear(C, C), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(C, NL))
        self.emb = nn.Embedding(NL, 32)
        self.off = nn.Sequential(nn.Linear(C + 3 + 32, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x, b):
        B, V, _ = x.shape
        vm = b["vmask"]
        logit = self.logit(x).masked_fill(~vm[..., None], -1e9)             # (B,V,NL)
        w = torch.softmax(logit, dim=1)                                     # over VERTICES
        p_soft = torch.einsum("bvl,bvd->bld", w, b["verts"])
        vstar = logit.argmax(1)                                             # (B,NL)
        pstar = _gather_pts(b["verts"], vstar)
        tb = _gather_pts(b["tanb"].reshape(B, V, 6), vstar).view(B, NL, 2, 3)
        hstar = _gather_pts(x, vstar)
        # bound the offset by the 1-ring radius so the candidate stays inside the fan
        nb = _gather_pts(b["verts"], b["grad_nbr"].reshape(B, -1)).view(B, V, -1, 3)
        ring = (nb - b["verts"][:, :, None, :]).norm(dim=-1).max(-1).values.detach()
        r = _gather_pts(ring[..., None], vstar)                             # (B,NL,1)
        e = self.emb.weight[None].expand(B, -1, -1)
        duv = r * torch.tanh(self.off(torch.cat([hstar, (p_soft - pstar) / SCALE, e], -1)))
        q = pstar + duv[..., 0:1] * tb[:, :, 0] + duv[..., 1:2] * tb[:, :, 1]
        lm, tri, bw, resid = fan_solve(q, vstar, b, b["verts"], b["verts"])
        return dict(lm=lm, p_soft=p_soft, logit=logit, heat=w, vstar=vstar, q=q,
                    tri=tri, bary=bw, snap_mm=resid)


# ------------------------------------------------------------------ HEAD 2: coord field
class CoordFieldHead(nn.Module):
    """dense per-vertex canonical TEMPLATE coordinate field, landmarks transferred by
    inverting it with exact barycentric interpolation on the fan of the nearest vertex."""
    def __init__(self, C, dropout=DROPOUT, tau=TAU):
        super().__init__()
        self.field = nn.Sequential(nn.Linear(C, C), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(C, 3))
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))

    def forward(self, x, b):
        vm = b["vmask"]
        L = b["tmpl_lm"]                                                    # (85,3)
        u = (L.mean(0) + SCALE * self.field(x)) * vm[..., None]             # (B,V,3)
        tau = torch.exp(self.log_tau).clamp(min=1e-3)
        # cdist, not a (B,V,85,3) difference tensor: at 30k vertices that intermediate is
        # 30 MB/ear and autograd keeps it
        d2 = torch.cdist(u, L[None].expand(u.shape[0], -1, -1)) ** 2        # (B,V,85)
        logit = (-d2 / (2 * tau ** 2)).masked_fill(~vm[..., None], -1e9)
        w = torch.softmax(logit, dim=1)                                     # over VERTICES
        p_soft = torch.einsum("bvl,bvd->bld", w, b["verts"])
        vstar = logit.argmax(1)                                             # (B,85)
        tgt = L[None].expand(x.shape[0], -1, -1)
        lm, tri, bw, resid = fan_solve(tgt, vstar, b, u, b["verts"])
        return dict(lm=lm, p_soft=p_soft, u=u, heat=w, vstar=vstar,
                    tri=tri, bary=bw, canon_resid=resid, tau=tau.detach())


# ------------------------------------------------------------------ model + loss
class FamDiffusionNet(nn.Module):
    def __init__(self, head=HEAD, width=WIDTH, blocks=BLOCKS, keig=KEIG, dropout=DROPOUT,
                 infeat=INFEAT, nhks=NHKS, tlo=TLO, thi=THI, tau=TAU):
        super().__init__()
        assert head in ("heatmap", "coordfield"), f"HEAD={head}"
        self.head_name, self.keig = head, keig
        self.backbone = Backbone(width, blocks, dropout, infeat, nhks, tlo, thi)
        self.head = (HeatmapHead(width, dropout) if head == "heatmap"
                     else CoordFieldHead(width, dropout, tau))

    def forward(self, b):
        # `mass` zero on padding is the ONE contract invariant the masking rests on: it is
        # what keeps padded vertices out of the spectral transform. Everything else is
        # masked explicitly (softmaxes, block outputs, 1-ring weights, triangle fans).
        assert not STRICT or float((b["mass"] * ~b["vmask"]).abs().max()) == 0.0, \
            "contract violation: mass must be EXACTLY 0 on padded vertices"
        b = dict(b)
        b["evals"] = b["evals"][:, :self.keig]
        b["evecs"] = b["evecs"][..., :self.keig]
        out = self.head(self.backbone(b), b)
        assert out["lm"].shape[1:] == (NL, 3)
        return out

    def loss(self, out, target, b, w_soft=W_SOFT, w_ce=W_CE, w_field=W_FIELD):
        """target (B,85,3) GT landmarks in the canonical frame -- TRAIN-FOLD EARS ONLY.

        Every term is restricted to real vertices: the softmax weights are already
        exactly 0 on padding, the cross-entropy target vertex is chosen among real
        vertices only, and the dense field term is averaged over canon_ok & vmask.
        """
        vm = b["vmask"]
        terms = {"lm": ((out["lm"] - target) ** 2).sum(-1).mean(),
                 "soft": ((out["p_soft"] - target) ** 2).sum(-1).mean()}
        total = terms["lm"] + w_soft * terms["soft"]
        if self.head_name == "heatmap" and w_ce > 0:
            d = torch.cdist(target, b["verts"]).masked_fill(~vm[:, None, :], 1e9)
            vt = d.argmin(-1)                                               # (B,85)
            lp = torch.log_softmax(out["logit"], dim=1).transpose(1, 2)     # (B,NL,V)
            terms["ce"] = -torch.gather(lp, 2, vt[..., None]).mean()
            total = total + w_ce * terms["ce"]
        if self.head_name == "coordfield" and "canon_uv" in b and w_field > 0:
            ok = (b["canon_ok"] & vm).float()
            terms["field"] = (((out["u"] - b["canon_uv"]) ** 2).sum(-1) * ok).sum() \
                / ok.sum().clamp(min=1.0)
            total = total + w_field * terms["field"]
        return total, {k: float(v.detach()) for k, v in terms.items()}


# ======================================================================================
# REFERENCE PREPROCESSING -- the executable specification of the CONTRACT above.
# build_mesh_data.py must produce exactly these fields, per ear, LOCALLY (numpy/scipy),
# and ship them padded in a .npz. Nothing below ever runs on the GPU box.
# ======================================================================================
def reference_mesh_tensors(V, F, keig):
    """cotan Laplacian eigenpairs, lumped mass, normals, tangent basis, 1-ring LSQ
    gradient weights and the incident-triangle fan for ONE ear."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spl
    V = np.asarray(V, np.float64); F = np.asarray(F, np.int64)
    n = len(V)
    i0, i1, i2 = F[:, 0], F[:, 1], F[:, 2]
    e0, e1, e2 = V[i2] - V[i1], V[i0] - V[i2], V[i1] - V[i0]
    cr = np.cross(e2, -e1)                                    # 2*area * face normal
    a2 = np.maximum(np.linalg.norm(cr, axis=1), 1e-14)
    cot = np.stack([-(e1 * e2).sum(1), -(e2 * e0).sum(1), -(e0 * e1).sum(1)]) / a2
    ii = np.r_[i1, i2, i2, i0, i0, i1]
    jj = np.r_[i2, i1, i0, i2, i1, i0]
    ww = 0.5 * np.r_[cot[0], cot[0], cot[1], cot[1], cot[2], cot[2]]
    Wm = sp.coo_matrix((ww, (ii, jj)), shape=(n, n)).tocsr()
    Lap = (sp.diags(np.asarray(Wm.sum(1)).ravel()) - Wm).tocsc()
    mass = np.zeros(n)
    np.add.at(mass, F.ravel(), np.repeat(a2 / 6.0, 3))        # barycentric lumping
    mass = np.maximum(mass, 1e-9)
    k = min(keig, n - 2)
    vals, vecs = spl.eigsh(Lap, k=k, M=sp.diags(mass).tocsc(), sigma=-1e-5, which="LM")
    o = np.argsort(vals)
    vals, vecs = np.maximum(vals[o], 0.0), vecs[:, o]
    vecs = vecs / np.sqrt((mass[:, None] * vecs ** 2).sum(0, keepdims=True))
    nrm = np.zeros((n, 3))
    np.add.at(nrm, F.ravel(), np.repeat(cr, 3, axis=0))
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    ax = np.where((np.abs(nrm[:, 0:1]) > 0.9), np.array([[0., 1., 0.]]), np.array([[1., 0., 0.]]))
    e_1 = ax - (ax * nrm).sum(1, keepdims=True) * nrm
    e_1 = e_1 / np.maximum(np.linalg.norm(e_1, axis=1, keepdims=True), 1e-12)
    tanb = np.stack([e_1, np.cross(nrm, e_1)], 1)                          # e1 x e2 = nrm
    nbrs = [set() for _ in range(n)]
    fans = [[] for _ in range(n)]
    for a, bb, c in F:
        nbrs[a] |= {bb, c}; nbrs[bb] |= {c, a}; nbrs[c] |= {a, bb}
        fans[a].append((bb, c)); fans[bb].append((c, a)); fans[c].append((a, bb))
    R = max(len(s) for s in nbrs); T = max(len(f) for f in fans)
    nbr = np.tile(np.arange(n)[:, None], (1, R))
    wx = np.zeros((n, R)); wy = np.zeros((n, R))
    for i, s in enumerate(nbrs):
        j = np.fromiter(sorted(s), int)
        D = (V[j] - V[i]) @ tanb[i].T                                      # (deg,2)
        G = np.linalg.solve(D.T @ D + 1e-6 * np.eye(2), D.T)                # (2,deg)
        nbr[i, :len(j)] = j; wx[i, :len(j)] = G[0]; wy[i, :len(j)] = G[1]
    fan = np.tile(np.arange(n)[:, None, None], (1, T, 2))
    fmk = np.zeros((n, T), bool)
    for i, f in enumerate(fans):
        fan[i, :len(f)] = np.asarray(f); fmk[i, :len(f)] = True
    return dict(verts=V, faces=F, mass=mass, evals=vals, evecs=vecs, nrm=nrm, tanb=tanb,
                grad_nbr=nbr, grad_wx=wx, grad_wy=wy, vfan=fan, vfan_mask=fmk)


def pack_batch(ears, device="cpu"):
    """pad a list of reference_mesh_tensors() dicts (DIFFERENT vertex counts) into a batch"""
    Vmax = max(len(e["verts"]) for e in ears)
    Fmax = max(len(e["faces"]) for e in ears)
    K = min(len(e["evals"]) for e in ears)
    R = max(e["grad_nbr"].shape[1] for e in ears)
    T = max(e["vfan"].shape[1] for e in ears)
    B = len(ears)
    b = dict(verts=np.zeros((B, Vmax, 3), np.float32), vmask=np.zeros((B, Vmax), bool),
             faces=np.zeros((B, Fmax, 3), np.int64), fmask=np.zeros((B, Fmax), bool),
             mass=np.zeros((B, Vmax), np.float32), evals=np.zeros((B, K), np.float32),
             evecs=np.zeros((B, Vmax, K), np.float32), nrm=np.zeros((B, Vmax, 3), np.float32),
             tanb=np.zeros((B, Vmax, 2, 3), np.float32),
             grad_nbr=np.zeros((B, Vmax, R), np.int64),
             grad_wx=np.zeros((B, Vmax, R), np.float32),
             grad_wy=np.zeros((B, Vmax, R), np.float32),
             vfan=np.zeros((B, Vmax, T, 2), np.int64),
             vfan_mask=np.zeros((B, Vmax, T), bool))
    for i, e in enumerate(ears):
        n, m = len(e["verts"]), len(e["faces"])
        b["verts"][i, :n] = e["verts"]; b["vmask"][i, :n] = True
        b["faces"][i, :m] = e["faces"]; b["fmask"][i, :m] = True
        b["mass"][i, :n] = e["mass"]                        # exactly 0 beyond n
        b["evals"][i] = e["evals"][:K]
        b["evecs"][i, :n] = e["evecs"][:, :K]               # exactly 0 beyond n
        b["nrm"][i, :n] = e["nrm"]; b["tanb"][i, :n] = e["tanb"]
        b["grad_nbr"][i] = np.arange(Vmax)[:, None]         # padding points at itself
        b["grad_nbr"][i, :n, :e["grad_nbr"].shape[1]] = e["grad_nbr"]
        b["grad_wx"][i, :n, :e["grad_wx"].shape[1]] = e["grad_wx"]
        b["grad_wy"][i, :n, :e["grad_wy"].shape[1]] = e["grad_wy"]
        b["vfan"][i] = np.arange(Vmax)[:, None, None]
        b["vfan"][i, :n, :e["vfan"].shape[1]] = e["vfan"]
        b["vfan_mask"][i, :n, :e["vfan_mask"].shape[1]] = e["vfan_mask"]
    return {k: torch.tensor(v).to(device) for k, v in b.items()}


# ======================================================================================
if __name__ == "__main__":
    import time
    t0 = time.time()
    torch.manual_seed(0); np.random.seed(0)

    def synth_ear(nu, nv, seed):
        """a curved triangulated patch at ear mm-scale -- real connectivity, so the
        Laplacian, the gradient operator and the triangle fans are all genuine."""
        rs = np.random.RandomState(seed)
        u = np.linspace(0, 40, nu); v = np.linspace(0, 30, nv)
        U, Vv = np.meshgrid(u, v, indexing="ij")
        Z = 6 * np.sin(U / 9.0) * np.cos(Vv / 7.0) + 0.15 * rs.randn(nu, nv)
        P = np.stack([U, Vv, Z], -1).reshape(-1, 3)
        idx = np.arange(nu * nv).reshape(nu, nv)
        a, bb = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel()
        c, dd = idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
        Fq = np.concatenate([np.stack([a, bb, c], -1), np.stack([a, c, dd], -1)])
        return P, Fq

    ears = [reference_mesh_tensors(*synth_ear(20, 18, 0), keig=KEIG),
            reference_mesh_tensors(*synth_ear(24, 22, 1), keig=KEIG)]
    ns = [len(e["verts"]) for e in ears]
    print(f"synthetic batch: B=2 ears with {ns[0]} and {ns[1]} vertices (DIFFERENT), "
          f"{len(ears[0]['evals'])} eigenpairs")
    b = pack_batch(ears)
    print(f"padded shapes: verts {tuple(b['verts'].shape)} evecs {tuple(b['evecs'].shape)} "
          f"grad_nbr {tuple(b['grad_nbr'].shape)} vfan {tuple(b['vfan'].shape)}")
    for i, n in enumerate(ns):
        G = (b["evecs"][i, :n].T * b["mass"][i, :n]) @ b["evecs"][i, :n]
        off = (G - torch.eye(G.shape[0])).abs().max().item()
        print(f"  ear{i}: M-orthonormality |evecs^T M evecs - I|_max = {off:.2e}, "
              f"lambda_1 = {b['evals'][i, 1]:.5f} 1/mm^2, area = {b['mass'][i].sum():.1f} mm^2")

    # the template's canonical landmark positions (fold-scoped artefact; synthetic here)
    P0 = ears[0]["verts"]
    b["tmpl_lm"] = torch.tensor(P0[np.linspace(0, len(P0) - 1, NL).astype(int)]).float()
    b["canon_uv"] = b["verts"].clone()
    b["canon_ok"] = b["vmask"].clone()
    target = b["tmpl_lm"][None].expand(2, -1, -1) + torch.randn(2, NL, 3) * 0.5

    for head in ("heatmap", "coordfield"):
        net = FamDiffusionNet(head=head)
        npar = sum(p.numel() for p in net.parameters())
        out = net(b)
        lm = out["lm"]
        assert lm.shape == (2, NL, 3), lm.shape
        loss, terms = net.loss(out, target, b)
        loss.backward()
        gs = {n: float(p.grad.norm()) for n, p in net.named_parameters() if p.grad is not None}
        assert len(gs) == len(list(net.parameters())), "some parameters received no gradient"
        assert all(np.isfinite(v) for v in gs.values()), "non-finite gradient"
        gn = sum(v ** 2 for v in gs.values()) ** .5
        # exactly-on-surface: the output IS a convex barycentric point of a real triangle
        tri = out["tri"]
        rec = (out["bary"][..., None] * _gather_pts(b["verts"], tri.reshape(2, -1))
               .view(2, NL, 3, 3)).sum(2)
        assert out["bary"].min() >= -1e-6 and (out["bary"].sum(-1) - 1).abs().max() < 1e-5
        assert (rec - lm).abs().max() < 1e-4, (rec - lm).abs().max()
        # masked vertices must not influence anything
        net.eval()
        ref = net(b)["lm"].detach().clone()
        b2 = dict(b)
        for key in ("verts", "nrm", "evecs", "tanb"):
            t = b[key].clone()
            for i, n in enumerate(ns):
                t[i, n:] = torch.randn_like(t[i, n:]) * 100
            b2[key] = t
        d_mask = (net(b2)["lm"].detach() - ref).abs().max().item()
        assert d_mask == 0.0, f"padded vertices leaked into the output ({d_mask})"
        net.train()
        print(f"\nHEAD={head:10s} params {npar:,}  out {tuple(lm.shape)}  "
              f"loss {float(loss):.4f} {terms}")
        print(f"  grad-norm {gn:.4e}  bary in simplex OK  on-surface reconstruction OK  "
              f"padding-invariance |delta|={d_mask:.1e}")
        # the intrinsic path must be live, not a dead branch beside the xyz channels
        print(f"  grad reaches log_t {gs['backbone.blocks.0.log_t']:.3e} and gradfeat "
              f"{gs['backbone.blocks.0.gradfeat.Wr.weight']:.3e}")
        print(f"  learned t range {float(net.backbone.blocks[0].log_t.exp().min()):.2f}"
              f"-{float(net.backbone.blocks[0].log_t.exp().max()):.1f} mm^2  "
              + (f"snap {float(out['snap_mm'].mean()):.4f} mm" if head == "heatmap"
                 else f"canon resid {float(out['canon_resid'].mean()):.4f} mm "
                      f"tau {float(out['tau']):.3f}"))

    # the diffusion operator really is the heat kernel: t -> inf goes to the mass-mean
    x = torch.randn(2, b["verts"].shape[1], 1) * b["vmask"][..., None]
    big = spectral_diffuse(x, b["evals"], b["evecs"], b["mass"], torch.tensor([12.0]))
    mean = ((b["mass"][..., None] * x).sum(1) / b["mass"].sum(1)[:, None])
    print(f"\nheat-kernel check: |diffuse(t=e^12) - mass-mean|_max = "
          f"{float(((big - mean[:, None]) * b['vmask'][..., None]).abs().max()):.2e}")
    print(f"smoke test OK in {time.time()-t0:.1f}s on "
          f"{'cuda' if torch.cuda.is_available() else 'cpu'}")
