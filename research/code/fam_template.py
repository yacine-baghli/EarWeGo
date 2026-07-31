"""
FAMILY C step 2: dense template correspondence + inference-time deformation.

The 85 landmarks are NEVER regressed as 85 free XYZ points. They are fixed barycentric
points on a canonical template mesh (built by research/code/build_template.py), so the
only thing the network can predict is a DEFORMATION of that template, and ordered
correspondence along each contour is a property of the template rather than something the
loss has to discover. Along-contour phase error -- 77% of the 1.31mm baseline's error
energy -- is structurally excluded: landmark 31 cannot slide past landmark 30 without
tearing the mesh.

Two directions of the same formulation, DIRECTION=t2s|s2t:
  t2s  each template control vertex attends to the target cloud around its current
       position and predicts where it goes (template -> surface).
  s2t  each target point predicts its own coordinate in template space, and each control
       vertex is the softmax-kernel average of the target points that claim it
       (surface -> template). Its output is a convex combination of OBSERVED points, so
       it cannot hallucinate off-surface geometry.
Both end in the same tail: control-vertex displacements are linear-blend-skinned to all
template vertices with FIXED weights from the npz, then the 85 landmarks are transported
through the FIXED barycentric map. Deformation, skinning and transport are all linear, so
gradients reach the encoder through the landmark loss without any per-landmark head.

A global dense-PCA coefficient head (fold-safe basis from build_template.py) supplies the
coarse shape; the control head only has to correct it, bounded by MAX_OFF.

INFERENCE-TIME REFINEMENT (optional, ITERS>0) -- a proper ARAP local/global solver, not a
gradient hack. It minimises, over the template vertices V,

    W_P2P   point-to-plane to the observed cloud (nearest point, trimmed at P2P_TRIM)
  + W_NRM   normal consistency: the deformed one-ring must lie in the plane orthogonal to
            the target normal, sum_j w_ij ((v_i-v_j).n_i)^2. Quadratic and sign-invariant,
            which the raw cosine 1-(n_V.n_q)^2 is not; the cosine is REPORTED as a monitor
            but not optimised, because it has no quadratic majorizer.
  + W_EDGE  rest-edge-length preservation (|v_a-v_b| - L0)^2
  + W_ARAP  as-rigid-as-possible, sum_i sum_j w_ij ||(v_i-v_j) - R_i e0_ij||^2 with R_i a
            true rotation from a per-vertex 3x3 SVD (Sorkine-Alexa), cotangent weights
  + W_CORR  stay near the predicted correspondence

Every term is a MEAN in mm^2, so the weights are dimensionless and comparable -- the
earlier version summed ARAP over ~3400 edge terms while averaging the data terms, and
Adam's scale-free first step then multiplied the ARAP energy by 13 (that is the bug this
solver replaces). The loop is:

  LOCAL  step: per-vertex SVD rotations R_i and per-edge unit directions u_ab, fitted at
               the current V. With those held fixed the objective Q is EXACTLY quadratic
               and majorizes the true energy, tightly at the current V (ARAP-with-optimal-R
               and |x-L0*u| >= ||x|-L0| are both tight there).
  GLOBAL step: minimise Q by conjugate gradients. No matrix is assembled: A x is read off
               autograd as 0.5*(grad Q(x) - grad Q(0)), and CG minimises Q over growing
               Krylov subspaces from the current V.
  Hence E(V_new) <= Q(V_new) <= Q(V_old) = E(V_old): the true energy cannot increase
  within a match block. Re-matching (every NN_EVERY iterations) changes the objective, so
  monotonicity is asserted per block, not across a re-match.

REST shape (REST env) decides what "rigid" is measured against:
  base   the fold-mean template -- a genuine shape prior, so ARAP starts high and FALLS.
  basis  the network's own predicted PCA shape Vb -- the correct statement for a
         refinement (only the residual must be locally rigid), but ARAP then starts at
         ~0 and can only rise. Pass Vb= to use it.
  pred   the predicted correspondence Vd itself.

The refinement uses ONLY the predicted correspondence and the observed target surface;
no ground truth. That is asserted at the point of use in RefineEnergy's docstring, and
the constructor signature is the proof -- there is no argument through which GT could
arrive.

Encoder is pluggable: ENCODER=dgcnn|pointnext|ptv3|diffusionnet|kpconv. Any non-dgcnn name
is imported from research/code/fam_<name>.py and must expose a class (`Encoder`, or the
per-module name in _ENC_CAND) constructible BY KEYWORD from `cin`/`C` alone, whose
    forward(pc (B,N,3), feat (B,N,cin-3)|None) -> per-point features (B,N,C')
C' is READ OFF a probe forward, so a sibling keeps its own width and the global vector is
pooled here. Any failure (module absent, symbol absent, signature mismatch, wrong output
rank, probe raises) prints the reason and falls back to the built-in DGCNN.
MEASURED, 2026-07-31: NONE of the four siblings currently satisfies this -- fam_ptv3.PTv3
and fam_kpconv.Encoder take a CONFIG OBJECT as their second argument, fam_diffusionnet's
Backbone.forward takes a spectral batch dict, and fam_pointnext exposes no encoder class.
All four fall back to DGCNN and say so. ENCODER is therefore a knob that does nothing today;
adopting a sibling means teaching it a keyword-constructible `Encoder`, not editing this.

    python research/code/fam_template.py                     # CPU smoke test, both directions
    FOLD=0 DIRECTION=s2t ENCODER=dgcnn python research/code/fam_template.py
"""
import os, inspect, importlib
import numpy as np
import torch
import torch.nn as nn

FOLD = int(os.environ.get("FOLD", "0"))
SEED = int(os.environ.get("SEED", "0"))
TEMPLATE = os.environ.get("TEMPLATE", f"scratch/template_f{FOLD}.npz")
DIRECTION = os.environ.get("DIRECTION", "t2s")
ENCODER = os.environ.get("ENCODER", "dgcnn")
C_WID = int(os.environ.get("C", "256"))
GK = int(os.environ.get("GK", "20"))              # encoder graph neighbours
K = int(os.environ.get("K", "32"))                # cross-attention window per control vertex
SCALE = float(os.environ.get("SCALE", "30.0"))
NBASIS = int(os.environ.get("NBASIS", "60"))      # dense-PCA coefficients predicted (0 = off)
BASIS_SD = float(os.environ.get("BASIS_SD", "3.0"))
MAX_OFF = float(os.environ.get("MAX_OFF", "6.0"))     # bound on a control displacement (mm)
MAX_PHI = float(os.environ.get("MAX_PHI", "8.0"))     # bound on the s2t template-space offset
SIG_S2T = float(os.environ.get("SIG_S2T", "2.5"))     # s2t kernel width (mm), learnable
KNRM = int(os.environ.get("KNRM", "16"))              # kNN for cloud normals
NL = 85

ITERS = int(os.environ.get("ITERS", "12"))            # local/global rounds (0 = refinement off)
CG_ITERS = int(os.environ.get("CG_ITERS", "12"))      # CG iterations per global step
NN_EVERY = int(os.environ.get("NN_EVERY", "4"))       # re-match to the cloud every n rounds
P2P_TRIM = float(os.environ.get("P2P_TRIM", "3.0"))   # reject matches beyond this (mm)
INIT = os.environ.get("INIT", "pred")                 # pred | snap (see _init_verts)
REST = os.environ.get("REST", "base")                 # base | basis | pred
W_P2P = float(os.environ.get("W_P2P", "1.0"))
W_NRM = float(os.environ.get("W_NRM", "0.1"))
W_EDGE = float(os.environ.get("W_EDGE", "0.5"))
W_ARAP = float(os.environ.get("W_ARAP", "2.0"))
W_CORR = float(os.environ.get("W_CORR", "1.0"))


# ------------------------------------------------------------------ shared primitives
def knn(q, pc, k):
    return torch.cdist(q, pc).topk(k, largest=False, dim=-1).indices


def edge_conv(gidx, feat, mlp):
    B, P, Cf = feat.shape; k = gidx.shape[-1]
    fj = torch.gather(feat, 1, gidx.reshape(B, P * k, 1).expand(-1, -1, Cf)).view(B, P, k, Cf)
    fi = feat[:, :, None, :].expand(-1, -1, k, -1)
    return mlp(torch.cat([fi, fj - fi], -1)).max(2).values


def gather_pts(x, idx):
    """x (B,N,D), idx (B,M,k) -> (B,M,k,D)"""
    B, M, k = idx.shape; D = x.shape[-1]
    return torch.gather(x, 1, idx.reshape(B, M * k, 1).expand(-1, -1, D)).view(B, M, k, D)


def cloud_normals(pc, k=KNRM):
    """unoriented per-point normals by local PCA; sign is irrelevant to both terms using them"""
    Q = gather_pts(pc, knn(pc, pc, k))
    Q = Q - Q.mean(2, keepdim=True)
    return torch.linalg.eigh(Q.transpose(-1, -2) @ Q)[1][..., 0]


def vertex_normals(V, F):
    """area-weighted vertex normals of the deformed template (differentiable)"""
    v0, v1, v2 = V[:, F[:, 0]], V[:, F[:, 1]], V[:, F[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    out = torch.zeros_like(V).index_add(1, F.reshape(-1), fn.repeat_interleave(3, dim=1))
    return out / out.norm(dim=-1, keepdim=True).clamp(min=1e-9)


def transport(V, tri, bw):
    """the 85 landmarks as FIXED barycentric points on V -- the only landmark output path"""
    return (bw[None, :, :, None] * V[:, tri]).sum(2)


# ------------------------------------------------------------------ encoders
class DGCNNEncoder(nn.Module):
    """the shipped 1.273mm model's backbone, reused verbatim as the reference encoder"""
    def __init__(self, cin=3, C=C_WID, gk=GK):
        super().__init__()
        self.ec1 = nn.Sequential(nn.Linear(2 * cin, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.ec2 = nn.Sequential(nn.Linear(2 * 64, 128), nn.ReLU())
        self.ec3 = nn.Sequential(nn.Linear(2 * 128, 128), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(320, C), nn.ReLU())
        self.mix = nn.Sequential(nn.Linear(2 * C, C), nn.ReLU())
        self.gk = gk

    def forward(self, pc, feat=None):
        pos = pc / SCALE
        gidx = knn(pos, pos, self.gk)
        x = pos if feat is None else torch.cat([pos, feat], -1)
        h1 = edge_conv(gidx, x, self.ec1)
        h2 = edge_conv(gidx, h1, self.ec2)
        h3 = edge_conv(gidx, h2, self.ec3)
        h = self.fuse(torch.cat([h1, h2, h3], -1))
        return self.mix(torch.cat([h, h.max(1, keepdim=True).values.expand(-1, pc.shape[1], -1)], -1))


# class names to try in fam_<name>.py, after "Encoder", when adopting a sibling encoder
_ENC_CAND = {"ptv3": ("PTv3",), "diffusionnet": ("Backbone",),
             "pointnext": ("PointNeXtEncoder",)}


class _EncAdapter(nn.Module):
    """normalise a sibling encoder's return to per-point features (B,N,C')"""
    def __init__(self, mod):
        super().__init__(); self.mod = mod

    def forward(self, pc, feat=None):
        h = self.mod(pc, feat)
        return h[0] if isinstance(h, (tuple, list)) else h


def _ctor_kwargs(cls, cin, C):
    """kwargs for cls.__init__ built by NAME, so a sibling renaming its width argument still
    binds. Yields nothing when a parameter without a default is not one we can supply --
    that is the honest signal that the class is not an encoder we can drive."""
    known = dict(cin=cin, in_dim=cin, c_in=cin, C=C, c=C, dim=C, width=C, wid=C)
    try:
        par = list(inspect.signature(cls.__init__).parameters.values())[1:]
    except (TypeError, ValueError):
        return
    kw, ok = {}, True
    for p in par:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if p.name in known:
            kw[p.name] = known[p.name]
        elif p.default is p.empty:
            ok = False
    if ok:
        yield kw
        if "cin" in kw and len(kw) > 1:
            yield dict(cin=cin)                     # let the sibling keep its own width


def make_encoder(name=ENCODER, cin=3, C=C_WID, gk=GK, npts=512):
    """-> (encoder module, per-point width). Falls back to dgcnn, loudly, on any failure."""
    if name == "dgcnn":
        return DGCNNEncoder(cin, C, gk), C
    fails = []
    try:
        mod = importlib.import_module(f"fam_{name}")
        probe = torch.randn(1, npts, 3) * 6.0
        pf = None if cin == 3 else torch.randn(1, npts, cin - 3)
        for cname in ("Encoder",) + _ENC_CAND.get(name, ()):
            cls = getattr(mod, cname, None)
            if cls is None:
                continue
            tried = list(_ctor_kwargs(cls, cin, C))
            if not tried:
                fails.append(f"fam_{name}.{cname}{inspect.signature(cls.__init__)} needs an "
                             f"argument this module cannot supply")
            for kw in tried:
                try:
                    enc = cls(**kw)
                    with torch.no_grad():
                        h = enc(probe, pf)
                    h = h[0] if isinstance(h, (tuple, list)) else h
                    assert h.dim() == 3 and h.shape[:2] == (1, npts), tuple(h.shape)
                except Exception as e:                       # wrong signature or contract
                    fails.append(f"fam_{name}.{cname}({', '.join(kw)}): "
                                 f"{type(e).__name__}: {e}")
                    continue
                print(f"  encoder '{name}' -> fam_{name}.{cname}, per-point width "
                      f"{h.shape[-1]}", flush=True)
                return _EncAdapter(enc), int(h.shape[-1])
    except Exception as e:
        fails.append(f"{type(e).__name__}: {e}")
    why = " | ".join(fails) or "no candidate class matched the encoder contract"
    print(f"  ! encoder '{name}' unusable ({why}) -> falling back to dgcnn", flush=True)
    return DGCNNEncoder(cin, C, gk), C


# ------------------------------------------------------------------ template pack
def template_pack(z, nbasis=NBASIS, fold=None):
    """build_template.py's arrays -> plain tensors (no mesh library needed).

    `z` is an npz handle OR the plain dict train_family.py hands over in meta['artefacts'].
    fold= makes the fold-safety proof mandatory: pointing TEMPLATE at another fold's file
    would silently train on ears that are validation ears here, and nothing downstream
    could tell. Reshape is by explicit n, not -1, so nbasis=0 (no PCA head) is legal.
    """
    keys = set(z.files) if hasattr(z, "files") else set(z)
    assert {"fold", "train_ear_mask"} <= keys, (
        "template artefact carries no fold/train_ear_mask, so it cannot prove it was built "
        "from this fold's TRAINING ears only -- refusing it (constraint 2)")
    assert fold is None or int(z["fold"]) == fold, \
        f"template artefact is fold {int(z['fold'])}, not {fold} -- loading it would leak"
    n = len(z["template_V"])
    kb = int(min(nbasis, len(z["eig"])))
    base = z["mean_V"] if "mean_V" in keys else z["template_V"]
    return dict(V0=torch.tensor(np.asarray(z["template_V"])).float(),
                base=torch.tensor(np.asarray(base)).float(),
                F=torch.tensor(np.asarray(z["template_F"])).long(),
                tri=torch.tensor(np.asarray(z["bary_tri"])).long(),
                bw=torch.tensor(np.asarray(z["bary_w"])).float(),
                nbr=torch.tensor(np.asarray(z["nbr"])).long(),
                nbr_w=torch.tensor(np.asarray(z["nbr_w"])).float(),
                nbr_mask=torch.tensor(np.asarray(z["nbr_mask"])).float(),
                edges=torch.tensor(np.asarray(z["edges"])).long(),
                ctrl=torch.tensor(np.asarray(z["ctrl_idx"])).long(),
                skin_idx=torch.tensor(np.asarray(z["skin_idx"])).long(),
                skin_w=torch.tensor(np.asarray(z["skin_w"])).float(),
                comps=torch.tensor(np.asarray(z["comps"])[:kb]).float().reshape(kb, n, 3),
                eig=torch.tensor(np.asarray(z["eig"])[:kb]).float())


def load_template(path=TEMPLATE, nbasis=NBASIS, fold=None):
    return template_pack(np.load(path, allow_pickle=True), nbasis, fold)


# ------------------------------------------------------------------ network
class TemplateNet(nn.Module):
    def __init__(self, tpl, cin=3, C=C_WID, direction=DIRECTION, k=K, max_off=MAX_OFF,
                 encoder=ENCODER):
        super().__init__()
        # persistent=False: the template is CONSTANT and rebuilt from the npz at construction,
        # so it has no business in a state_dict the trainer clones on every eval.
        for name, t in tpl.items():
            self.register_buffer(name, t, persistent=False)
        n, M = len(self.base), len(self.ctrl)
        self.n, self.M, self.dir, self.k, self.max_off = n, M, direction, k, max_off
        self.kb = len(self.comps)
        self.register_buffer("wm", self.nbr_w * self.nbr_mask, persistent=False)
        self.enc, C = make_encoder(encoder, cin, C, GK)
        self.C = C
        self.basis = nn.Sequential(nn.Linear(2 * C, 256), nn.ReLU(),
                                   nn.Linear(256, max(self.kb, 1)))
        if direction == "t2s":
            self.emb = nn.Embedding(M, 32)          # only t2s conditions on control identity
            self.attn = nn.Sequential(nn.Linear(C + 3 + 32, 128), nn.ReLU(),
                                      nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
            self.disp = nn.Sequential(nn.Linear(2 * C + 32 + 3, 256), nn.ReLU(),
                                      nn.Dropout(0.1), nn.Linear(256, 128), nn.ReLU(),
                                      nn.Linear(128, 3))
        else:
            self.phi = nn.Sequential(nn.Linear(C + 3, 256), nn.ReLU(),
                                     nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 3))
            self.log_sig = nn.Parameter(torch.tensor(float(np.log(SIG_S2T))))

    # ---- shape basis (coarse) -------------------------------------------------
    def basis_shape(self, g):
        if self.kb == 0:
            return self.base[None].expand(g.shape[0], -1, -1), None
        a = BASIS_SD * self.eig.clamp(min=1e-9).sqrt() * torch.tanh(self.basis(g))
        return self.base[None] + torch.einsum("bk,knd->bnd", a, self.comps), a

    # ---- control displacement (fine), one head per direction -------------------
    def ctrl_disp_t2s(self, pc, h, q):
        idx = knn(q, pc, self.k)
        fK, pK = gather_pts(h, idx), gather_pts(pc, idx)
        rel = (pK - q[:, :, None, :]) / SCALE
        e = self.emb.weight[None, :, None, :].expand(q.shape[0], self.M, self.k, -1)
        w = torch.softmax(self.attn(torch.cat([fK, rel, e], -1)).squeeze(-1), -1)[..., None]
        ctx = torch.cat([(w * fK).sum(2), fK.max(2).values], -1)
        pt = (w * pK).sum(2)                                  # attention-picked surface point
        raw = self.disp(torch.cat([ctx, e[:, :, 0], (pt - q) / SCALE], -1))
        return self.max_off * torch.tanh(raw)

    def ctrl_disp_s2t(self, pc, h, q):
        phi = pc + MAX_PHI * torch.tanh(self.phi(torch.cat([h, pc / SCALE], -1)))
        sig = self.log_sig.exp().clamp(0.25, 20.0)
        w = torch.softmax(-torch.cdist(q, phi) ** 2 / (2 * sig ** 2), dim=-1)   # (B,M,N)
        return w @ pc - q                                     # convex hull of OBSERVED points

    def forward(self, pc, feat=None):
        h = self.enc(pc, feat)
        Vb, a = self.basis_shape(torch.cat([h.mean(1), h.max(1).values], -1))
        q = Vb[:, self.ctrl]
        d = self.ctrl_disp_t2s(pc, h, q) if self.dir == "t2s" else self.ctrl_disp_s2t(pc, h, q)
        Vd = Vb + (self.skin_w[..., None] * d[:, self.skin_idx]).sum(2)     # fixed skinning
        return dict(Vd=Vd, Vb=Vb, coef=a, ctrl_disp=d,
                    lm=transport(Vd, self.tri, self.bw))

    # ---- TRAINING-FOLD ONLY: pseudo-correspondence + landmark supervision -------
    def losses(self, out, corr_target=None, lm_target=None):
        L = {}
        if corr_target is not None:
            L["corr"] = ((out["Vd"] - corr_target) ** 2).sum(-1).mean()
            L["basis"] = ((out["Vb"] - corr_target) ** 2).sum(-1).mean()
        if lm_target is not None:
            L["lm"] = ((out["lm"] - lm_target) ** 2).sum(-1).mean()
        L["total"] = sum(L.values())
        return L


# ------------------------------------------------------------------ refinement
def arap_rotations(V, e0, wm, nbr):
    """per-vertex rotation by 3x3 SVD: argmin_{R in SO(3)} sum_j w_ij ||e_ij - R e0_ij||^2.

    S_i = sum_j w_ij e0_ij e_ij^T, S_i = U S V^T, R_i = V U^T (Sorkine-Alexa), with the
    last right singular vector flipped when that would otherwise give a reflection.
    """
    e = V[:, nbr] - V[:, :, None, :]
    S = torch.einsum("nd,bndk,bndl->bnkl", wm, e0, e)
    U, _, Vh = torch.linalg.svd(S)
    Vt = Vh.transpose(-1, -2)
    Ut = U.transpose(-1, -2)
    sgn = torch.det(Vt @ Ut).sign()[..., None, None]
    return torch.cat([Vt[..., :2], Vt[..., 2:] * sgn], -1) @ Ut


def nearest(V, pc, chunk=4096):
    ds, js = [], []
    for a in range(0, V.shape[1], chunk):
        d, j = torch.cdist(V[:, a:a + chunk], pc).min(-1)
        ds.append(d); js.append(j)
    return torch.cat(ds, 1), torch.cat(js, 1)


def quad_grad(fn, x):
    x = x.detach().requires_grad_(True)
    with torch.enable_grad():
        y = fn(x)
    return torch.autograd.grad(y, x)[0]


def cg_solve(fn, x0, iters):
    """Minimise the exact quadratic fn by conjugate gradients, starting at x0.

    fn(x) = x'Ax - 2b'x + c, so 0.5*grad fn(x) = Ax - b: A p is read off autograd as
    0.5*(grad fn(p) - grad fn(0)) and no matrix is ever assembled. CG minimises fn over
    growing Krylov subspaces from x0, so fn(x_k) is non-increasing. The batch shares the
    scalar step sizes, i.e. this is CG on the block-diagonal system -- the TOTAL energy
    descends monotonically, individual batch items need not.
    """
    g0 = quad_grad(fn, torch.zeros_like(x0))
    x = x0.detach().clone()
    r = -0.5 * quad_grad(fn, x)                          # b - A x
    p, rs = r.clone(), (r * r).sum()
    for _ in range(iters):
        if rs <= 1e-14:
            break
        Ap = 0.5 * (quad_grad(fn, p) - g0)
        al = rs / (p * Ap).sum().clamp(min=1e-20)
        x = x + al * p
        r = r - al * Ap
        rs2 = (r * r).sum()
        p = r + (rs2 / rs.clamp(min=1e-20)) * p
        rs = rs2
    return x.detach()


class RefineEnergy:
    """The refinement objective and its quadratic majorizer.

    NO-GT ASSERTION, AT THE POINT OF USE: the only inputs are `Vd` (the network's predicted
    correspondence), `rest` (template geometry the network chose), and `pc`/`nrm` (the
    OBSERVED target cloud plus its own unoriented local-PCA normals). There is no argument
    through which ground truth, a landmark file or a per-ear tuned constant could arrive,
    and nothing here reads a global other than the W_* weights -- so this class cannot leak
    even by accident. Constraint 2 is structural, not a convention.

    Every term is a mean in mm^2. `quad` is exactly quadratic in V given the LOCAL step's
    rotations/edge directions and the current match, and majorizes the true energy that
    `report` measures, tightly at the V the local step was fitted at (the smoke test
    asserts both halves of that claim).
    """

    def __init__(self, net, Vd, rest, pc, nrm=None, trim=P2P_TRIM):
        self.wm, self.nbr, self.ed, self.F = net.wm, net.nbr, net.edges, net.F
        self.Vd, self.pc = Vd.detach(), pc.detach()
        self.nrm = (cloud_normals(pc) if nrm is None else nrm).detach()
        self.trim = trim
        self.e0 = rest[:, self.nbr] - rest[:, :, None, :]
        self.L0 = (rest[:, self.ed[:, 0]] - rest[:, self.ed[:, 1]]).norm(dim=-1)
        self.wsum = self.wm.sum().clamp(min=1e-9) * Vd.shape[0]

    def match(self, V):
        """nearest observed point per template vertex + trimming mask (data, not variables)"""
        with torch.no_grad():
            d, j = nearest(V, self.pc)
            self.q = gather_pts(self.pc, j[..., None])[:, :, 0]
            self.nq = gather_pts(self.nrm, j[..., None])[:, :, 0]
            self.m = (d < self.trim).float()
            self.msum = self.m.sum().clamp(min=1e-9)
            self.nsum = (self.m[:, :, None] * self.wm[None]).sum().clamp(min=1e-9)

    def local(self, V):
        """LOCAL step: SVD rotations and unit edge directions, fixed for the global step."""
        with torch.no_grad():
            R = arap_rotations(V, self.e0, self.wm, self.nbr)
            self.Re0 = torch.einsum("bnkl,bndl->bndk", R, self.e0)
            ev = V[:, self.ed[:, 0]] - V[:, self.ed[:, 1]]
            ln = ev.norm(dim=-1, keepdim=True)
            # u must be a UNIT vector for ||x - L0 u||^2 >= (|x| - L0)^2 to hold. A collapsed
            # edge (two vertices snapped to the same cloud point) would otherwise give u = 0
            # and BREAK the majorizer, so it gets an arbitrary unit direction instead.
            self.u = torch.where(ln > 1e-6, ev / ln.clamp(min=1e-9),
                                 torch.tensor([1.0, 0.0, 0.0]).to(ev).expand_as(ev))

    def _common(self, V):
        e = V[:, self.nbr] - V[:, :, None, :]
        arap = (self.wm[..., None] * (e - self.Re0) ** 2).sum() / self.wsum
        p2p = (self.m * ((V - self.q) * self.nq).sum(-1) ** 2).sum() / self.msum
        ring = (self.m[:, :, None] * self.wm[None]
                * (e * self.nq[:, :, None, :]).sum(-1) ** 2).sum() / self.nsum
        corr = ((V - self.Vd) ** 2).sum(-1).mean()
        return arap, p2p, ring, corr

    def quad(self, V):
        """the exactly-quadratic majorizer minimised by the global step"""
        arap, p2p, ring, corr = self._common(V)
        ev = V[:, self.ed[:, 0]] - V[:, self.ed[:, 1]]
        edge = ((ev - self.L0[..., None] * self.u) ** 2).sum(-1).mean()
        return W_P2P * p2p + W_NRM * ring + W_EDGE * edge + W_ARAP * arap + W_CORR * corr

    def report(self, V):
        """the TRUE energy (exact edge-length term) plus non-optimised monitors"""
        with torch.no_grad():
            arap, p2p, ring, corr = self._common(V)
            ev = (V[:, self.ed[:, 0]] - V[:, self.ed[:, 1]]).norm(dim=-1)
            edge = ((ev - self.L0) ** 2).mean()
            tot = W_P2P * p2p + W_NRM * ring + W_EDGE * edge + W_ARAP * arap + W_CORR * corr
            cos = (self.m * (1 - (vertex_normals(V, self.F) * self.nq).sum(-1) ** 2)).sum() \
                / self.msum
            d, _ = nearest(V, self.pc)
            out = dict(total=tot, quad=self.quad(V), arap=arap, p2plane=p2p, normal=ring,
                       normal_cos=cos, edge=edge, corr=corr, surf_mean=d.mean(),
                       matched=self.m.mean())
        return {k: round(float(v), 6) for k, v in out.items()}


def _init_verts(Vd, pc, init):
    """pred: start at the predicted correspondence. snap: send every template vertex to its
    nearest OBSERVED point first -- a correspondence-destroying start that tests whether the
    energy can repair a bad match, and the setting in which ARAP visibly falls."""
    if init != "snap":
        return Vd.detach()
    with torch.no_grad():
        _, j = nearest(Vd.detach(), pc)
        return gather_pts(pc, j[..., None])[:, :, 0]


def refine(net, Vd, pc, Vb=None, nrm=None, iters=ITERS, cg=CG_ITERS, init=INIT, rest=REST,
           log=False):
    """Deform the predicted template onto the observed surface, then transport the 85.

    Returns (deformed vertices, landmarks (B,85,3), energy history). The landmarks come
    from `transport` and nowhere else, so the output path holds no free XYZ regression.
    """
    Vd = Vd.detach()
    if iters <= 0:
        return Vd, transport(Vd, net.tri, net.bw), []
    Vr = {"base": net.base[None].expand_as(Vd), "basis": Vd if Vb is None else Vb.detach(),
          "pred": Vd}[rest]
    en = RefineEnergy(net, Vd, Vr, pc, nrm)
    V = _init_verts(Vd, pc, init)
    en.match(V); en.local(V)
    hist = [en.report(V)]
    for it in range(iters):
        if it and it % NN_EVERY == 0:
            en.match(V)
            hist.append(dict(en.report(V), rematch=1))
        en.local(V)                                       # LOCAL: rotations + edge directions
        V = cg_solve(en.quad, V, cg)                      # GLOBAL: CG on the majorizer
        hist.append(en.report(V))
        if log:
            h = hist[-1]
            print(f"    it{it:3d} total {h['total']:9.5f} arap {h['arap']:9.5f} "
                  f"p2plane {h['p2plane']:8.5f} normal {h['normal']:8.5f} "
                  f"edge {h['edge']:8.5f} corr {h['corr']:8.5f} "
                  f"surf {h['surf_mean']:6.4f}", flush=True)
    return V, transport(V, net.tri, net.bw), hist


# ------------------------------------------------------------------ train_family adapter
class MODEL(nn.Module):
    """Family C as train_family.py's REGISTRY["template"] resolves it.

    'pred' is the 85 landmarks TRANSPORTED through the template's FIXED barycentric map, so
    the trained output path contains no free XYZ regression and no per-landmark head.

    LEAKAGE (constraint 2): the dense pseudo-correspondence target sits in a buffer indexed
    by EAR and is reachable only from loss(). Its rows come from the artefact's `used_ear`,
    which `train_ear_mask` proves is a subset of this fold's training ears, so a validation
    ear maps to row -1 and contributes nothing -- the mask is the mechanism, not a comment.

    AUGMENT is None deliberately. The template is a CANONICAL-frame buffer and
    default_augment rotates/scales the cloud and the target but cannot rotate a buffer, so
    this family would train against an inconsistent frame while every shape check passed.
    """
    DEFAULTS = dict(direction=DIRECTION, encoder=ENCODER, width=C_WID, gk=GK, k=K,
                    nbasis=NBASIS, max_off=MAX_OFF, w_corr=1.0, w_basis=0.1, use_nrm=False)
    SEARCH_SPACE = dict(direction=["t2s", "s2t"], k=[16, 32, 64], nbasis=[0, 30, 60, 120],
                        max_off=[3.0, 6.0, 10.0], width=[128, 256],
                        w_corr=[0.0, 0.3, 1.0, 3.0], w_basis=[0.0, 0.1, 1.0])
    NEEDS = ("nrm",) if os.environ.get("CFG_USE_NRM", "0") not in ("0", "false", "False") \
        else ()
    ROTATES = ()
    SAMPLES = 1
    AUGMENT = None

    def __init__(self, cfg, meta):
        super().__init__()
        z = meta["artefacts"]
        assert z, ("FAMILY=template needs ARTEFACTS=scratch/template_f<FOLD>.npz from "
                   "research/code/build_template.py")
        self.net = TemplateNet(template_pack(z, int(cfg["nbasis"]), meta["fold"]),
                               cin=3 + 3 * bool(cfg["use_nrm"]), C=int(cfg["width"]),
                               direction=cfg["direction"], k=int(cfg["k"]),
                               max_off=float(cfg["max_off"]), encoder=cfg["encoder"])
        self.use_nrm = bool(cfg["use_nrm"])
        self.w_corr, self.w_basis = float(cfg["w_corr"]), float(cfg["w_basis"])
        used = np.asarray(z["used_ear"]).astype(np.int64)
        mask = np.asarray(z["train_ear_mask"]).astype(bool)
        assert mask[used].all() and int(mask.sum()) == len(used), \
            "used_ear and train_ear_mask disagree -- the artefact cannot prove fold safety"
        row = np.full(len(mask), -1, np.int64); row[used] = np.arange(len(used))
        self.register_buffer("corr_row", torch.tensor(row), persistent=False)
        self.register_buffer("corr", torch.tensor(np.asarray(z["corr_V"])).float(),
                             persistent=False)

    def forward(self, b):
        out = self.net(b["pc"], b["nrm"] if self.use_nrm else None)
        return dict(pred=out["lm"], Vd=out["Vd"], Vb=out["Vb"])

    def loss(self, out, tg, batch):
        """The ONLY place a target is read. `tg` is the trainer's training-fold GT; `corr`
        is the training-fold pseudo-correspondence, masked to rows that exist."""
        L = ((out["pred"] - tg) ** 2).sum(-1).mean()
        r = self.corr_row[batch["ear"]]
        m = r >= 0
        if bool(m.any()) and (self.w_corr or self.w_basis):
            t = self.corr[r[m]]
            L = L + self.w_corr * ((out["Vd"][m] - t) ** 2).sum(-1).mean() \
                  + self.w_basis * ((out["Vb"][m] - t) ** 2).sum(-1).mean()
        return L


# ------------------------------------------------------------------ smoke test
def _synthetic_template(nu=26, nv=22):
    """a bumpy grid patch standing in for the ear template (CPU smoke test only)"""
    u, v = np.meshgrid(np.linspace(-15, 15, nu), np.linspace(-12, 12, nv), indexing="ij")
    z = 3.5 * np.sin(u / 5.0) * np.cos(v / 4.0)
    V = np.stack([u, v, z], -1).reshape(-1, 3)
    ij = np.arange(nu * nv).reshape(nu, nv)
    a, b = ij[:-1, :-1].ravel(), ij[1:, :-1].ravel()
    c, d = ij[1:, 1:].ravel(), ij[:-1, 1:].ravel()
    F = np.concatenate([np.stack([a, b, c], -1), np.stack([a, c, d], -1)])
    return V, F


def _smoke():
    import scipy.sparse as sp
    from build_template import (cotan_adjacency, padded_neighbours, farthest_point,
                                skinning, anchor_operator)
    from nicp import edges_from_faces
    torch.manual_seed(SEED); rs = np.random.RandomState(SEED)
    V, F = _synthetic_template()
    n = len(V)
    E = edges_from_faces(F)
    nbr, nbw, nbm = padded_neighbours(cotan_adjacency(V, F))
    ctrl = farthest_point(V, 96)
    ski, skw, sig = skinning(V, ctrl, 6)
    bf = rs.randint(0, len(F), NL)
    bwt = rs.dirichlet(np.ones(3), NL)
    # 4 smooth PCA-like components, orthonormalised over the 3n vector
    Cm = np.stack([np.stack([np.sin(V[:, 0] / f), np.cos(V[:, 1] / f), np.sin(V[:, 2] / f)], -1)
                   for f in (6.0, 9.0, 13.0, 20.0)]).reshape(4, -1)
    Cm = np.linalg.qr(Cm.T)[0].T
    print(f"synthetic template: {n} verts {len(F)} faces {len(E)} edges | "
          f"ARAP degree {nbr.shape[1]} | {len(ctrl)} ctrl | skinning sigma {sig:.2f}mm")
    A = anchor_operator(n, F[bf], bwt)
    assert isinstance(A, sp.csr_matrix) and A.shape == (NL, n)

    # B=2 synthetic targets: a known smooth warp of the template, sampled as a cloud
    B, NPTS, NE = 2, 2048, 8
    warp, clouds = [], []
    for b in range(B):
        w = V + np.stack([1.5 * np.sin(V[:, 1] / 7 + b), 1.2 * np.cos(V[:, 0] / 8 - b),
                          2.0 * np.sin(V[:, 0] / 9 + V[:, 1] / 11)], -1)
        fi = rs.randint(0, len(F), NPTS)
        bb = rs.dirichlet(np.ones(3), NPTS)
        clouds.append((bb[..., None] * w[F[fi]]).sum(1))
        warp.append(w)
    pc = torch.tensor(np.stack(clouds)).float()

    # exactly the npz build_template.py writes, so template_pack/MODEL are tested on the
    # real key names. Ears 0,1 (subject 0) are "training"; 2..7 stand in for validation.
    tmask = np.zeros(NE, bool); tmask[:B] = True
    art = dict(fold=np.int64(FOLD), train_ear_mask=tmask, used_ear=np.arange(B, dtype=np.int32),
               corr_V=np.stack(warp).astype(np.float32),
               template_V=V.astype(np.float32), template_F=F.astype(np.int32),
               mean_V=V.astype(np.float32), bary_tri=F[bf].astype(np.int32),
               bary_w=bwt.astype(np.float32), nbr=nbr, nbr_w=nbw, nbr_mask=nbm,
               edges=E.astype(np.int32), ctrl_idx=ctrl.astype(np.int32),
               skin_idx=ski, skin_w=skw, comps=Cm.astype(np.float32),
               eig=np.float32([9.0, 4.0, 2.0, 1.0]))
    tpl = template_pack(art, 4, FOLD)
    corr_t = torch.tensor(np.stack(warp)).float()
    lm_t = transport(corr_t, tpl["tri"], tpl["bw"])

    for direction in ("t2s", "s2t"):
        net = TemplateNet(tpl, direction=direction, k=16)
        npar = sum(p.numel() for p in net.parameters())
        out = net(pc)
        L = net.losses(out, corr_target=corr_t, lm_target=lm_t)
        L["total"].backward()
        gn = sum(float(p.grad.norm()) for p in net.parameters() if p.grad is not None)
        print(f"\n[{direction}] params {npar:,} | Vd {tuple(out['Vd'].shape)} "
              f"ctrl_disp {tuple(out['ctrl_disp'].shape)} lm {tuple(out['lm'].shape)} | "
              f"loss {float(L['total']):.3f} grad-norm {gn:.3f}")
        assert out["lm"].shape == (2, NL, 3), out["lm"].shape
        assert torch.isfinite(out["lm"]).all() and gn > 0
        for init in ("pred", "snap"):
            print(f"  [{direction}/init={init}/rest=base] ARAP local-global refinement")
            Vr, lm, hist = refine(net, out["Vd"], pc, Vb=out["Vb"], iters=6, init=init,
                                  log=True)
            assert lm.shape == (2, NL, 3), lm.shape
            assert torch.isfinite(lm).all()
            blk = [h for h in hist if not h.get("rematch")][:NN_EVERY + 1]
            assert all(b["total"] <= a["total"] + 1e-6 for a, b in zip(blk, blk[1:])), \
                "energy increased inside a match block -- the majorizer is not tight"
            assert all(h["quad"] >= h["total"] - 1e-6 for h in hist), "Q does not majorize E"
            assert abs(hist[0]["quad"] - hist[0]["total"]) < 1e-6, \
                "Q is not TIGHT where the local step was fitted"
            print(f"  [{direction}/init={init}] refined lm {tuple(lm.shape)} | ARAP "
                  f"{hist[0]['arap']:.5f} -> {hist[-1]['arap']:.5f} | total "
                  f"{hist[0]['total']:.5f} -> {hist[-1]['total']:.5f} | surf_mean "
                  f"{hist[0]['surf_mean']:.4f} -> {hist[-1]['surf_mean']:.4f}mm | "
                  f"landmark move {float((lm - out['lm']).norm(dim=-1).mean()):.3f}mm")
            if init == "snap":
                assert hist[-1]["arap"] < hist[0]["arap"], "ARAP energy did not decrease"
        _, lm0, h0 = refine(net, out["Vd"], pc, iters=0)
        assert lm0.shape == (2, NL, 3) and h0 == []

    # ---- the train_family.py contract, driven through train_family's OWN plumbing ----
    import train_family as T
    Fam = T.resolve_family("template")       # the class the harness will actually construct
    assert Fam.__name__ == "MODEL" and Fam.AUGMENT is None and Fam.SAMPLES == 1
    cfg = {**T.TRAIN_DEFAULTS, **Fam.DEFAULTS}
    for nb, nm in ((4, "nbasis=4"), (0, "nbasis=0 (no PCA head)")):
        m = Fam({**cfg, "nbasis": nb, "k": 16},
                  dict(nl=NL, contours=T.CONTOURS, scale=SCALE, npts=NPTS, fold=FOLD,
                       dev="cpu", n_train_ears=B, artefacts=art))
        data = dict(clouds=pc[:, None].expand(-1, 2, -1, -1).contiguous(),
                    coarse=lm_t, true=lm_t, E=B, M=2, N=NPTS, extra={})
        for ears, why in (([0, 1], "TRAIN ears: corr rows exist"),
                          ([0, 0], "repeated ear")):
            b = T._flatten_samples(T.make_batch(data, ears, [[0], [1]], 1), 1)
            o = m(b)
            L = T.default_loss(o, lm_t, m, b)
            assert o["pred"].shape == (2, NL, 3), o["pred"].shape
            L.backward()
            gn = sum(float(p.grad.norm()) for p in m.parameters() if p.grad is not None)
            npar = sum(p.numel() for p in m.parameters())
            print(f"  [MODEL {nm} ears={ears}] params {npar:,} pred {tuple(o['pred'].shape)} "
                  f"loss {float(L):.3f} grad-norm {gn:.2f}   ({why})")
            assert gn > 0 and torch.isfinite(o["pred"]).all()
            m.zero_grad()
    # a NON-training ear has no correspondence row, so the corr terms must vanish: the
    # mask is what stops a validation ear's dense target existing at all (constraint 2).
    data["E"] = NE
    data["clouds"] = data["clouds"][[0, 1] * (NE // 2)]
    data["coarse"] = data["true"] = lm_t[[0, 1] * (NE // 2)]
    b = T._flatten_samples(T.make_batch(data, [6, 7], [[0], [1]], 1), 1)
    o = m(b)
    assert (m.corr_row[b["ear"]] < 0).all(), "a validation ear was given a corr row -- LEAK"
    lm_only = float(((o["pred"] - lm_t) ** 2).sum(-1).mean())
    assert abs(float(T.default_loss(o, lm_t, m, b)) - lm_only) < 1e-5
    print(f"  [MODEL ears=[6, 7]] corr rows -1, loss == landmark MSE {lm_only:.3f}  "
          f"(no dense target exists for a non-training ear)")
    print("\nOK")


if __name__ == "__main__":
    _smoke()
