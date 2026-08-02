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

  HEAD=heatmap    85 per-vertex heatmaps -> masked softmax over the TOP-K vertices ->
                  expected position (this is what carries the gradient), then a bounded
                  tangent offset which is projected EXACTLY onto an incident triangle.
                  The output is a genuine barycentric point of a real triangle, so it
                  lies ON the surface by construction (the annotated landmarks lie
                  0.0212 mm from the emitted surface on average, p99 0.15 mm; raw
                  predictions sit ~0.17 mm off it, and snapping alone was worth
                  1.329 -> 1.309 mm).
  HEAD=coordfield a dense per-vertex CANONICAL TEMPLATE COORDINATE field u: V -> R^3.
                  Landmarks are FIXED points of the template's canonical space, so the
                  85 landmarks are transferred by inverting u: find the target vertex
                  whose predicted canonical coordinate is nearest the template landmark,
                  then solve for that landmark's clamped barycentric coordinates inside
                  the fan triangle MEASURED IN CANONICAL SPACE and apply those same
                  weights to the vertices' TARGET-space positions.
                  *** NOT RUNNABLE TODAY: it needs `tmpl_lm` (85 landmarks in the fold's
                  template canonical space) and ideally `canon_uv` (per-vertex NICP
                  correspondence). Neither artefact exists. __init__ refuses rather than
                  substituting something. See "WHAT IS STILL MISSING" below. ***

The backbone is genuinely intrinsic: learned per-channel diffusion time applied
spectrally, plus direction-aware spatial-gradient features (inner products of per-vertex
tangent gradients under a learned complex channel mixing, i.e. DiffusionNet's
rotation-equivariant gradient features), plus a pointwise MLP. INFEAT=hks is pure heat
kernel signature; the default xyzhks adds the canonical-frame xyz/normal, which is
legitimately available and informative -- but xyz is never the sole signal, and
INFEAT=xyz exists only as the ablation that proves the intrinsic channels carry signal.

-------------------------------------------------------------------------------------
INTEGRATION -- this file was written against a tensor contract that build_mesh_data.py
never emitted, and was never trained. Everything below is now routed through
research/code/mesh_batch.py, which is the ONE bridge from the ragged
scratch/mesh_data.npz (+ scratch/mesh_spec.npz) to a padded batch, and which
research/code/fam_vheat.py shares. The defects that were fixed to get here are listed at
the bottom of this docstring so they are not reintroduced.
-------------------------------------------------------------------------------------

LEAKAGE. NEEDS is empty; cls.BATCH returns only geometry from mesh_batch.MeshStore, which
never loads lm_vert/lm_face/lm_bary; ground truth is read only inside loss(). `tmpl_lm`
(HEAD=coordfield) is a FOLD-SCOPED artefact handed in through train_family.py's ARTEFACTS
mechanism, which asserts the fold and that no validation ear built it.

THE DECODER'S CEILING IS NOT THE PROBLEM. HEAD=heatmap shares its decoder with
fam_vheat.py, whose `decoder_ceiling` measures it on 40 real ears of
scratch/mesh_data.npz: nearest vertex 0.3415 mm, + exact fan projection 0.0328 mm, and
0.1446 mm for a heatmap that is EXACTLY the training target (k=32, sigma=1.0 mm). So
this head can express any answer down to ~0.14 mm; if it fails it will be because the
peak lands on the wrong vertex, which is a PHASE failure and exactly the thing
HEAD=coordfield was designed for.

WHAT IS STILL MISSING (do not paper over it)
  * HEAD=coordfield needs ARTEFACTS=<npz with tmpl_lm (85,3), fold, train_ear_mask>.
  * `canon_uv` / `canon_ok` (dense correspondence supervision) are not produced by any
    script here, so w_field is inert even once tmpl_lm exists.
  * scratch/mesh_data.npz is DECIMATED (MAXV=12000 -> 0.85/0.99/1.10 mm spacing) while
    the native crops are 19.5k-52.9k vertices at ~0.67 mm. Rebuilding with MAXV=0 raises
    the decoder ceiling (see fam_vheat.decoder_ceiling) at ~4 GB of artefact.

ENV
  MESH_DATA scratch/mesh_data.npz   ragged geometry
  MESH_SPEC scratch/mesh_spec.npz   eigenpairs; read only because KEIG > 0
  KEIG      128    eigenpairs LOADED. Read at MODULE level, not from cfg, because
                   cls.BATCH is a class-level hook the trainer calls without an instance
                   -- __init__ asserts cfg['keig'] matches so CFG_KEIG cannot silently
                   diverge from what was loaded.
  HEAD      heatmap | coordfield
  STRICT    1      re-check the padding invariants every forward

    FAMILY=diffusionnet FOLD=0 SEED=0 EPOCHS=600 CFG_BS=6 \
        python3 research/code/train_family.py
    python research/code/fam_diffusionnet.py        # CPU smoke test, real connectivity

DEFECTS FIXED (each one blocked a real run; all were invisible to the old smoke test,
which drove the model directly on hand-packed synthetic tensors and never went through
train_family.py)
  1. no module-level `MODEL`, so resolve_family('diffusionnet') asserted out.
  2. `__init__(self, head=HEAD, width=WIDTH, ...)`; the trainer calls `cls(cfg, meta)`, so
     cfg bound to `head` and the very first assert fired.
  3. no DEFAULTS / NEEDS / ROTATES / SAMPLES / BATCH / AUGMENT. With no BATCH the batch
     carries no mesh at all; with no AUGMENT the DEFAULT augmenter rotates pc, coarse and
     the TARGET and leaves `verts` untouched -- every shape check passes and the model
     trains against an inconsistent frame.
  4. forward returned {'lm': ...}; train_family.evaluate reads out['pred'].
  5. loss returned a TUPLE (total, terms); the trainer calls .backward() on it.
  6. the tensor contract did not exist. `tanb`, `grad_nbr`, `grad_wx`, `grad_wy`, `vfan`,
     `vfan_mask` are emitted by nothing. build_mesh_data.py ships basis_x/basis_y, nbr,
     grad_x/grad_y (+ the diagonal grad_xd/grad_yd) and no fan at all; mesh_batch.py
     builds the fan and does the ragged->padded gather.
  7. `tanb` (B,V,2,3) and `vfan` (B,V,T,2) are ndim>=4, and train_family._flatten_samples
     squeezes axis 1 of EVERY ndim>=4 batch tensor when SAMPLES == 1. Both would have
     arrived as (B,2,3) / (B,T,2) with no error anywhere. They are now basis_x/basis_y
     and fan1/fan2, all ndim<=3, and mesh_batch's smoke test asserts that property.
  8. the heatmap softmax ran over ALL ~11 000 vertices. At initialisation that puts the
     expected position at the ear's centroid and the useful gradient is ~1e-4 of the
     total; the tail also never fully vanishes. It is now a top-k softmax (k vertices,
     default 32 ~ a 3 mm geodesic disc), which is also what fam_vheat.py does, so the two
     families differ in BACKBONE and not in decoder.
  9. HEAD=coordfield silently produced garbage when `tmpl_lm` was absent (it is read out
     of the batch). It now refuses at construction.
"""
import os, sys, math, time
import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from mesh_batch import (store, mesh_augment, ring_gather, fan_project, gather_pts,
                        synth_artefact, MESH_DATA, MESH_SPEC)

NL = 85
SCALE = 30.0                      # mm normalizer, same as gpu_screen.py
CONTOURS = [(0, 24), (25, 54), (55, 74), (75, 84)]

KEIG = int(os.environ.get("KEIG", "128"))          # eigenpairs LOADED (class-level, see above)
HEAD = os.environ.get("HEAD", "heatmap")           # heatmap | coordfield
STRICT = int(os.environ.get("STRICT", "1"))        # re-check the mass-padding invariant
                                                   # every forward. Costs one device sync
                                                   # per step; set 0 once a driver has
                                                   # verified its packed tensors.

# --------------------------------------------------------------------------------------
# TENSOR CONTRACT -- supplied by mesh_batch.MeshStore.pad(), which is the executable
# specification. Ears in a batch have DIFFERENT vertex counts, so everything is padded to
# Vmax and `vmask` is authoritative. NOTHING here is ndim >= 4 (defect 7).
# --------------------------------------------------------------------------------------
CONTRACT = dict(
    verts=("(B,V,3) f32", "vertex positions in the per-ear canonical frame, mm; EXACTLY 0 "
                          "on padding"),
    vmask=("(B,V) bool", "True for real vertices; authoritative everywhere"),
    nv=("(B,) i64", "real vertex count per ear"),
    mass=("(B,V) f32", "lumped (barycentric) vertex areas, mm^2. EXACTLY 0 on padding -- "
                       "this is what keeps padding out of the spectral transform"),
    evals=("(B,K) f32", "smallest K generalized eigenvalues of the cotan Laplacian "
                        "(L phi = lambda M phi), ascending, >= 0, units 1/mm^2"),
    evecs=("(B,V,K) f32", "matching eigenvectors, M-orthonormal. EXACTLY 0 on padding"),
    nrm=("(B,V,3) f32", "unit vertex normals (area-weighted), consistently oriented"),
    basis_x=("(B,V,3) f32", "tangent frame e1; grad_x is expressed in it"),
    basis_y=("(B,V,3) f32", "tangent frame e2; (e1, e2, n) right-handed"),
    nbr=("(B,V,D) i64", "1-ring neighbour ids, LOCAL to the padded row; padded slots point "
                        "at the vertex itself, so their difference is 0"),
    nbr_mask=("(B,V,D) bool", "True for real one-ring slots"),
    grad_x=("(B,V,D) f32", "off-diagonal tangent-gradient weights, e1 component: "
                           "grad_e1 u [i] = sum_j grad_x[i,j] (u[j] - u[i]). 0 on padding"),
    grad_y=("(B,V,D) f32", "same, e2 component"),
    fan1=("(B,V,T) i64", "for each vertex, the SECOND corner of each incident triangle "
                         "(winding preserved); padded with the vertex itself"),
    fan2=("(B,V,T) i64", "the THIRD corner"),
    fan_mask=("(B,V,T) bool", "True for real incident triangles"),
    tmpl_lm=("(85,3) f32", "HEAD=coordfield ONLY, from ARTEFACTS: the 85 landmarks in the "
                           "TEMPLATE's canonical space. FOLD-SCOPED."),
    canon_uv=("(B,V,3) f32", "OPTIONAL, TRAIN EARS ONLY, read only inside loss(): the "
                             "ear's ground-truth canonical coordinate per vertex. NOT "
                             "PRODUCED BY ANYTHING YET."),
    canon_ok=("(B,V) bool", "which vertices have a trustworthy canon_uv"),
)


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
    """per-vertex tangent-plane gradient of every channel, in the (e1,e2) basis.

    Written as sum_j w_ij (x_j - x_i), which is IDENTICAL to build_mesh_data.py's
    grad_xd_i x_i + sum_j grad_x_ij x_j because grad_xd = -sum_j grad_x (verified in
    mesh_batch.py's smoke test), and needs one fewer tensor.
    """
    xj = ring_gather(x, nbr)
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
        gr, gi = tangent_grad(d, b["nbr"], b["grad_x"], b["grad_y"])
        y = self.mlp(torch.cat([x, d, self.gradfeat(gr, gi)], -1))
        return self.norm(x + y) * b["vmask"][..., None]


class Backbone(nn.Module):
    def __init__(self, C, blocks, dropout, infeat, nhks, tlo, thi):
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
    """85 per-vertex heatmaps -> masked TOP-K softmax -> expected position, then a bounded
    tangent offset at the peak vertex projected exactly onto an incident triangle.

    REACH. The offset is bounded by the peak vertex's 1-ring radius, but the fan the point
    is projected onto belongs to the top-k vertex NEAREST the offset point, not to the peak
    -- so the decoder is not confined to one 1-ring. What it still cannot do is move the
    PEAK: if the heatmap maximum is a millimetre off in phase along the contour, the
    top-k window (k vertices ~ a 3 mm geodesic disc at 0.99 mm spacing) is where the
    answer has to live. That is exactly the failure HEAD=coordfield exists to route
    around, by deciding phase with a dense global field instead of a local peak.
    """
    def __init__(self, C, dropout, topk, rho):
        super().__init__()
        self.topk, self.rho = topk, rho
        self.logit = nn.Sequential(nn.Linear(C, C), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(C, NL))
        self.emb = nn.Embedding(NL, 32)
        self.off = nn.Sequential(nn.Linear(C + 3 + 32, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x, b):
        B, V, _ = x.shape
        vm = b["vmask"]
        logit = self.logit(x).masked_fill(~vm[..., None], -1e9)              # (B,V,NL)
        k = V if self.topk <= 0 else min(self.topk, int(b["nv"].min()))
        topv, topi = logit.topk(k, dim=1)                                    # (B,k,NL)
        if STRICT:
            assert bool(torch.gather(vm, 1, topi.reshape(B, -1)).all()), \
                "a PADDED vertex entered the top-k softmax"
        pos = gather_pts(b["verts"], topi.reshape(B, -1)).view(B, k, NL, 3)
        vstar, pstar = topi[:, 0], pos[:, 0]
        d2 = ((pos - pstar[:, None]) ** 2).sum(-1)                           # (B,k,NL) mm^2
        w = torch.softmax(topv - d2 / (2 * self.rho ** 2), dim=1)
        p_soft = (w[..., None] * pos).sum(1)                                 # (B,NL,3)
        spread = ((w * ((pos - p_soft[:, None]) ** 2).sum(-1)).sum(1)).clamp(min=0).sqrt()

        # bound the offset by the 1-ring radius so the candidate stays near the fan
        ring = (ring_gather(b["verts"], b["nbr"]) - b["verts"][:, :, None]
                ).norm(dim=-1).max(2).values.detach()
        r = gather_pts(ring[..., None], vstar)                               # (B,NL,1)
        e = self.emb.weight[None].expand(B, -1, -1)
        duv = r * torch.tanh(self.off(torch.cat(
            [gather_pts(x, vstar), (p_soft - pstar) / SCALE, e], -1)))
        q = p_soft + duv[..., 0:1] * gather_pts(b["basis_x"], vstar) \
                   + duv[..., 1:2] * gather_pts(b["basis_y"], vstar)
        j = (pos - q[:, None]).norm(dim=-1).argmin(1)
        vfan = torch.gather(topi, 1, j[:, None]).squeeze(1)
        lm, tri, bw, resid = fan_project(q, vfan, b)
        return dict(pred=lm, p_soft=p_soft, logit=logit, vstar=vstar, vfan=vfan, q=q,
                    tri=tri, bary=bw, snap_mm=resid, spread_mm=spread, topk=k)


# ------------------------------------------------------------------ HEAD 2: coord field
class CoordFieldHead(nn.Module):
    """dense per-vertex canonical TEMPLATE coordinate field, landmarks transferred by
    inverting it with exact barycentric interpolation on the fan of the nearest vertex."""
    def __init__(self, C, dropout, tau, tmpl_lm):
        super().__init__()
        self.field = nn.Sequential(nn.Linear(C, C), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(C, 3))
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
        self.register_buffer("tmpl_lm", tmpl_lm)

    def forward(self, x, b):
        vm = b["vmask"]
        L = self.tmpl_lm                                                    # (85,3)
        u = (L.mean(0) + SCALE * self.field(x)) * vm[..., None]             # (B,V,3)
        tau = torch.exp(self.log_tau).clamp(min=1e-3)
        # cdist, not a (B,V,85,3) difference tensor: at 11k vertices that intermediate is
        # 11 MB/ear and autograd keeps it
        d2 = torch.cdist(u, L[None].expand(u.shape[0], -1, -1)) ** 2        # (B,V,85)
        logit = (-d2 / (2 * tau ** 2)).masked_fill(~vm[..., None], -1e9)
        w = torch.softmax(logit, dim=1)                                     # over VERTICES
        p_soft = torch.einsum("bvl,bvd->bld", w, b["verts"])
        vstar = logit.argmax(1)                                             # (B,85)
        tgt = L[None].expand(x.shape[0], -1, -1)
        lm, tri, bw, resid = fan_project(tgt, vstar, b, coords=u, out_coords=b["verts"])
        return dict(pred=lm, p_soft=p_soft, u=u, logit=logit, vstar=vstar,
                    tri=tri, bary=bw, canon_resid=resid, tau=tau.detach())


# ------------------------------------------------------------------ model + loss
class FamDiffusionNet(nn.Module):
    DEFAULTS = dict(head=HEAD, width=128, blocks=4, keig=KEIG, dropout=0.1,
                    infeat="xyzhks", nhks=16, tlo=1.0, thi=400.0, tau=1.0,
                    topk=32, rho=2.0, sigma=1.0, tgt_gate=1.0,
                    # w_lm/w_soft are SQUARED MILLIMETRES (~1130 at init); w_ce is nats
                    # (~9). At 1/1/0.3 the coordinate terms carry ~99% of the gradient and
                    # they can only reweight the current top-k -- they cannot RELOCATE the
                    # peak vertex. Measured on the sibling family (fam_vheat, same decoder,
                    # 2 real ears, 80 steps): 1/1/1 -> 16.40 mm, 0.02/0.02/1 -> 0.50 mm.
                    w_lm=0.02, w_soft=0.02, w_ce=1.0, w_field=1.0,
                    lr=1.0e-3, bs=6, wd=1e-4,
                    aug_rot=0.35, aug_scale=0.0, aug_jit=0.0, aug_qjit=0.0, sub_frac=1.0)
    SEARCH_SPACE = dict(width=[96, 128, 192], blocks=[3, 4, 6], nhks=[8, 16, 32],
                        infeat=["hks", "xyz", "xyzhks"], topk=[16, 32, 64],
                        thi=[100.0, 400.0, 1600.0], sigma=[0.5, 1.0, 2.0],
                        w_lm=[0.005, 0.02, 0.05], w_soft=[0.005, 0.02, 0.05],
                        lr=[5e-4, 1e-3, 2e-3],
                        aug_rot=[0.0, 0.35, 0.8])
    NEEDS, ROTATES, SAMPLES = (), (), 1
    AUGMENT = mesh_augment                       # NOT default_augment: see mesh_batch.py

    @staticmethod
    def BATCH(ears, samples, meta):
        """Ragged mesh + eigenpairs -> padded batch. KEIG is module-level on purpose.

        meta['mesh_data'] / meta['mesh_spec'] override MESH_DATA / MESH_SPEC; the trainer
        sets neither, so a real run reads the environment. The smoke test sets them, which
        is the only way it can exercise THIS hook instead of a hand-built batch.
        """
        st = store(meta["dev"], keig=KEIG, path=meta.get("mesh_data"),
                   spec=meta.get("mesh_spec"))
        assert int(np.max(ears)) < st.ne, \
            f"ear {int(np.max(ears))} is outside the {st.ne}-ear mesh artefact"
        b = st.pad(ears, want_spec=KEIG > 0)
        tl = meta.get("artefacts", {}).get("tmpl_lm")
        if tl is not None:
            b["tmpl_lm"] = torch.as_tensor(np.asarray(tl)).float().to(meta["dev"])
        return b

    def __init__(self, cfg, meta):
        super().__init__()
        head = str(cfg["head"])
        assert head in ("heatmap", "coordfield"), f"head={head}"
        assert int(cfg["keig"]) == KEIG, (
            f"cfg keig={cfg['keig']} but the module loaded KEIG={KEIG}. cls.BATCH is read "
            f"BEFORE any instance exists, so the number of eigenpairs is a module/env "
            f"constant: use `KEIG=... FAMILY=diffusionnet ...`, not CFG_KEIG.")
        self.head_name, self.keig, self.cfg = head, int(cfg["keig"]), cfg
        C = int(cfg["width"])
        self.backbone = Backbone(C, int(cfg["blocks"]), float(cfg["dropout"]),
                                 str(cfg["infeat"]), int(cfg["nhks"]),
                                 float(cfg["tlo"]), float(cfg["thi"]))
        if head == "heatmap":
            self.head = HeatmapHead(C, float(cfg["dropout"]), int(cfg["topk"]),
                                    float(cfg["rho"]))
        else:
            tl = meta.get("artefacts", {}).get("tmpl_lm")
            assert tl is not None, (
                "HEAD=coordfield needs `tmpl_lm` (85,3) -- the 85 landmarks in the FOLD's "
                "template canonical space. No script in this repo emits it. Pass "
                "ARTEFACTS=<npz with tmpl_lm, fold, train_ear_mask>, which train_family.py "
                "fold-checks, or use HEAD=heatmap. Refusing rather than inventing one.")
            self.head = CoordFieldHead(C, float(cfg["dropout"]), float(cfg["tau"]),
                                       torch.as_tensor(np.asarray(tl)).float())

    def forward(self, b):
        # `mass` zero on padding is the ONE contract invariant the masking rests on: it is
        # what keeps padded vertices out of the spectral transform. Everything else is
        # masked explicitly (softmaxes, block outputs, 1-ring weights, triangle fans).
        if STRICT:
            assert float((b["mass"] * ~b["vmask"]).abs().max()) == 0.0, \
                "contract violation: mass must be EXACTLY 0 on padded vertices"
            assert float((b["evecs"] * ~b["vmask"][..., None]).abs().max()) == 0.0, \
                "contract violation: evecs must be EXACTLY 0 on padded vertices"
        b = dict(b)
        b["evals"] = b["evals"][:, :self.keig]
        b["evecs"] = b["evecs"][..., :self.keig]
        out = self.head(self.backbone(b), b)
        assert out["pred"].shape[1:] == (NL, 3), out["pred"].shape
        return out

    def loss(self, out, target, b):
        """target (B,85,3) GT landmarks in the canonical frame -- TRAIN-FOLD EARS ONLY.

        Returns a SCALAR (train_family.py calls .backward() on it); the breakdown lands in
        self.last_terms. Every term is restricted to real vertices: the softmax weights are
        already exactly 0 on padding, the cross-entropy target is a distribution over real
        vertices only, and the dense field term is averaged over canon_ok & vmask.
        """
        c = self.cfg
        vm = b["vmask"]
        terms = {"lm": ((out["pred"] - target) ** 2).sum(-1).mean(),
                 "soft": ((out["p_soft"] - target) ** 2).sum(-1).mean()}
        total = float(c["w_lm"]) * terms["lm"] + float(c["w_soft"]) * terms["soft"]
        if self.head_name == "heatmap" and float(c["w_ce"]) > 0:
            # SPATIAL target, not a one-hot: a hard argmax label cannot express sub-vertex
            # precision, and its gradient is zero the moment the peak vertex is correct.
            d = torch.cdist(target, b["verts"])                             # (B,NL,V) mm
            s = float(c["sigma"])
            t = torch.exp(-(d ** 2) / (2 * s * s))
            if float(c["tgt_gate"]) > 0:
                vt = d.masked_fill(~vm[:, None, :], 1e9).argmin(-1)
                cosn = torch.einsum("bld,bvd->blv", gather_pts(b["nrm"], vt),
                                    b["nrm"]).clamp(min=0)
                t = t * cosn ** float(c["tgt_gate"])
            t = (t * vm[:, None, :]).clamp(min=0)
            t = t / t.sum(-1, keepdim=True).clamp(min=1e-20)
            lp = torch.log_softmax(out["logit"], dim=1).transpose(1, 2)     # (B,NL,V)
            terms["ce"] = -(t * lp).sum(-1).mean()
            total = total + float(c["w_ce"]) * terms["ce"]
        if self.head_name == "coordfield" and "canon_uv" in b and float(c["w_field"]) > 0:
            ok = (b["canon_ok"] & vm).float()
            terms["field"] = (((out["u"] - b["canon_uv"]) ** 2).sum(-1) * ok).sum() \
                / ok.sum().clamp(min=1.0)
            total = total + float(c["w_field"]) * terms["field"]
        self.last_terms = {k: round(float(v.detach()), 5) for k, v in terms.items()}
        return total


MODEL = FamDiffusionNet


# ======================================================================================
# LOCAL PREPROCESSING for the smoke test only: the eigenpairs mesh_spec.npz would carry.
# The real artefact comes from build_mesh_data.py; this rebuilds them for a synthetic
# mesh_data.npz so the test drives the ACTUAL loader instead of hand-packed tensors.
# ======================================================================================
def synth_spec(mesh_path, spec_path, keig):
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    z = np.load(mesh_path)
    vp, dp = z["v_ptr"], z["deg_ptr"]
    E = len(vp) - 1
    evals = np.zeros((E, keig), np.float32)
    evecs = np.zeros((int(vp[-1]), keig), np.float16)
    for e in range(E):
        g0, g1 = int(vp[e]), int(vp[e + 1]); n = g1 - g0
        d0, d1 = int(dp[g0]), int(dp[g1])
        row = np.repeat(np.arange(n), np.diff(dp[g0:g1 + 1]))
        col = z["nbr"][d0:d1].astype(np.int64) - g0
        W = sp.coo_matrix((z["lap_w"][d0:d1].astype(np.float64), (row, col)),
                          shape=(n, n)).tocsc()
        L = (sp.diags(z["lap_diag"][g0:g1].astype(np.float64)) - W).tocsc()
        M = sp.diags(z["mass"][g0:g1].astype(np.float64)).tocsc()
        ev, ec = spla.eigsh(L, k=keig, M=M, sigma=-1e-8, which="LM")
        o = np.argsort(ev)
        evals[e], evecs[g0:g1] = np.clip(ev[o], 0, None), ec[:, o]
    np.savez(spec_path, evals=evals, evecs=evecs, v_ptr=vp,
             n_eig=np.full(E, keig, np.int32))
    return spec_path


# ======================================================================================
if __name__ == "__main__":
    t0 = time.time()
    torch.manual_seed(0); np.random.seed(0)
    dev = "cpu"
    K = int(os.environ.get("SMOKE_KEIG", "32"))
    KEIG = K                                          # the class-level constant BATCH reads
    print("=" * 86)

    tmp = os.environ.get("SMOKE_DIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "dnet"))
    os.makedirs(tmp, exist_ok=True)
    mp = synth_artefact(os.path.join(tmp, "synth_mesh.npz"))
    spp = synth_spec(mp, os.path.join(tmp, "synth_spec.npz"), K)
    st = store(dev, keig=K, path=mp, spec=spp)
    ns = [int(x) for x in st.nv]
    meta = dict(nl=NL, contours=CONTOURS, scale=SCALE, npts=2048, fold=0, dev=dev,
                n_train_ears=3, artefacts={}, mesh_data=mp, mesh_spec=spp)
    b = FamDiffusionNet.BATCH(np.array([0, 1]), None, meta)
    z = np.load(mp)
    b["coarse"] = torch.tensor(z["coarse"][:2]).float()
    B, P = b["verts"].shape[:2]
    print(f"synthetic RAGGED artefact through the REAL loader: 3 ears {ns}, "
          f"batch of 2 padded to {P}, {K} eigenpairs")
    for i, n in enumerate(ns[:2]):
        G = (b["evecs"][i, :n].T * b["mass"][i, :n]) @ b["evecs"][i, :n]
        print(f"  ear{i}: |evecs^T M evecs - I|_max {float((G-torch.eye(K)).abs().max()):.2e}"
              f"  lambda_1 {float(b['evals'][i,1]):.5f} 1/mm^2  area "
              f"{float(b['mass'][i].sum()):.1f} mm^2")
    assert all(v.dim() <= 3 for v in b.values() if torch.is_tensor(v)), \
        "a batch tensor is ndim>=4: train_family._flatten_samples would squeeze axis 1"
    print("  no batch tensor is ndim>=4 (defect 7)")

    # exact on-surface GT from the artefact's (face, barycentric) encoding
    tri0 = torch.tensor(z["faces"].astype(np.int64))[z["lm_face"][:2].astype(np.int64)] \
        - st.v_ptr[:2, None, None]
    gt = (torch.tensor(z["lm_bary"][:2]).float()[..., None]
          * gather_pts(b["verts"], tri0.reshape(2, -1)).view(2, NL, 3, 3)).sum(2)

    cfg0 = {**FamDiffusionNet.DEFAULTS, "keig": K, "width": 64, "blocks": 2, "nhks": 8}
    meta_cf = dict(meta, artefacts={"tmpl_lm": gt[0].numpy()})
    for head, m in (("heatmap", meta), ("coordfield", meta_cf)):
        cfg = {**cfg0, "head": head}
        net = FamDiffusionNet(cfg, m)
        npar = sum(p.numel() for p in net.parameters())
        bb = dict(b)
        if head == "coordfield":
            bb["canon_uv"] = b["verts"].clone(); bb["canon_ok"] = b["vmask"].clone()
        out = net(bb)
        assert out["pred"].shape == (2, NL, 3), out["pred"].shape
        loss = net.loss(out, gt, bb)
        assert torch.is_tensor(loss) and loss.dim() == 0, "loss must be a SCALAR (defect 5)"
        loss.backward()
        gs = {n: float(p.grad.norm()) for n, p in net.named_parameters() if p.grad is not None}
        assert len(gs) == len(list(net.parameters())), \
            f"no gradient reached {[n for n,_ in net.named_parameters() if n not in gs]}"
        assert all(np.isfinite(v) for v in gs.values()), "non-finite gradient"
        gn = sum(v ** 2 for v in gs.values()) ** .5
        rec = (out["bary"][..., None]
               * gather_pts(b["verts"], out["tri"].reshape(2, -1)).view(2, NL, 3, 3)).sum(2)
        assert float(out["bary"].min()) >= -1e-6 and \
            float((out["bary"].sum(-1) - 1).abs().max()) < 1e-5
        assert float((rec - out["pred"]).abs().max()) < 1e-4
        print(f"\nHEAD={head:10s} params {npar:,}  pred {tuple(out['pred'].shape)}  "
              f"loss {float(loss):.4f} {net.last_terms}")
        print(f"  grad-norm {gn:.4e}; log_t {gs['backbone.blocks.0.log_t']:.3e}, gradfeat "
              f"{gs['backbone.blocks.0.gradfeat.Wr.weight']:.3e} -> the intrinsic path is live")
        print(f"  learned t range {float(net.backbone.blocks[0].log_t.exp().min()):.2f}"
              f"-{float(net.backbone.blocks[0].log_t.exp().max()):.1f} mm^2; "
              + (f"k={out['topk']} spread {float(out['spread_mm'].mean()):.3f} mm snap "
                 f"{float(out['snap_mm'].mean()):.4f} mm" if head == "heatmap" else
                 f"canon resid {float(out['canon_resid'].mean()):.4f} tau "
                 f"{float(out['tau']):.3f}"))
        print(f"  on-surface: bary in the simplex, |reconstruction - pred|_max "
              f"{float((rec-out['pred']).abs().max()):.1e}")

        # padded vertices must not influence anything
        net.eval()
        ref = net(bb)["pred"].detach().clone()
        b2 = dict(bb)
        for key in ("verts", "nrm", "basis_x", "basis_y", "evecs"):
            t = bb[key].clone()
            for i, n in enumerate(ns[:2]):
                t[i, n:] = torch.randn_like(t[i, n:]) * 100
            b2[key] = t
        globals()["STRICT"] = 0
        d_mask = float((net(b2)["pred"].detach() - ref).abs().max())
        globals()["STRICT"] = 1
        assert d_mask == 0.0, f"padded vertices leaked into the output ({d_mask})"
        w_all = torch.softmax(net(bb)["logit"], dim=1)
        assert float(w_all.masked_select(~bb["vmask"][..., None]).abs().max()) == 0.0
        print(f"  padding: |delta|={d_mask:.1e} with junk in every padded row, softmax "
              f"weight on padding exactly 0")
        net.train()

    # HEAD=coordfield must REFUSE to be built without its fold-scoped artefact (defect 9)
    try:
        FamDiffusionNet({**cfg0, "head": "coordfield"}, meta)
        raise SystemExit("coordfield was built with no tmpl_lm -- it would emit garbage")
    except AssertionError as e:
        assert "tmpl_lm" in str(e)
    # CFG_KEIG must not be able to diverge from the loaded KEIG (defect: silent slicing)
    try:
        FamDiffusionNet({**cfg0, "keig": K + 1}, meta)
        raise SystemExit("cfg keig was allowed to diverge from the module KEIG")
    except AssertionError as e:
        assert "BEFORE any instance exists" in str(e)
    print("\n  refused: coordfield without tmpl_lm; cfg keig != module KEIG")

    # the diffusion operator really is the heat kernel: t -> inf goes to the mass-mean
    x = torch.randn(2, P, 1) * b["vmask"][..., None]
    big = spectral_diffuse(x, b["evals"], b["evecs"], b["mass"], torch.tensor([12.0]))
    mean = ((b["mass"][..., None] * x).sum(1) / b["mass"].sum(1)[:, None])
    err = float(((big - mean[:, None]) * b["vmask"][..., None]).abs().max())
    print(f"  heat-kernel check: |diffuse(t=e^12) - mass-mean|_max = {err:.2e}")
    assert err < 5e-3, err

    # the trainer's contract, checked as the trainer reads it
    for k in ("DEFAULTS", "NEEDS", "ROTATES", "SAMPLES", "BATCH", "AUGMENT"):
        assert hasattr(FamDiffusionNet, k), k
    assert MODEL is FamDiffusionNet
    assert FamDiffusionNet.AUGMENT.__name__ == "mesh_augment"
    assert not set(FamDiffusionNet.NEEDS)
    print(f"  FAMILY CONTRACT: MODEL present, NEEDS=(), AUGMENT=mesh_augment, "
          f"BATCH hook returns {len(b)} mesh tensors")
    print(f"\nSMOKE PASS in {time.time()-t0:.1f}s on {dev}")
    print("=" * 86)
