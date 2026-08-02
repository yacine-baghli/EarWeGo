"""
FAMILY vheat -- NATIVE-MESH PER-VERTEX HEATMAP LOCALISATION, with no spectral machinery.

Every model in this repo regresses 85 coordinates from a 1024-8192 point cloud whose
spacing is 1.09-2.19 mm, and the pooled OOF has sat at 1.1776 mm through seven DGCNN
variants, four backbones and every ensemble weighting that was tried. The mesh those
clouds are sampled from has ~1.0 mm vertex spacing after decimation (0.85/0.99/1.10 mm
min/median/max, scratch/mesh_data.npz) and the GT landmarks lie 0.021 mm from it (mean;
p99 0.15 mm). So the surface itself is two orders of magnitude more accurate than any
prediction, and NO model has ever used it as the output space. This family does:

    85 per-vertex logits  ->  masked softmax over the TOP-K vertices  ->  soft-argmax
    ->  bounded tangent offset  ->  exact barycentric point of a real incident triangle.

It is deliberately the SIMPLE alternative to fam_diffusionnet.py: no eigendecomposition,
no heat kernel, no MESH_SPEC (which is a 965 MB artefact and the slowest part of the mesh
pipeline). The backbone is `blocks` rounds of one-ring message passing using exactly the
operators build_mesh_data.py already shipped -- cotangent Laplacian, the two tangent
derivatives, ring mean, ring max -- over per-vertex features that are local geometry only.
If the heatmap representation is what unlocks sub-1 mm, this file should show it without
DiffusionNet's spectral quirks being a confound; if BOTH families fail, the failure is the
representation and not the backbone.

WHAT IS ACTUALLY NEW HERE, stated so it can be falsified:
  * the output is a point ON the scanned surface by construction (a convex barycentric
    combination of a real triangle), not a free R^3 vector that is snapped afterwards;
  * the gradient reaches the position through a SPATIAL soft-argmax, so the model can be
    wrong by a fraction of the vertex spacing, which a classification loss cannot express;
  * the localisation resolution is the mesh's, not the point cloud's.

THE THREE NUMBERS THAT BOUND IT (measured by `decoder_ceiling`, which is run before any
training claim -- see the CEILING=1 report in __main__):
  1. nearest-vertex distance      what a pure argmax classifier could ever reach;
  2. perfect-peak fan projection  what THIS decoder reaches when the peak is the right
                                  vertex and the offset is perfect;
  3. ideal-Gaussian round trip    what THIS decoder reaches when the heatmap is EXACTLY
                                  the training target, i.e. the bias the loss itself
                                  builds in as a function of (sigma, k, rho).
(3) is the honest ceiling of the method as trained, and it is larger than (2).

MEASURED, on scratch/mesh_data.npz, 40 real ears spread over the 340, against the ACTUAL
annotated landmarks (`CEILING=1 NEARS=40 python research/code/fam_vheat.py`):
    vertex spacing                             0.853 / 0.963 / 1.067 mm (min/median/max)
    1. nearest vertex (argmax floor)           0.3415 mm   p90 0.5469
    2. + exact fan projection                  0.0328 mm   p90 0.0748
    3. ideal-target round trip, k=32:  sigma 0.5 -> 0.1384   1.0 -> 0.1446
                                             1.5 -> 0.1690   3.0 -> 0.1982 mm
       (k=8 is worse at small sigma, k=64 worse at large sigma; see the full table)
    representation floor (landmark -> emitted surface)  mean 0.0194  p99 0.1232 mm
So the OUTPUT REPRESENTATION is not the binding constraint anywhere near the current
1.1776 mm: it costs ~0.14 mm, and even a heatmap that only ever picks the right VERTEX
and does nothing else would sit at 0.34 mm. Whatever stops this family will be the
heatmap's ability to pick the right vertex, not the decoding of it. sigma=1.0 mm is the
default because it is the best ceiling that still spreads gradient over 3-5 vertices.
CAVEAT: mesh_data.npz is DECIMATED (MAXV=12000). The native crops are 19.5k-52.9k
vertices at ~0.67 mm, which would push (1) to ~0.23 mm and (3) below 0.1 mm, at ~4 GB of
artefact. Nothing here was measured on the native mesh.

LOSS-WEIGHT SCALES ARE NOT FREE -- MEASURED, do not "tune" them back to 1/1/1.
The `lm`/`soft` terms are SQUARED MILLIMETRES and start at ~1130 mm^2; the `ce` term is
~9 nats. At w_lm=w_soft=w_ce=1 the coordinate terms carry ~99 % of the gradient, and they
can only REWEIGHT the current top-k -- nothing in them can move the peak vertex, which is
the only thing that matters while the peak is 30 mm wrong. Measured on 2 REAL ears (real
landmarks, width 64, 2 blocks, AdamW 3e-3, no augmentation, 80 full-batch steps, seed 0):
    w_lm=1    w_soft=1    w_ce=1     ->  16.40 mm   (fan residual drifts to 0.74 mm)
    w_lm=0.02 w_soft=0.02 w_ce=1     ->   0.50 mm   (fan residual 0.08 mm)   <- DEFAULT
    w_lm=0    w_soft=0    w_ce=1     ->   0.44 mm   but the offset head gets NO gradient,
                                         so `lm` (0.50 mm) ends up WORSE than the raw
                                         soft-argmax (0.26 mm): the coordinate terms are
                                         what train the refinement, they just must not
                                         dominate the term that relocates the peak.
0.02 is 1/50, i.e. it puts the coordinate terms at ~10-20 units against the CE's ~9 once
the error is a few mm. Reproduce before changing.

TRAINING SIGNAL (all three terms, because no one of them suffices)
  w_ce   * soft cross-entropy against a SPATIAL target: t_v ~ exp(-|v - gt|^2 / 2 sigma^2)
           over real vertices, sigma in MILLIMETRES (default 1.0 mm, i.e. one vertex
           spacing -- chosen off the measured ceiling table, not by taste), normalised to
           a distribution. Euclidean, not geodesic -- see CAVEAT below -- optionally gated
           by the normal agreement with the GT vertex to kill the opposite sheet of a fold.
  w_soft * squared error of the soft-argmax position. This is the term that carries
           sub-vertex precision; the CE term alone tops out at the vertex spacing.
  w_lm   * squared error of the final on-surface point.

CAVEAT, EUCLIDEAN vs GEODESIC. The spatial target uses Euclidean distance because a
geodesic distance field per landmark per ear is a preprocessing job that does not exist.
On the helix rim two sheets of the surface pass within ~1 mm of each other, so a
Euclidean Gaussian puts mass on the WRONG sheet. `tgt_gate` multiplies the target by
clamp(dot(n_v, n_gt), 0, 1)^tgt_gate, which removes the anti-parallel sheet and is the
cheap 90 % of what a geodesic would buy. It does not fix two sheets that are parallel.

SOFT-ARGMAX, IN INTERPRETABLE UNITS. Weights are
      w = softmax_over_topk( logit / tau  -  |v - v_peak|^2 / (2 rho^2) )
  * topk   k = 32 VERTICES. At 0.99 mm median spacing (0.94 mm^2 per vertex) that is a
           surface patch of ~30 mm^2, i.e. a geodesic disc of radius ~3.1 mm. Restricting
           to top-k matters: over all ~11 000 vertices the tail of the softmax drags the
           expectation toward the ear centroid.
  * rho    2.0 MILLIMETRES. A hard spatial prior around the peak vertex, so the estimate
           cannot wander more than a few mm from the peak whatever the logits do. This is
           the "temperature" with units.
  * tau    learnable, unitless, initialised to 1 -- it only rescales the logits. The
           REALISED spread is reported as spread_mm = sqrt(sum_k w_k |v_k - p_soft|^2),
           which is the number to look at, not tau.

LEAKAGE. NEEDS is empty, cls.BATCH returns only geometry from mesh_batch.MeshStore (which
never loads lm_vert/lm_face/lm_bary at all), and every ground-truth read is inside loss().
The population statistics used are none.

ENV (CFG_<NAME> in train_family.py overrides any of the DEFAULTS below)
  MESH_DATA scratch/mesh_data.npz   ragged artefact (read via mesh_batch.store)
  STRICT    1                       re-check the padding invariants every forward
                                    (a few device syncs per step; set 0 once verified)

    FAMILY=vheat FOLD=0 SEED=0 EPOCHS=600 CFG_BS=8 python3 research/code/train_family.py
    python research/code/fam_vheat.py               # CPU smoke test (synthetic mesh)
    CEILING=1 python research/code/fam_vheat.py     # decoder ceiling on the REAL mesh
"""
import os, sys, math, time
import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from mesh_batch import (store, mesh_augment, mesh_ops, ring_gather, ring_smooth,
                        curvature, fan_project, gather_pts, synth_artefact, MESH_DATA)

NL, SCALE = 85, 30.0
STRICT = int(os.environ.get("STRICT", "1"))


# ------------------------------------------------------------------ per-vertex features
def vfeat(b, rounds=(4, 12), cfeat=False):
    """Local geometry per vertex, plus the same features smoothed over larger one-ring
    neighbourhoods so block 0 already sees context.

    Channels (17 per scale):
      verts/SCALE (3)          position in the per-ear canonical frame
      nrm (3)                  unit vertex normal
      asinh(5*curv) (6)        k1, k2, H, K, H from the cotan Laplacian, |H n|; asinh
                               because 1/mm curvatures on an ear span 0.01..5 and a raw
                               feature would be dominated by a handful of rim vertices
      ring stats (5)           mean and max one-ring edge length (mm), mean normal
                               deviation 1-<n_i,n_j>, degree/6, log lumped area (mm^2)
    `rounds` are ADDITIONAL smoothed copies (0.5-lazy one-ring averaging, no parameters):
    (4, 12) gives scales of ~2 mm and ~4 mm on a 1 mm mesh. cin = 17 * (1 + len(rounds)).

    Nothing here has parameters and nothing requires grad, so the whole function is free
    in the backward pass.
    """
    v, n = b["verts"], b["nrm"]
    nm = b["nbr_mask"]
    xj = ring_gather(torch.cat([v, n], -1), b["nbr"])
    cnt = nm.sum(2, keepdim=True).clamp(min=1).float()
    el = (xj[..., :3] - v[:, :, None, :]).norm(dim=-1) * nm
    emean = el.sum(2, keepdim=True) / cnt
    emax = el.max(2, keepdim=True).values
    nvar = ((1 - (xj[..., 3:] * n[:, :, None, :]).sum(-1)) * nm).sum(2, keepdim=True) / cnt
    curv = torch.stack(curvature(b), -1)
    f = torch.cat([v / SCALE, n, torch.asinh(5.0 * curv), emean, emax, nvar,
                   cnt / 6.0, torch.log(b["mass"].clamp(min=1e-6))[..., None]], -1)
    if cfeat:                       # distance to the coarse init: legitimately available
        d = torch.cdist(v, b["coarse"])                                   # (B,V,85)
        f = torch.cat([f, torch.stack(
            [d[..., lo:hi + 1].min(-1).values for lo, hi in
             ((0, 24), (25, 54), (55, 74), (75, 84))] + [d.min(-1).values], -1) / SCALE], -1)
    out, s = [f], f
    for r in rounds:
        s = ring_smooth(s, b, r)
        out.append(s)
    return torch.cat(out, -1) * b["vmask"][..., None]


FEAT_BASE = 17


# ------------------------------------------------------------------ backbone
class MPBlock(nn.Module):
    """one round of one-ring message passing with the shipped mesh operators.

    mesh_ops returns (cotan Laplacian, d/de1, d/de2, ring mean, ring max) from a SINGLE
    gather of x, which is the only memory-heavy op in the model. The two tangent
    derivatives are what make the block anisotropic -- a plain ring mean/max cannot tell
    "along the helix" from "across it", which is where 77 % of the error energy lives.
    They are expressed in the (basis_x, basis_y) frame that ships with the mesh and that
    mesh_augment rotates together with the vertices, so they are consistent under
    augmentation.
    """
    def __init__(self, C, dropout):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(6 * C, 2 * C), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(2 * C, C))
        self.norm = nn.LayerNorm(C)

    def forward(self, x, b):
        lap, gx, gy, mean, mx = mesh_ops(x, b)
        y = self.mlp(torch.cat([x, lap, gx, gy, mean, mx], -1))
        return self.norm(x + y) * b["vmask"][..., None]


# ------------------------------------------------------------------ head
class VHeatHead(nn.Module):
    """85 per-vertex logits -> top-k masked softmax -> soft-argmax -> tangent offset ->
    exact barycentric point of a real incident triangle.

    THE ORDER MATTERS. The peak vertex chooses the neighbourhood; the soft-argmax places
    the point inside it with sub-vertex precision; the offset lets the point leave the
    convex hull of the top-k vertices (a soft-argmax alone can never reach outside it, so
    it can never sit on a rim vertex's outer side); and the fan projection is what puts
    the answer exactly on the scanned surface. The triangle fan used for that projection
    belongs to the top-k vertex NEAREST the offset point -- not to the peak vertex -- which
    is what removes the 1-ring reach limit the fam_diffusionnet heatmap head documents.
    """
    def __init__(self, C, nl=NL, topk=32, rho=2.0, dropout=0.1):
        super().__init__()
        self.nl, self.topk, self.rho = nl, topk, rho
        self.logit = nn.Sequential(nn.Linear(C, C), nn.ReLU(), nn.Dropout(dropout),
                                   nn.Linear(C, nl))
        self.emb = nn.Embedding(nl, 32)
        self.off = nn.Sequential(nn.Linear(C + 3 + 32, 128), nn.ReLU(), nn.Linear(128, 2))
        self.log_tau = nn.Parameter(torch.zeros(()))

    def forward(self, x, b):
        B, V, _ = x.shape
        vm = b["vmask"]
        k = min(self.topk, int(b["nv"].min()))
        logit = self.logit(x).masked_fill(~vm[..., None], -1e9)              # (B,V,NL)
        topv, topi = logit.topk(k, dim=1)                                    # (B,k,NL)
        if STRICT:
            assert bool(torch.gather(vm, 1, topi.reshape(B, -1)).all()), \
                "a PADDED vertex entered the top-k softmax"
        pos = gather_pts(b["verts"], topi.reshape(B, -1)).view(B, k, self.nl, 3)
        vpeak, ppeak = topi[:, 0], pos[:, 0]                                 # (B,NL) (B,NL,3)
        d2 = ((pos - ppeak[:, None]) ** 2).sum(-1)                           # (B,k,NL) mm^2
        tau = self.log_tau.exp().clamp(0.05, 20.0)
        w = torch.softmax(topv / tau - d2 / (2 * self.rho ** 2), dim=1)
        p_soft = (w[..., None] * pos).sum(1)                                 # (B,NL,3)
        spread = ((w * ((pos - p_soft[:, None]) ** 2).sum(-1)).sum(1)).clamp(min=0).sqrt()

        ring = (ring_gather(b["verts"], b["nbr"]) - b["verts"][:, :, None]
                ).norm(dim=-1).max(2).values.detach()                        # (B,V) mm
        r = gather_pts(ring[..., None], vpeak)                               # (B,NL,1)
        h = gather_pts(x, vpeak)
        e = self.emb.weight[None].expand(B, -1, -1)
        duv = r * torch.tanh(self.off(torch.cat([h, (p_soft - ppeak) / SCALE, e], -1)))
        q = p_soft + duv[..., 0:1] * gather_pts(b["basis_x"], vpeak) \
                   + duv[..., 1:2] * gather_pts(b["basis_y"], vpeak)
        j = (pos - q[:, None]).norm(dim=-1).argmin(1)                        # (B,NL)
        vstar = torch.gather(topi, 1, j[:, None]).squeeze(1)
        lm, tri, bary, resid = fan_project(q, vstar, b)
        return dict(pred=lm, p_soft=p_soft, logit=logit, q=q, vpeak=vpeak, vstar=vstar,
                    tri=tri, bary=bary, snap_mm=resid, spread_mm=spread, topk=k,
                    tau=tau.detach())


# ------------------------------------------------------------------ family
class FamVHeat(nn.Module):
    # w_lm/w_soft are mm^2 against the CE's nats -- see the LOSS-WEIGHT SCALES block in the
    # module docstring. 1/1/1 does not train (16.4 mm vs 0.50 mm after 80 steps, measured).
    DEFAULTS = dict(width=128, blocks=4, dropout=0.1, topk=32, rho=2.0, sigma=1.0,
                    tgt_gate=1.0, w_lm=0.02, w_soft=0.02, w_ce=1.0, cfeat=0,
                    smooth=(4, 12), lr=1.0e-3, bs=8, wd=1e-4,
                    aug_rot=0.35, aug_scale=0.0, aug_jit=0.0, aug_qjit=0.0, sub_frac=1.0)
    SEARCH_SPACE = dict(width=[96, 128, 192], blocks=[3, 4, 6], topk=[16, 32, 64],
                        rho=[1.5, 2.0, 3.0], sigma=[1.0, 1.5, 2.5],
                        w_lm=[0.005, 0.02, 0.05], w_soft=[0.005, 0.02, 0.05],
                        lr=[5e-4, 1e-3, 2e-3], cfeat=[0, 1], aug_rot=[0.0, 0.35, 0.8])
    NEEDS, ROTATES, SAMPLES = (), (), 1
    AUGMENT = mesh_augment                       # NOT default_augment: see mesh_batch.py

    @staticmethod
    def BATCH(ears, samples, meta):
        """Ragged mesh -> padded batch. keig=0, so MESH_SPEC is never touched.

        meta['mesh_data'] overrides MESH_DATA; the trainer sets it for nobody, so a real
        run reads the environment. The smoke test sets it, which is the only way it can
        exercise THIS hook rather than a hand-built batch.
        """
        st = store(meta["dev"], keig=0, path=meta.get("mesh_data"))
        assert int(np.max(ears)) < st.ne, \
            f"ear {int(np.max(ears))} is outside the {st.ne}-ear mesh artefact"
        return st.pad(ears)

    def __init__(self, cfg, meta):
        super().__init__()
        C, nl = int(cfg["width"]), meta["nl"]
        self.nl, self.cfg = nl, cfg
        self.smooth = tuple(cfg["smooth"])
        self.cfeat = bool(int(cfg["cfeat"]))
        cin = (FEAT_BASE + (5 if self.cfeat else 0)) * (1 + len(self.smooth))
        self.cin = cin
        self.lift = nn.Sequential(nn.Linear(cin, C), nn.ReLU(), nn.Linear(C, C))
        self.inorm = nn.LayerNorm(C)
        self.blocks = nn.ModuleList([MPBlock(C, float(cfg["dropout"]))
                                     for _ in range(int(cfg["blocks"]))])
        self.head = VHeatHead(C, nl, int(cfg["topk"]), float(cfg["rho"]),
                              float(cfg["dropout"]))

    def forward(self, b):
        if STRICT:
            assert float((b["mass"] * ~b["vmask"]).abs().max()) == 0.0, \
                "contract violation: mass must be EXACTLY 0 on padded vertices"
            assert float((b["verts"] * ~b["vmask"][..., None]).abs().max()) == 0.0, \
                "contract violation: padded vertex positions must be 0"
        with torch.no_grad():
            f = vfeat(b, self.smooth, self.cfeat)
        x = self.inorm(self.lift(f)) * b["vmask"][..., None]
        for blk in self.blocks:
            x = blk(x, b)
        out = self.head(x, b)
        assert out["pred"].shape[1:] == (self.nl, 3), out["pred"].shape
        return out

    # ---- loss ------------------------------------------------------------------
    def loss(self, out, target, b):
        """target (B,85,3) canonical-frame GT -- TRAIN-FOLD EARS ONLY, reached nowhere else.

        The spatial CE target is built here and only here, so no forward pass can see it.
        """
        c = self.cfg
        vm = b["vmask"]
        d = torch.cdist(target, b["verts"])                                  # (B,NL,V) mm
        terms = {"lm": ((out["pred"] - target) ** 2).sum(-1).mean(),
                 "soft": ((out["p_soft"] - target) ** 2).sum(-1).mean()}
        total = float(c["w_lm"]) * terms["lm"] + float(c["w_soft"]) * terms["soft"]
        if float(c["w_ce"]) > 0:
            s = float(c["sigma"])
            t = torch.exp(-(d ** 2) / (2 * s * s))
            if float(c["tgt_gate"]) > 0:
                vt = d.masked_fill(~vm[:, None, :], 1e9).argmin(-1)          # (B,NL)
                ng = gather_pts(b["nrm"], vt)                                # (B,NL,3)
                cosn = torch.einsum("bld,bvd->blv", ng, b["nrm"]).clamp(min=0)
                t = t * cosn ** float(c["tgt_gate"])
            t = (t * vm[:, None, :]).clamp(min=0)
            t = t / t.sum(-1, keepdim=True).clamp(min=1e-20)
            lp = torch.log_softmax(out["logit"], dim=1).transpose(1, 2)      # (B,NL,V)
            terms["ce"] = -(t * lp).sum(-1).mean()
            total = total + float(c["w_ce"]) * terms["ce"]
        self.last_terms = {k: round(float(v.detach()), 5) for k, v in terms.items()}
        return total


MODEL = FamVHeat


# ======================================================================================
# THE CEILING. Run BEFORE believing any training number.
# ======================================================================================
def decoder_ceiling(b, gt, ks=(8, 16, 32, 64), sigmas=(0.5, 1.0, 1.5, 2.0, 3.0), rho=2.0):
    """How accurately can THIS decoder recover a point it is handed exactly right?

    Three regimes, all with a PERFECT heatmap:
      nn    nearest-vertex distance             -- a pure argmax classifier's floor
      fan   closest point on that vertex's fan  -- perfect peak + perfect offset
      ideal soft-argmax with the exact training target weights, then the same fan
            projection -- the bias the LOSS builds in, per (k, sigma)
    `gt` may be the annotated landmark (which sits ~0.02 mm off the emitted surface) or its
    exact surface projection; the caller says which.
    """
    B, L, _ = gt.shape
    d = torch.cdist(gt, b["verts"]).masked_fill(~b["vmask"][:, None], 1e9)   # (B,L,V)
    dnn, vnn = d.min(-1)
    _, _, _, dfan = fan_project(gt, vnn, b)
    rows = []
    for k in ks:
        dk, ik = torch.topk(d, k, dim=-1, largest=False)
        pos = gather_pts(b["verts"], ik.reshape(B, -1)).view(B, L, k, 3)
        pk = pos[:, :, 0]
        sp = ((pos - pk[:, :, None]) ** 2).sum(-1)
        for s in sigmas:
            # the exact training target, renormalised over the same top-k the head uses,
            # then the head's own spatial prior -- this IS the head with a perfect logit
            w = torch.softmax(-dk ** 2 / (2 * s * s) - sp / (2 * rho ** 2), dim=-1)
            p = (w[..., None] * pos).sum(2)
            j = (pos - p[:, :, None]).norm(dim=-1).argmin(-1)
            vs = torch.gather(ik, 2, j[..., None]).squeeze(-1)
            pf, _, _, _ = fan_project(p, vs, b)
            spread = ((w * ((pos - p[:, :, None]) ** 2).sum(-1)).sum(-1)).clamp(min=0).sqrt()
            rows.append(dict(k=k, sigma=s,
                             soft_mm=float((p - gt).norm(dim=-1).mean()),
                             decoded_mm=float((pf - gt).norm(dim=-1).mean()),
                             p90_mm=float(np.percentile((pf - gt).norm(dim=-1).cpu().numpy(), 90)),
                             spread_mm=float(spread.mean())))
    return dict(nn_mm=float(dnn.mean()), nn_p90_mm=float(np.percentile(dnn.cpu().numpy(), 90)),
                fan_mm=float(dfan.mean()),
                fan_p90_mm=float(np.percentile(dfan.cpu().numpy(), 90)), rows=rows)


def _print_ceiling(c, label):
    print(f"\n  DECODER CEILING [{label}]  (perfect heatmap, no training involved)")
    print(f"    nearest vertex (argmax classifier floor)  {c['nn_mm']:.4f} mm  "
          f"p90 {c['nn_p90_mm']:.4f}")
    print(f"    + exact fan projection (perfect offset)   {c['fan_mm']:.4f} mm  "
          f"p90 {c['fan_p90_mm']:.4f}")
    print(f"    ideal-target round trip:   k  sigma   soft_argmax   decoded    p90   spread")
    for r in c["rows"]:
        print(f"                              {r['k']:3d}  {r['sigma']:4.1f}    "
              f"{r['soft_mm']:8.4f}   {r['decoded_mm']:7.4f}  {r['p90_mm']:6.4f}  "
              f"{r['spread_mm']:6.3f} mm")


def real_mesh_ceiling(nears=40, path=None, data="scratch/screen_data_2048.npz"):
    """The number that matters: the ceiling on scratch/mesh_data.npz, against the ACTUAL
    annotated landmarks (not their surface projection), for `nears` real ears."""
    st = store("cpu", keig=0, path=path or MESH_DATA)
    z = np.load(st.path)
    ears = np.linspace(0, st.ne - 1, nears).astype(int)
    zz = np.load(data, allow_pickle=True)
    true = torch.tensor(zz["true"][ears]).float()
    sp = np.sqrt(np.asarray(z["area"])[ears] / (0.866 * np.asarray(z["n_vert"])[ears]))
    print(f"  {nears} ears, {int(st.nv[ears].min())}-{int(st.nv[ears].max())} vertices, "
          f"vertex spacing {sp.min():.3f}/{np.median(sp):.3f}/{sp.max():.3f} mm "
          f"(min/median/max)")
    out, chunk = [], 4
    for i in range(0, len(ears), chunk):
        e = ears[i:i + chunk]
        b = st.pad(e)
        out.append(decoder_ceiling(b, true[i:i + chunk]))
    agg = dict(nn_mm=float(np.mean([o["nn_mm"] for o in out])),
               nn_p90_mm=float(np.mean([o["nn_p90_mm"] for o in out])),
               fan_mm=float(np.mean([o["fan_mm"] for o in out])),
               fan_p90_mm=float(np.mean([o["fan_p90_mm"] for o in out])),
               rows=[{k: (v if k in ("k", "sigma") else
                          float(np.mean([o["rows"][j][k] for o in out])))
                      for k, v in out[0]["rows"][j].items()}
                     for j in range(len(out[0]["rows"]))])
    _print_ceiling(agg, f"REAL mesh_data.npz, {nears} ears, vs ANNOTATED landmarks")
    lmd = np.asarray(z["lm_dist"])[ears]
    print(f"    representation floor (annotated landmark -> emitted surface): "
          f"mean {lmd.mean():.4f} p99 {np.percentile(lmd, 99):.4f} max {lmd.max():.4f} mm")
    return agg


# ======================================================================================
if __name__ == "__main__":
    t0 = time.time()
    torch.manual_seed(0); np.random.seed(0)
    dev = "cpu"

    if int(os.environ.get("CEILING", "0")):
        print("=" * 86)
        print("DECODER CEILING ON THE REAL MESH -- no model, no training, no ground-truth leak")
        print("=" * 86)
        real_mesh_ceiling(int(os.environ.get("NEARS", "40")))
        print(f"\ndone in {time.time()-t0:.1f}s")
        sys.exit(0)

    print("=" * 86)
    tmp = os.environ.get("SMOKE_DIR", os.path.join(os.environ.get("TMPDIR", "/tmp"), "vheat"))
    os.makedirs(tmp, exist_ok=True)
    p = synth_artefact(os.path.join(tmp, "synth_mesh.npz"))
    st = store(dev, keig=0, path=p)
    ns = [int(x) for x in st.nv]
    cfg = {**FamVHeat.DEFAULTS}
    meta = dict(nl=NL, contours=[(0, 24), (25, 54), (55, 74), (75, 84)], scale=SCALE,
                npts=2048, fold=0, dev=dev, n_train_ears=2, artefacts={}, mesh_data=p)
    b = FamVHeat.BATCH(np.array([0, 1]), None, meta)       # the trainer's actual hook
    z = np.load(p)
    b["coarse"] = torch.tensor(z["coarse"][:2]).float()
    B, P = b["verts"].shape[:2]
    assert all(v.dim() <= 3 for v in b.values() if torch.is_tensor(v)), \
        "a batch tensor is ndim>=4: train_family._flatten_samples would squeeze axis 1"
    print(f"synthetic RAGGED batch through cls.BATCH: B=2 ears with {ns[:2]} vertices, "
          f"padded to {P}; no tensor is ndim>=4")

    # exact on-surface GT from the artefact's (face, barycentric) encoding
    tri = torch.tensor(z["faces"].astype(np.int64))[z["lm_face"][:2].astype(np.int64)] \
        - st.v_ptr[:2, None, None]
    gt = (torch.tensor(z["lm_bary"][:2]).float()[..., None]
          * gather_pts(b["verts"], tri.reshape(2, -1)).view(2, NL, 3, 3)).sum(2)

    for cf in (0, 1):
        net = FamVHeat({**cfg, "cfeat": cf, "width": 64, "blocks": 3}, meta)
        npar = sum(q.numel() for q in net.parameters())
        out = net(b)
        loss = net.loss(out, gt, b)
        loss.backward()
        gs = {n: float(q.grad.norm()) for n, q in net.named_parameters() if q.grad is not None}
        assert len(gs) == len(list(net.parameters())), \
            f"no gradient reached {[n for n,_ in net.named_parameters() if n not in gs]}"
        assert all(np.isfinite(v) for v in gs.values()), "non-finite gradient"
        assert out["pred"].shape == (2, NL, 3), out["pred"].shape
        gn = sum(v ** 2 for v in gs.values()) ** .5
        print(f"\ncfeat={cf}  cin={net.cin}  params {npar:,}  pred {tuple(out['pred'].shape)}  "
              f"loss {float(loss):.4f} {net.last_terms}")
        print(f"  grad-norm {gn:.4e} reaches logit {gs['head.logit.3.weight']:.3e}, "
              f"offset {gs['head.off.2.weight']:.3e}, log_tau {gs['head.log_tau']:.3e}, "
              f"block0 {gs['blocks.0.mlp.0.weight']:.3e}")
        print(f"  k={out['topk']} vertices  tau {float(out['tau']):.3f}  realised spread "
              f"{float(out['spread_mm'].mean()):.3f} mm  fan residual "
              f"{float(out['snap_mm'].mean()):.4f} mm")

    # 1. EXACTLY on the surface: the output is a convex barycentric point of a real triangle
    rec = (out["bary"][..., None]
           * gather_pts(b["verts"], out["tri"].reshape(2, -1)).view(2, NL, 3, 3)).sum(2)
    assert float(out["bary"].min()) >= -1e-6 and \
        float((out["bary"].sum(-1) - 1).abs().max()) < 1e-5, "barycentric weights left the simplex"
    assert float((rec - out["pred"]).abs().max()) < 1e-4
    print(f"\n  on-surface: weights in the simplex, |bary-reconstruction - pred|_max "
          f"{float((rec-out['pred']).abs().max()):.2e}")

    # 2. NO padded vertex may influence anything (the whole point of asserting the mask)
    net.eval()
    ref = net(b)["pred"].detach().clone()
    b2 = dict(b)
    for key in ("verts", "nrm", "basis_x", "basis_y"):
        t = b[key].clone()
        for i, n in enumerate(ns[:2]):
            t[i, n:] = torch.randn_like(t[i, n:]) * 100
        b2[key] = t
    # the padding invariant is `vmask`, so trash the padded rows and demand bit-identity.
    # STRICT would refuse the batch outright (which is the production behaviour), so it is
    # switched off for the length of this test -- the point is to prove the MASKING works,
    # not merely that the guard fires.
    b2["vmask"] = b["vmask"]
    globals()["STRICT"] = 0
    d_mask = float((net(b2)["pred"].detach() - ref).abs().max())
    globals()["STRICT"] = 1
    assert d_mask == 0.0, f"padded vertices leaked into the output ({d_mask})"
    try:
        net(b2)
        raise SystemExit("STRICT accepted a batch with non-zero padded vertex rows")
    except AssertionError as e:
        assert "padded vertex positions" in str(e)
    print(f"  padding-invariance |delta| = {d_mask:.1e} with junk in every padded row "
          f"(and STRICT refuses that batch outright)")
    lg = net(b)["logit"]
    w_all = torch.softmax(lg, dim=1)
    assert float(w_all.masked_select(~b["vmask"][..., None]).abs().max()) == 0.0, \
        "a padded vertex has non-zero softmax weight"
    print(f"  full-V softmax weight on padded vertices = "
          f"{float(w_all.masked_select(~b['vmask'][...,None]).abs().max()):.1e} (exactly 0)")
    net.train()

    # 3. the augmentation moves the mesh and the target as ONE rigid body
    g = torch.Generator(); g.manual_seed(3)
    ba, gta = mesh_augment(b, gt, {**cfg, "aug_rot": 1.2}, (), g)
    d0 = torch.cdist(gt[0], b["verts"][0, :ns[0]])
    d1 = torch.cdist(gta[0], ba["verts"][0, :ns[0]])
    assert float((d0 - d1).abs().max()) < 2e-3
    print(f"  mesh_augment: target-to-vertex distances invariant to "
          f"{float((d0-d1).abs().max()):.1e} mm")

    # 4. THE CEILING, on the synthetic mesh (the real-mesh number is CEILING=1)
    _print_ceiling(decoder_ceiling(b, gt, ks=(8, 32), sigmas=(0.5, 1.5, 3.0)),
                   f"synthetic, spacing "
                   f"{float((b['verts'][0,1]-b['verts'][0,0]).norm()):.2f} mm")

    print(f"\nSMOKE PASS in {time.time()-t0:.1f}s on {dev}")
    print("=" * 86)
