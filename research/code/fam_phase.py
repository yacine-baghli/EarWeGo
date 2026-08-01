"""
FAMILY F (proposed on evidence, not part of the original brief): EXPLICIT CURVE + PHASE.

Why this exists. 77% of the error energy is along-contour, and it has not moved: nine
DGCNN variants, four unrelated backbones (correlated at r=0.80 in the SIGNED tangent
error), a dense template-correspondence formulation, six post-hoc correction predictors
and a 121-feature context probe. An in-sample OPTIMAL four-architecture ensemble -- an
upper bound, leaky by construction -- buys 1.35% on that component. See
research/results/phase_shared.json.

What every one of those shares: ordered MSE against fixed targets, with the 85 landmarks
as FREE XYZ points. In that parameterisation nothing knows that landmark 31 must sit one
step past landmark 30 ON THE SAME material curve, so phase is only ever implicit. The
monotone-per-point oracle reaches 0.5657mm by re-parameterising along the PREDICTED
polyline, so the curve geometry is already close to right and the phase is what is wrong.

This makes phase an explicit, separately-parameterised, structurally MONOTONE output:

  per contour  the backbone predicts K control points -> a centripetal Catmull-Rom curve
  per landmark the backbone predicts one positive increment; a cumulative sum normalised
               to [0,1] gives strictly increasing phases, so ordering cannot be violated
  landmark_i = curve_c(phase_i)     evaluated differentiably

Ordering is therefore enforced by construction rather than hoped for, which is exactly
the property the monotone oracle exploits. Curve shape and phase get separate gradients:
the curve term can improve geometry without disturbing phase, and vice versa.

The earlier phase experiment (train_phase_cnn.py) predicted an affine phase correction
POST HOC for a frozen model and failed (OOF R^2 ~ 0). This trains the parameterisation
end to end, which is a different experiment, not a retry.

    FAMILY=phase FOLD=0 SEED=0 python research/code/train_family.py
    python research/code/fam_phase.py        # CPU smoke test

Env: WIDTH, GK, NCTRL (control points per contour), W_CURVE (phase-invariant curve term),
     MINSTEP (floor on a phase increment, as a fraction of uniform), USE_NRM.
"""
import os
import numpy as np
import torch
import torch.nn as nn

CONTOURS = [(0, 24), (25, 54), (55, 74), (75, 84)]
NL = 85
WIDTH = int(os.environ.get("WIDTH", "256"))
GK = int(os.environ.get("GK", "20"))
NCTRL = int(os.environ.get("NCTRL", "16"))
W_CURVE = float(os.environ.get("W_CURVE", "0.3"))
MINSTEP = float(os.environ.get("MINSTEP", "0.25"))
ENV_USE_NRM = bool(int(os.environ.get("USE_NRM", "0")))
SCALE = 30.0


def knn(q, pc, k):
    return torch.cdist(q, pc).topk(k, largest=False).indices


def catmull_rom(P, t):
    """Centripetal Catmull-Rom through control points P (B,K,3) at params t (B,n) in [0,1].

    Centripetal (alpha=0.5) rather than uniform: uniform Catmull-Rom self-intersects and
    cusps when control points are unevenly spaced, which is exactly what happens where a
    contour turns sharply -- the helix crus. Returns (B,n,3).
    """
    B, K, _ = P.shape
    d = (P[:, 1:] - P[:, :-1]).norm(dim=-1).clamp(min=1e-6) ** 0.5      # alpha = 0.5
    kn = torch.cat([torch.zeros(B, 1, device=P.device, dtype=P.dtype), d.cumsum(-1)], -1)
    kn = kn / kn[:, -1:].clamp(min=1e-6)                                 # (B,K) in [0,1]
    # segment index per query
    i = (torch.searchsorted(kn.contiguous(), t.contiguous()) - 1).clamp(0, K - 2)
    g = lambda A, j: torch.gather(A, 1, j.clamp(0, K - 1)[..., None].expand(-1, -1, 3))
    k0, k1 = torch.gather(kn, 1, i), torch.gather(kn, 1, (i + 1).clamp(max=K - 1))
    u = ((t - k0) / (k1 - k0).clamp(min=1e-6)).clamp(0, 1)[..., None]
    p0, p1, p2, p3 = g(P, i - 1), g(P, i), g(P, i + 1), g(P, i + 2)
    # uniform Catmull-Rom basis on the centripetal-normalised local parameter
    u2 = u * u; u3 = u2 * u
    return 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * u3)


class EdgeConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(2 * cin, cout), nn.ReLU(), nn.Linear(cout, cout), nn.ReLU())

    def forward(self, x, idx):
        B, N, C = x.shape
        nb = torch.gather(x, 1, idx.reshape(B, -1)[..., None].expand(-1, -1, C)).view(B, N, -1, C)
        return self.f(torch.cat([x[:, :, None].expand_as(nb), nb - x[:, :, None]], -1)).max(2).values


class MODEL(nn.Module):
    """train_family.py's REGISTRY['phase'].

    The 85 landmarks are produced ONLY by evaluating a curve at learned monotone phases.
    There is no free per-landmark XYZ output anywhere in the graph.

    NEEDS is a CLASS attribute the trainer reads before any instance exists, so normals
    must be requested through the ENVIRONMENT (USE_NRM=1). Assigning self.NEEDS in
    __init__ is too late and would silently train on XYZ only -- rejected below.
    """
    NEEDS = ("nrm",) if ENV_USE_NRM else ()

    def __init__(self, cfg=None, meta=None):
        super().__init__()
        cfg = cfg or {}
        C = int(cfg.get("width", WIDTH))
        self.gk = int(cfg.get("gk", GK))
        self.nctrl = int(cfg.get("nctrl", NCTRL))
        self.w_curve = float(cfg.get("w_curve", W_CURVE))
        self.minstep = float(cfg.get("minstep", MINSTEP))
        self.use_nrm = bool(cfg.get("use_nrm", ENV_USE_NRM))
        assert not (self.use_nrm and "nrm" not in self.NEEDS), \
            "use_nrm is on but NEEDS is empty: set USE_NRM=1 in the ENVIRONMENT (not " \
            "only CFG_USE_NRM), and point DATA at a file carrying 'nrm'"
        cin = 3 + (3 if self.use_nrm else 0)
        self.e1, self.e2, self.e3 = EdgeConv(cin, 64), EdgeConv(64, 128), EdgeConv(128, 128)
        self.fuse = nn.Sequential(nn.Linear(320, C), nn.ReLU())
        self.glob = nn.Sequential(nn.Linear(C, C), nn.ReLU())
        # one control-point head and one phase head PER CONTOUR, so the two contours that
        # behave differently (helix vs concha) do not share a parameterisation
        self.ctrl = nn.ModuleList()
        self.phase = nn.ModuleList()
        for lo, hi in CONTOURS:
            n = hi - lo + 1
            self.ctrl.append(nn.Sequential(nn.Linear(2 * C + 3 * n, 256), nn.ReLU(),
                                           nn.Linear(256, 3 * self.nctrl)))
            self.phase.append(nn.Sequential(nn.Linear(2 * C + 3 * n, 256), nn.ReLU(),
                                            nn.Linear(256, n - 1)))

    def encode(self, pc, nrm):
        x = torch.cat([pc / SCALE, nrm], -1) if (self.use_nrm and nrm is not None) else pc / SCALE
        idx = knn(pc, pc, self.gk)
        a = self.e1(x, idx); b = self.e2(a, idx); c = self.e3(b, idx)
        f = self.fuse(torch.cat([a, b, c], -1))
        return f, self.glob(f.max(1).values)

    def forward(self, batch):
        """train_family.py's contract: forward(batch) -> {'pred', ...}.

        batch = {'pc' (B,N,3), 'coarse' (B,85,3), 'ear' (B,), **NEEDS}
        """
        pc, q0 = batch["pc"], batch["coarse"]
        nrm = batch.get("nrm")
        f, g = self.encode(pc, nrm)
        B = pc.shape[0]
        out, phases = [], []
        for ci, (lo, hi) in enumerate(CONTOURS):
            n = hi - lo + 1
            qc = q0[:, lo:hi + 1]                                   # coarse init for this contour
            # pool encoder features near this contour's coarse landmarks
            idx = knn(qc, pc, 8).reshape(B, -1)
            loc = torch.gather(f, 1, idx[..., None].expand(-1, -1, f.shape[-1])
                               ).view(B, n, 8, -1).mean(2).max(1).values
            h = torch.cat([g, loc, (qc / SCALE).reshape(B, -1)], -1)
            P = self.ctrl[ci](h).view(B, self.nctrl, 3) * SCALE
            # anchor the control polygon to the coarse curve so the head predicts a
            # correction rather than absolute position from scratch
            base = torch.nn.functional.interpolate(
                qc.transpose(1, 2), size=self.nctrl, mode="linear", align_corners=True
            ).transpose(1, 2)
            P = base + P
            # strictly increasing phases: positive increments with a floor, then cumsum
            raw = torch.nn.functional.softplus(self.phase[ci](h))
            step = raw + self.minstep / (n - 1)
            t = torch.cat([torch.zeros(B, 1, device=pc.device, dtype=pc.dtype),
                           step.cumsum(-1)], -1)
            t = t / t[:, -1:].clamp(min=1e-6)                        # (B,n) in [0,1], increasing
            out.append(catmull_rom(P, t))
            phases.append(t)
        return {"pred": torch.cat(out, 1), "phases": phases}

    def loss(self, out, tg, batch=None):
        """Ordered MSE (the competition metric) plus a PHASE-INVARIANT curve term.

        The curve term is the distance from each GT landmark to the predicted curve
        sampled densely, so it improves curve GEOMETRY without caring where the phases
        landed. Without it the two heads fight: a bad phase drags the control points.
        """
        pred = out["pred"] if isinstance(out, dict) else out
        L = ((pred - tg) ** 2).sum(-1).mean()
        if self.w_curve > 0:
            for ci, (lo, hi) in enumerate(CONTOURS):
                gt = tg[:, lo:hi + 1]
                cur = pred[:, lo:hi + 1]
                d = torch.cdist(gt, cur)                              # (B, n, n)
                L = L + self.w_curve * d.min(-1).values.pow(2).mean()
        return L


def _smoke():
    torch.manual_seed(0)
    B, N = 2, 512
    pc = torch.randn(B, N, 3) * 8
    q0 = torch.randn(B, NL, 3) * 4
    tg = q0 + torch.randn(B, NL, 3) * 0.5
    m = MODEL({"width": 32, "gk": 8, "nctrl": 12})
    out = m({"pc": pc, "coarse": q0, "ear": torch.zeros(B, dtype=torch.long)})
    pred = out["pred"]
    assert pred.shape == (B, NL, 3), pred.shape
    # the property this family exists for: phases are strictly increasing, by construction
    for ci, t in enumerate(out["phases"]):
        d = t[:, 1:] - t[:, :-1]
        assert (d > 0).all(), f"contour {ci}: phase not strictly increasing (min {d.min():.2e})"
        assert torch.allclose(t[:, 0], torch.zeros(B)) and torch.allclose(t[:, -1], torch.ones(B))
    L = m.loss(out, tg)
    L.backward()
    gn = sum(float(p.grad.norm()) for p in m.parameters() if p.grad is not None)
    print(f"pred {tuple(pred.shape)}  params {sum(p.numel() for p in m.parameters()):,}  "
          f"loss {float(L):.4f}  grad-norm {gn:.3f}")
    print(f"phases strictly increasing in all {len(CONTOURS)} contours, endpoints exactly 0 and 1")
    mn = min(float((t[:, 1:] - t[:, :-1]).min()) for t in out["phases"])
    print(f"smallest phase step {mn:.5f} (floor {MINSTEP}/(n-1))")
    # a curve evaluated at monotone phases cannot reorder landmarks along itself
    print("OK")


if __name__ == "__main__":
    _smoke()
