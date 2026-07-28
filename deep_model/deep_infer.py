"""
Dependency-free NUMPY forward pass for the DGCNN offset+snap landmark net
(scratch/gpu_train.py). No torch needed at inference -> shippable in the submission.

Replicates exactly:
  backbone: static coord kNN graph (GK) -> 3 EdgeConv layers -> fuse -> +global -> mix
  head (T passes): OFFSET (pooled window ctx + embO -> MLP -> relocate) then
                   SNAP (soft-argmax over K nearest surface pts to relocated query)
Then ssm_project (Procrustes to mean shape, project to 30-dim subspace, invert).

predict(cloud, coarse) works in ANY frame (world or canonical): the graph/windows
are relative, and ssm_project re-aligns. Ensembling/TTA handled by the caller.
"""
import numpy as np

GK = 20; K = 48


def _relu(x):
    return np.maximum(x, 0.0)


def _lin(x, W, b):
    return x @ W.T + b


def _knn(q, pc, k):
    # q (L,3), pc (P,3) -> idx (L,k) nearest pc rows to each q row
    d = ((q[:, None, :] - pc[None, :, :]) ** 2).sum(-1)
    return np.argpartition(d, k, axis=1)[:, :k]


CONTOURS = [(0, 24), (25, 54), (55, 74), (75, 84)]


def _conv1d(x, W, b, pad):
    # x (Cin,L), W (Cout,Cin,k), b (Cout,) -> (Cout,L) with zero padding
    Cin, L = x.shape; Cout, _, k = W.shape
    xp = np.zeros((Cin, L + 2 * pad)); xp[:, pad:pad + L] = x
    out = np.empty((Cout, L))
    for i in range(L):
        out[:, i] = np.tensordot(W, xp[:, i:i + k], axes=([1, 2], [0, 1])) + b
    return out


class DeepNet:
    def __init__(self, weights_path, scale=30.0, T=4):
        z = np.load(weights_path)
        self.w = {k: z[k].astype(np.float64) for k in z.files
                  if k not in ("best", "K", "T", "SCALE", "seed")}
        self.SCALE = float(z["SCALE"]) if "SCALE" in z.files else scale
        self.T = int(z["T"]) if "T" in z.files else T
        self.embW = self.w["emb.weight"]       # (85,32)
        self.embOW = self.w["embO.weight"]     # (85,32)
        self.NL = self.embW.shape[0]
        self.has_contour = "embC.weight" in self.w

    def _seq(self, x, prefix, idxs):
        # EdgeConv MLPs are Linear->ReLU (->Linear->ReLU), i.e. ReLU after EVERY linear
        for li in idxs:
            x = _relu(_lin(x, self.w[f"{prefix}.{li}.weight"], self.w[f"{prefix}.{li}.bias"]))
        return x

    def _edgeconv(self, gidx, feat, prefix, idxs):
        P = feat.shape[0]
        fj = feat[gidx]                                   # (P,GK,Cin)
        fi = np.repeat(feat[:, None, :], gidx.shape[1], axis=1)
        e = np.concatenate([fi, fj - fi], -1)             # (P,GK,2Cin)
        e = self._seq(e, prefix, idxs)                    # (P,GK,Cout)
        return e.max(1)                                   # (P,Cout)

    def backbone(self, pc):
        x = pc / self.SCALE
        gidx = _knn(x, x, GK)
        h1 = self._edgeconv(gidx, x, "ec1", [0, 2])       # (P,64)
        h2 = self._edgeconv(gidx, h1, "ec2", [0])         # (P,128)
        h3 = self._edgeconv(gidx, h2, "ec3", [0])         # (P,128)
        h = _relu(_lin(np.concatenate([h1, h2, h3], -1), self.w["fuse.0.weight"], self.w["fuse.0.bias"]))
        g = np.repeat(h.max(0, keepdims=True), h.shape[0], axis=0)
        h = _relu(_lin(np.concatenate([h, g], -1), self.w["mix.0.weight"], self.w["mix.0.bias"]))
        return h                                          # (P,C)

    def _step(self, pc, h, q):
        # OFFSET: window around q, pooled ctx + embO -> displacement
        idx = _knn(q, pc, K)                              # (L,K)
        featK = h[idx]                                    # (L,K,C)
        ctx = np.concatenate([featK.mean(1), featK.max(1)], -1)   # (L,2C)
        o = np.concatenate([ctx, self.embOW], -1)
        o = _relu(_lin(o, self.w["offset.0.weight"], self.w["offset.0.bias"]))
        o = _relu(_lin(o, self.w["offset.3.weight"], self.w["offset.3.bias"]))
        off = _lin(o, self.w["offset.5.weight"], self.w["offset.5.bias"])
        q1 = q + off                                      # (L,3)
        # SNAP: re-window at q1, soft-argmax over K neighbours
        idx2 = _knn(q1, pc, K)
        featK2 = h[idx2]; posK2 = pc[idx2]                # (L,K,C),(L,K,3)
        rel = (posK2 - q1[:, None, :]) / self.SCALE
        e = np.repeat(self.embW[:, None, :], K, axis=1)   # (L,K,32)
        a = np.concatenate([featK2, rel, e], -1)
        a = _relu(_lin(a, self.w["attn.0.weight"], self.w["attn.0.bias"]))
        a = _relu(_lin(a, self.w["attn.2.weight"], self.w["attn.2.bias"]))
        logit = _lin(a, self.w["attn.4.weight"], self.w["attn.4.bias"])[..., 0]   # (L,K)
        logit -= logit.max(1, keepdims=True)
        wgt = np.exp(logit); wgt /= wgt.sum(1, keepdims=True)
        q2 = (wgt[..., None] * posK2).sum(1)              # (L,3)
        return q2

    def _contour_refine(self, pc, h, q):
        idx = _knn(q, pc, 1)[:, 0]                        # (85,) nearest cloud pt
        f = _relu(_lin(h[idx], self.w["lmfeat.0.weight"], self.w["lmfeat.0.bias"]))  # (85,64)
        inp = np.concatenate([q / self.SCALE, f, self.w["embC.weight"]], -1)         # (85,99)
        out = np.zeros((self.NL, 3))
        for c, (lo, hi) in enumerate(CONTOURS):
            x = inp[lo:hi + 1].T                          # (99,L)
            p = f"contour_nets.{c}"
            x = _relu(_conv1d(x, self.w[f"{p}.0.weight"], self.w[f"{p}.0.bias"], 2))
            x = _relu(_conv1d(x, self.w[f"{p}.2.weight"], self.w[f"{p}.2.bias"], 1))
            x = _conv1d(x, self.w[f"{p}.4.weight"], self.w[f"{p}.4.bias"], 0)         # (3,L)
            out[lo:hi + 1] = x.T
        return q + out

    def predict(self, cloud, coarse):
        h = self.backbone(cloud)
        q = coarse.astype(np.float64)
        for _ in range(self.T):
            q = self._step(cloud, h, q)
        if self.has_contour:
            q = self._contour_refine(cloud, h, q)
        return q                                          # (85,3) same frame as input
