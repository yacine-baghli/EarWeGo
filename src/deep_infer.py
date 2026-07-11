"""
Dependency-free (numpy-only) inference for the heatmap/local-soft-argmax deep
landmark net trained on Colab (scripts/colab_train_heatmap.py).

Loads `deep_weights.npz` (exported torch state_dict as numpy arrays) and runs
the exact same forward pass in numpy — so the submission needs NO torch.
Mirrors Net.forward in colab_train_heatmap.py; keep the two in sync.
"""
import numpy as np

SCALE = 30.0
K = 48
NL = 85


def _relu(x):
    return np.maximum(x, 0.0)


def _lin(x, w, b):
    return x @ w.T + b


class DeepLandmarkNet:
    """Numpy forward of the trained local-soft-argmax net."""

    def __init__(self, weights):
        # weights: dict-like (np.load(...)) with torch state_dict keys
        self.w = {k: np.asarray(weights[k], dtype=np.float64) for k in weights.files} \
            if hasattr(weights, "files") else {k: np.asarray(v, dtype=np.float64) for k, v in weights.items()}

    @classmethod
    def load(cls, path):
        return cls(np.load(path))

    def _enc(self, x):                                   # x (P,3)
        w = self.w
        x = _relu(_lin(x, w["enc.0.weight"], w["enc.0.bias"]))
        x = _relu(_lin(x, w["enc.2.weight"], w["enc.2.bias"]))
        return _relu(_lin(x, w["enc.4.weight"], w["enc.4.bias"]))   # (P,256)

    def predict(self, cloud, coarse):
        """cloud (P,3), coarse (85,3) in canonical mm frame -> pred (85,3)."""
        w = self.w
        P = cloud.shape[0]
        h = self._enc(cloud / SCALE)                                # (P,256)
        g = h.max(0, keepdims=True)                                 # (1,256)
        h = _relu(_lin(np.concatenate([h, np.repeat(g, P, 0)], 1),
                       w["pointfeat.0.weight"], w["pointfeat.0.bias"]))  # (P,256)
        # nearest-K cloud points to each coarse landmark
        d = np.linalg.norm(coarse[:, None, :] - cloud[None, :, :], axis=2)  # (85,P)
        nn = np.argsort(d, axis=1)[:, :K]                           # (85,K)
        featK = h[nn]                                               # (85,K,256)
        posK = cloud[nn]                                            # (85,K,3)
        rel = (posK - coarse[:, None, :]) / SCALE                   # (85,K,3)
        emb = w["emb.weight"][np.arange(NL)][:, None, :].repeat(K, 1)  # (85,K,32)
        x = np.concatenate([featK, rel, emb], -1)                   # (85,K,291)
        x = _relu(_lin(x, w["attn.0.weight"], w["attn.0.bias"]))
        x = _relu(_lin(x, w["attn.2.weight"], w["attn.2.bias"]))
        logit = _lin(x, w["attn.4.weight"], w["attn.4.bias"])[..., 0]  # (85,K)
        logit = logit - logit.max(1, keepdims=True)
        a = np.exp(logit); a /= a.sum(1, keepdims=True)             # softmax over K
        return (a[..., None] * posK).sum(1)                        # (85,3)
