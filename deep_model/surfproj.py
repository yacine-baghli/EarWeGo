"""Exact point-to-triangle projection, pure numpy (no rtree/trimesh dependency).

Ericson's barycentric region algorithm, vectorized over candidate triangles.
Used to snap predicted landmarks onto the mesh SURFACE along the normal (minimal
tangential motion), unlike a nearest-VERTEX snap which drags points sideways.
"""
import numpy as np
from scipy.spatial import cKDTree


def closest_on_triangles(p, A, B, C):
    """closest point to p (3,) on each triangle (A,B,C each (M,3)) -> (M,3)"""
    ab, ac, ap = B - A, C - A, p - A
    d1 = np.einsum("ij,ij->i", ab, ap); d2 = np.einsum("ij,ij->i", ac, ap)
    bp = p - B
    d3 = np.einsum("ij,ij->i", ab, bp); d4 = np.einsum("ij,ij->i", ac, bp)
    cp = p - C
    d5 = np.einsum("ij,ij->i", ab, cp); d6 = np.einsum("ij,ij->i", ac, cp)

    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = va + vb + vc

    def sdiv(n, d):
        return np.divide(n, d, out=np.zeros_like(n, dtype=float), where=np.abs(d) > 1e-20)

    v = sdiv(vb, denom); w = sdiv(vc, denom)
    res = A + v[:, None] * ab + w[:, None] * ac                      # interior (region 0)
    # edge regions
    m_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    res = np.where(m_bc[:, None], B + sdiv(d4 - d3, (d4 - d3) + (d5 - d6))[:, None] * (C - B), res)
    m_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    res = np.where(m_ac[:, None], A + sdiv(d2, d2 - d6)[:, None] * ac, res)
    m_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    res = np.where(m_ab[:, None], A + sdiv(d1, d1 - d3)[:, None] * ab, res)
    # vertex regions (highest priority)
    res = np.where(((d6 >= 0) & (d5 <= d6))[:, None], C, res)
    res = np.where(((d3 >= 0) & (d4 <= d3))[:, None], B, res)
    res = np.where(((d1 <= 0) & (d2 <= 0))[:, None], A, res)
    return res


class SurfaceProjector:
    """project points onto a mesh surface using local candidate triangles"""

    def __init__(self, verts, faces, k=12):
        self.V = np.asarray(verts, float); self.F = np.asarray(faces, int)
        self.tree = cKDTree(self.V); self.k = k
        # vertex -> incident faces
        self.v2f = [[] for _ in range(len(self.V))]
        for fi, f in enumerate(self.F):
            for vi in f:
                self.v2f[vi].append(fi)

    def project(self, pts):
        pts = np.asarray(pts, float)
        out = np.empty_like(pts); dist = np.empty(len(pts))
        _, nn = self.tree.query(pts, k=self.k)
        for i, p in enumerate(pts):
            cand = set()
            for vi in np.atleast_1d(nn[i]):
                cand.update(self.v2f[vi])
            fi = np.fromiter(cand, int)
            T = self.F[fi]
            q = closest_on_triangles(p, self.V[T[:, 0]], self.V[T[:, 1]], self.V[T[:, 2]])
            d = np.linalg.norm(q - p, axis=1)
            j = np.argmin(d)
            out[i] = q[j]; dist[i] = d[j]
        return out, dist
