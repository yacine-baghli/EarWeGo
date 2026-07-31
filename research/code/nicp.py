"""
Laplacian-regularized non-rigid ICP (dense surface registration).

Deforms a template mesh onto a target surface, minimizing
    || W (V - C) ||^2  +  lambda * || L (V - V0) ||^2
where C = closest target-surface points, L = uniform graph Laplacian, V0 = the
template's initial (rigidly-placed) vertices. The Laplacian term preserves the
template's local shape, so the deformation is smooth and correspondence-preserving.
lambda is annealed from stiff -> flexible.

Purpose: transport the template's landmarks (fixed barycentric points) onto a new
ear using DENSE surface information, which over-determines the along-contour
position that local detection cannot resolve.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.spatial import cKDTree

from surfproj import SurfaceProjector


def uniform_laplacian(n, edges):
    """graph Laplacian L = D - A from an edge list"""
    i = np.r_[edges[:, 0], edges[:, 1]]
    j = np.r_[edges[:, 1], edges[:, 0]]
    A = sp.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    A.data[:] = 1.0                      # dedupe weights
    deg = np.asarray(A.sum(1)).ravel()
    return sp.diags(deg) - A


def edges_from_faces(F):
    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0)


class NICP:
    def __init__(self, tmpl_verts, tmpl_faces):
        self.V0 = np.asarray(tmpl_verts, float)
        self.F = np.asarray(tmpl_faces, int)
        self.E = edges_from_faces(self.F)
        self.L = uniform_laplacian(len(self.V0), self.E).tocsr()
        self.LtL = (self.L.T @ self.L).tocsc()

    def fit(self, target_verts, target_faces, V_init=None,
            lambdas=(50.0, 20.0, 8.0, 3.0, 1.0, 0.4), iters_per=3,
            max_dist=6.0, verbose=False):
        """deform the template onto the target surface; returns deformed vertices"""
        V = (self.V0 if V_init is None else np.asarray(V_init, float)).copy()
        proj = SurfaceProjector(target_verts, target_faces, k=8)
        tree = cKDTree(target_verts)
        n = len(V)
        LtL = self.LtL
        for lam in lambdas:
            for _ in range(iters_per):
                C, dist = proj.project(V)                      # closest surface points
                w = (dist < max_dist).astype(float)            # reject far matches
                # also reject if the nearest target vertex is far (holes/boundaries)
                dv, _ = tree.query(V)
                w[dv > max_dist] = 0.0
                W = sp.diags(w)
                Aop = (W + lam * LtL).tocsc()
                rhs = W @ C + lam * (LtL @ self.V0)
                V = np.column_stack([spl.spsolve(Aop, rhs[:, d]) for d in range(3)])
            if verbose:
                C, dist = proj.project(V)
                print(f"    lam={lam:5.1f} mean surf dist {dist.mean():.3f}mm", flush=True)
        return V


def barycentric_of(points, verts, faces):
    """express points as (face_index, barycentric weights) on the mesh"""
    proj = SurfaceProjector(verts, faces, k=12)
    out_f, out_b = [], []
    tree = cKDTree(verts)
    _, nn = tree.query(points, k=12)
    for p, cand_v in zip(points, np.atleast_2d(nn)):
        cand = set()
        for vi in cand_v:
            cand.update(np.where((faces == vi).any(1))[0])
        best, bd, bb = -1, 1e18, None
        for fi in cand:
            a, b, c = verts[faces[fi]]
            q = _closest_in_tri(p, a, b, c)
            d = np.linalg.norm(q - p)
            if d < bd:
                bd, best = d, fi
                bb = _bary(q, a, b, c)
        out_f.append(best); out_b.append(bb)
    return np.array(out_f), np.array(out_b)


def _closest_in_tri(p, a, b, c):
    n = np.cross(b - a, c - a); nn = np.linalg.norm(n)
    if nn < 1e-12:
        return a
    n = n / nn
    q = p - n * np.dot(p - a, n)
    w = _bary(q, a, b, c)
    if (w >= -1e-9).all():
        return q
    # clamp to edges
    best, bd = a, 1e18
    for u, v in ((a, b), (b, c), (c, a)):
        t = np.clip(np.dot(p - u, v - u) / max(np.dot(v - u, v - u), 1e-12), 0, 1)
        cand = u + t * (v - u)
        d = np.linalg.norm(cand - p)
        if d < bd:
            bd, best = d, cand
    return best


def _bary(q, a, b, c):
    v0, v1, v2 = b - a, c - a, q - a
    d00, d01, d11 = v0 @ v0, v0 @ v1, v1 @ v1
    d20, d21 = v2 @ v0, v2 @ v1
    den = d00 * d11 - d01 * d01
    if abs(den) < 1e-18:
        return np.array([1.0, 0.0, 0.0])
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return np.array([1 - v - w, v, w])


def transport(faces, bary, V):
    """evaluate barycentric points on deformed vertices V"""
    return np.einsum("ij,ijk->ik", bary, V[faces])
