"""ORACLE HONESTY AUDIT: is the affine-oracle gap real shared structure, or is the
oracle partly fitting noise? Fit (a,b) on HALF the landmarks of the contour, then score
the OTHER half. A gain that transfers to held-out landmarks is real phase structure."""
import numpy as np, sys, os
sys.path.insert(0, 'scratch'); import oracles_v2 as O
z = np.load('scratch/oof_final.npz'); P = z['pred'].astype(float); G = z['gt'].astype(float)
CONT = [(0,24,'outer helix'),(25,54,'concha'),(55,74,'inner helix'),(75,84,'sup. antihelix')]
NE = int(os.environ.get('NE', '340'))
A = np.arange(0.82, 1.1801, 0.004); B = np.arange(-O.PAD, O.PAD+1e-9, 0.04)
print(f"{'contour':16s} {'base(held)':>10s} {'full-fit':>9s} {'half-fit':>9s} {'honest':>8s} {'/full':>7s}")
for lo, hi, nm in CONT:
    n = hi-lo+1; bh=[]; ff=[]; hf=[]
    for i in range(NE):
        Pc, Gc = P[i,lo:hi+1], G[i,lo:hi+1]
        s = O.arc(Pc); s0 = s[0]
        Q = (s0 + A[:,None]*(s-s0)[None,:])[:,None,:] + B[None,:,None]
        E = np.linalg.norm(O.eval_poly(Pc, Q) - Gc[None,None,:,:], axis=-1)   # (nA,nB,n)
        Eall = E.mean(-1); ja, jb = np.unravel_index(int(np.argmin(Eall)), Eall.shape)
        for par in (0,1):
            fit = np.arange(n)%2==par; hold = ~fit
            bh.append(np.linalg.norm(Pc[hold]-Gc[hold],axis=1).mean())
            Ef = E[:,:,fit].mean(-1); ia, ib = np.unravel_index(int(np.argmin(Ef)), Ef.shape)
            hf.append(E[ia,ib,hold].mean())
            ff.append(E[ja,jb,hold].mean())
    bh, ff, hf = np.mean(bh), np.mean(ff), np.mean(hf)
    print(f"{nm:16s} {bh:10.4f} {ff:9.4f} {hf:9.4f} {bh-hf:8.4f} {100*(bh-hf)/max(bh-ff,1e-9):6.0f}%",
          flush=True)
