"""
RESOLUTION SWEEP: measure the numbers the sweep's neighbourhood parameters must be set
from, and measure what a forward+backward actually costs.

WHY. Every backbone in this repo has run on clouds whose GRID-EQUIVALENT SPACING is
1.09mm (8192 pts) to 2.17mm (2048 pts), while the native mesh has ~0.67mm vertex spacing
and GT landmarks sit 0.021mm from that surface. The open question is whether resolution
is the binding constraint. Answering it needs 16384/32768-point clouds AND neighbourhood
parameters rescaled so the PHYSICAL window is held -- the earlier 8192-point DGCNN test
was undecidable precisely because K/GK were left at their 2048-point values, shrinking
every window by 2x at the same time as the density grew.

THE ONE NUMBER EVERYTHING HANGS ON is the grid-equivalent spacing s = sqrt(A/N), A being
the sampled crop area. It is NOT the mean nearest-neighbour distance: for a random (not
gridded) sample of a surface E[NN] = s/2, so a V0 set from a mean-NN measurement is 2x
too small, which quadruples kpconv's auto KMAX and quarters its intended stride. Both are
printed side by side here, and `2*meanNN` is checked against sqrt(A/N).

    MODE=geom  FILES=scratch/screen_data_8192nrm.npz,... python research/code/res_sweep_prep.py
    MODE=mem   FAM=kpconv N=32768 B=2 python research/code/res_sweep_prep.py
    python research/code/res_sweep_prep.py          # <- CPU smoke test, <90s

MODE=mem reports torch.cuda.max_memory_allocated() when a GPU is present and the process
PEAK WORKING SET otherwise. The CPU number is a PROXY, not a GPU measurement: tensor
shapes are identical but CUDA's SDPA takes a flash kernel where CPU takes the math
fallback (so ptv3's CPU peak OVERSTATES the GPU when no attention mask is needed), and
the caching allocator fragments where CPU frees eagerly. Run this file with MODE=mem on
the box before launching a sweep -- that is the only measured answer.

ENVIRONMENT
  MODE      smoke   geom | mem | smoke
  FILES     scratch/screen_data_8192nrm.npz      comma-separated npz list (MODE=geom)
  EARS      8       ears sampled per file, evenly spread over the 340
  RADII     2.5,5,10,20   mm ball radii whose occupancy is reported
  VOXELS    0.425,0.55,0.6,0.85,1.09,1.5,2.125,3.0,4.25,5.31,8.5   mm, ptv3 grid ladder
  FAM       kpconv  kpconv | ptv3        (MODE=mem)
  N         8192    points per cloud     (MODE=mem)
  B         2       batch size           (MODE=mem)
  SUBFRAC   0.625   train-time subsample; 1.0 probes the evaluation forward
  NOGRAD    0       1 = no_grad (the evaluation path)
  CFG_JSON  {}      merged over the family defaults, e.g. '{"r0":2.5,"v0":0.543}'
"""
import os, sys, json, math, time
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

MODE = os.environ.get("MODE", "smoke")
NEARS = int(os.environ.get("EARS", "8"))
RADII = [float(x) for x in os.environ.get("RADII", "2.5,5,10,20").split(",")]
VOXELS = [float(x) for x in os.environ.get(
    "VOXELS", "0.425,0.55,0.6,0.85,1.09,1.5,2.125,3.0,4.25,5.31,8.5").split(",")]


# ------------------------------------------------------------------ peak memory
def peak_bytes(reset=False):
    """CUDA allocator peak when a GPU is present, else the process peak working set."""
    if torch.cuda.is_available():
        if reset:
            torch.cuda.reset_peak_memory_stats()
            return 0
        return int(torch.cuda.max_memory_allocated())
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return int(c.PeakWorkingSetSize)
    import resource
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


# ------------------------------------------------------------------ geometry
def spacing_stats(P):
    """P (n,3) one cloud -> mean nearest-neighbour distance and the bbox extent."""
    from scipy.spatial import cKDTree
    d = cKDTree(P).query(P, k=2, workers=-1)[0][:, 1]
    return float(d.mean()), P.max(0) - P.min(0)


def ball_counts(P, r):
    from scipy.spatial import cKDTree
    t = cKDTree(P)
    c = np.asarray(t.query_ball_point(P, r, return_length=True, workers=-1))
    return c


def occupied_voxels(P, v):
    g = np.floor(P / v).astype(np.int64)
    g -= g.min(0)
    D = g.max(0) + 1
    return int(len(np.unique((g[:, 0] * D[1] + g[:, 1]) * D[2] + g[:, 2])))


def kp_ladder(P, v0, r0, stages):
    """Run fam_kpconv's OWN grid_subsample ladder, then count each level's true ball."""
    from fam_kpconv import grid_subsample
    t = torch.tensor(P, dtype=torch.float32)[None]
    m = torch.ones(1, t.shape[1], dtype=torch.bool)
    out = []
    for l in range(stages + 1):
        if l:
            t, m, _, _ = grid_subsample(t, m, v0 * 2 ** l)
        p = t[0][m[0]].numpy().astype(np.float64)
        c = ball_counts(p, r0 * 2 ** l)
        out.append((l, v0 * 2 ** l if l else None, r0 * 2 ** l, len(p),
                    int(np.median(c)), float(c.mean()), int(c.max())))
    return out


def snap_jitter(C, q, k=48):
    """Surrogate for the head's output jitter across fresh surface samples.

    The offset/snap head returns a softmax-weighted mean of the k cloud points nearest a
    landmark, so how much that k-NN centroid MOVES between two independent samples of the
    same surface is the geometric part of `fresh_sample_pred_variance_mm`. TTA over M
    samples shrinks it by 1/sqrt(M); that is what choosing M costs.
    """
    from scipy.spatial import cKDTree
    ctr = []
    for s in range(C.shape[0]):
        t = cKDTree(C[s])
        idx = t.query(q, k=k, workers=-1)[1]
        ctr.append(C[s][idx].mean(1))
    ctr = np.stack(ctr)
    return float(np.linalg.norm(ctr - ctr.mean(0), axis=-1).mean())


def geom(files):
    from scipy.spatial import cKDTree
    for f in files:
        if not os.path.exists(f):
            print(f"\n### {f}  ABSENT -- skipped"); continue
        z = np.load(f)
        C, Q = z["clouds"], z["coarse"]
        E, M, N = C.shape[:3]
        ears = np.linspace(0, E - 1, min(NEARS, E)).astype(int)
        mb = os.path.getsize(f) / 1e6
        print(f"\n### {os.path.basename(f)}  ({mb:.1f} MB on disk)  E={E} M={M} N={N}")
        nn, ext, f16, jit = [], [], [], []
        bc = {r: [] for r in RADII}
        vx = {v: [] for v in VOXELS}
        for e in ears:
            P = C[e, 0].astype(np.float64)
            a, b = spacing_stats(P); nn.append(a); ext.append(b)
            d16 = np.linalg.norm(P.astype(np.float16).astype(np.float64) - P, axis=1)
            f16.append((d16.mean(), d16.max()))
            for r in RADII:
                bc[r].append(ball_counts(P, r))
            for v in VOXELS:
                vx[v].append(occupied_voxels(P, v))
            if M > 1:
                jit.append(snap_jitter(C[e].astype(np.float64), Q[e].astype(np.float64)))
        nn = float(np.mean(nn))
        area = N * (2 * nn) ** 2                      # E[NN] = s/2 for a random surface sample
        print(f"  mean-NN {nn:.4f}mm   2*meanNN {2*nn:.4f}mm   implied crop area "
              f"{area:.0f} mm^2   bbox extent {np.mean(ext, 0).round(1)}")
        print(f"  GRID-EQUIVALENT SPACING to use as kpconv V0: {2*nn:.4f} mm "
              f"(sqrt(A/N) with A=9674 mm^2 -> {math.sqrt(9674/N):.4f} mm)")
        print(f"  fp16 round-trip displacement: mean {np.mean([a for a, _ in f16])*1000:.1f} um"
              f"   max {np.max([b for _, b in f16])*1000:.1f} um")
        if jit:
            print(f"  48-NN snap-target jitter across the {M} fresh samples: "
                  f"{np.mean(jit):.4f} mm  (TTA over k samples -> /sqrt(k))")
        print(f"  {'ball r_mm':>10s} {'min':>5s} {'med':>6s} {'mean':>7s} {'p99.9':>6s} {'max':>6s}"
              f"   <- KMAX must exceed max")
        for r in RADII:
            c = np.concatenate(bc[r])
            print(f"  {r:10.2f} {c.min():5d} {int(np.median(c)):6d} {c.mean():7.1f} "
                  f"{int(np.percentile(c, 99.9)):6d} {c.max():6d}")
        print(f"  {'voxel_mm':>10s} {'occupied cells (mean over ears)':>34s}   "
              f"{'saturation vs A/v^2':>20s}")
        for v in VOXELS:
            o = np.mean(vx[v])
            print(f"  {v:10.3f} {o:34.0f}   {o / (9674 / v ** 2):20.2f}")
        del z, C


# ------------------------------------------------------------------ memory probe
def synth_cloud(B, N, seed=0):
    """A pinna-scale folded sheet: 62.8 x 35.9 x 87.5mm bbox and ~9674 mm^2 of surface,
    matching the measured canonical crops, so voxel occupancy and ball sizes are realistic."""
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(B, N, generator=g); th = torch.rand(B, N, generator=g) * 6.283
    pc = torch.stack([27 * th.cos() * (1 + 0.1 * (3 * t).sin()),
                      17.9 * th.sin(), 87.5 * t - 43.7], -1)
    return pc + torch.randn(B, N, 3, generator=g) * 0.15


def mem(fam, N, B, sub, nograd, cfg):
    import importlib
    nsub = max(8, int(round(N * sub)))
    pc = synth_cloud(B, N)[:, :nsub]
    ft = torch.nn.functional.normalize(pc, dim=-1)
    q0 = pc[:, torch.linspace(0, nsub - 1, 85).long()] + torch.randn(B, 85, 3) * 0.8
    if fam == "kpconv":
        m = importlib.import_module("fam_kpconv")
        net = m.Net(cin=6, cfg={**cfg, "use_nrm": 1, "nb_stats": 1})
        c = net.c
        head = (f"R0={c['r0']} V0={c['v0']} STAGES={c['stages']} KMAX={c['kmax']} "
                f"WIDTH={c['width']} radii {[round(x,2) for x in c['rad']]}mm")
        run = lambda: net(pc, q0, ft)[1]
    else:
        m = importlib.import_module("fam_ptv3")
        net = m.Net(cin=6, cfg={**cfg, "use_nrm": 1})
        c = net.c
        head = (f"VOXEL={c.voxel} VOXGROW={c.voxgrow} POOLR={c.poolr} PATCH={c.patch} "
                f"STAGES={c.stages} slots {m.counts(nsub, c)} "
                f"voxels {[round(c.voxel * c.voxgrow ** s, 2) for s in range(c.stages)]}mm")
        run = lambda: net([pc], q0, [ft])[1]
    npar = sum(p.numel() for p in net.parameters())
    base = peak_bytes(reset=True) or peak_bytes()
    t0 = time.time()
    if nograd:
        net.eval()
        with torch.no_grad():
            out = run()
    else:
        out = run()
        (out ** 2).sum().backward()
    dt = time.time() - t0
    pk = peak_bytes()
    dev = "cuda" if torch.cuda.is_available() else "cpu(proxy)"
    print(f"{fam:7s} N={N:6d} nsub={nsub:6d} B={B:2d} {'eval' if nograd else 'train':5s}  "
          f"{head}")
    print(f"        params {npar:,}  pred {tuple(out.shape)}  peak[{dev}] "
          f"{(pk - base if torch.cuda.is_available() else pk)/2**30:.2f} GiB  {dt:.1f}s")
    if fam == "ptv3" and net.enc.pool_stats:
        print(f"        grid pooling: " + "  ".join(f"{o:.0f} occupied -> {s} slots"
                                                    for o, s in net.enc.pool_stats)
              + "   (occupied < slots => padded slots => attention MASK => "
                "SDPA math fallback, not flash)")
    return out, npar


# ------------------------------------------------------------------ smoke test
def smoke():
    t0 = time.time()
    print("=" * 78)
    print("SMOKE 1/3 -- geometry measurement on a synthetic bundle")
    tmp = os.environ.get("SMOKE_DIR", os.path.join(
        __import__("tempfile").gettempdir(), "res_sweep_smoke"))
    os.makedirs(tmp, exist_ok=True)
    pc = synth_cloud(3 * 2, 1024).reshape(3, 2, 1024, 3).numpy()
    q = pc[:, 0, ::12][:, :85]
    fp = f"{tmp}/tiny.npz"
    np.savez(fp, clouds=pc.astype(np.float32), coarse=q.astype(np.float32))
    g = globals()
    g["RADII"], g["VOXELS"], g["NEARS"] = [5.0, 10.0], [2.0, 5.0], 3
    geom([fp, f"{tmp}/does_not_exist.npz"])

    print("\nSMOKE 2/3 -- kpconv forward+backward, B=2")
    o1, p1 = mem("kpconv", 1024, 2, 1.0, 0,
                 dict(npts=1024, width=16, fdim=64, stages=2, r0=3.0, v0=1.5, npass=2))
    assert tuple(o1.shape) == (2, 85, 3), o1.shape

    print("\nSMOKE 3/3 -- ptv3 forward+backward, B=2")
    o2, p2 = mem("ptv3", 1024, 2, 1.0, 0,
                 dict(npts=1024, width=32, stages=2, depth=1, patch=256, headc=64,
                      voxel=1.09, voxgrow=2.5, poolr=4))
    assert tuple(o2.shape) == (2, 85, 3), o2.shape
    assert p1 > 0 and p2 > 0 and torch.isfinite(o1).all() and torch.isfinite(o2).all()
    print(f"\nSMOKE PASS  ({time.time()-t0:.1f}s)")
    print("=" * 78)


if __name__ == "__main__":
    if MODE == "geom":
        geom(os.environ.get("FILES", "scratch/screen_data_8192nrm.npz").split(","))
    elif MODE == "mem":
        mem(os.environ.get("FAM", "kpconv"), int(os.environ.get("N", "8192")),
            int(os.environ.get("B", "2")), float(os.environ.get("SUBFRAC", "0.625")),
            int(os.environ.get("NOGRAD", "0")),
            json.loads(os.environ.get("CFG_JSON", "{}")))
    else:
        smoke()
