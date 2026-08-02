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

AND IT NEEDS MORE THAN 8 EARS. A is a property of the EAR, ranging 8000..10418 mm^2
(build_hires_data.py's own crop-area print, 60 ears), so s has a ~4% per-ear sd and 8
ears leave a 1.4% standard error. The first pass of this file measured over EARS=8 and
got 1.0965 / 0.7739 / 0.5468; 24 and 64 ears both give 1.067 / 0.754 / 0.533, and the
triangulated crop area (median 9416 mm^2) independently gives 1.0715 at 8192. MODE=geom
now defaults to 64 ears and prints the sem. The 8-ear error was +2.68 / +2.64 / +2.66%,
i.e. the SAME at every arm, so it biased no comparison -- but every absolute millimetre
below that was derived from it (head window, implied crop area) is ~2.7% high, and the
head window is really 4.17mm, not 4.29mm.

    MODE=geom  FILES=scratch/screen_data_8192nrm.npz,... python research/code/res_sweep_prep.py
    MODE=pool  FILES=...8192nrm.npz,...32768nrm.npz PATCHES=256,1024 python ...
    MODE=mem   FAM=kpconv N=32768 B=2 python research/code/res_sweep_prep.py
    python research/code/res_sweep_prep.py          # <- CPU smoke test, <90s

WHAT IT MEASURED (8 ears per file, sample 0, area-weighted surface clouds from
build_hires_data.py; reproduce with EARS=8 MODE=geom / MODE=pool / MODE=mem. The
spacing and head-window columns are the 8-ear values and are ~2.7% high at every arm --
see the paragraph above; the ball counts and the snap-jitter are not affected by that):

    N       spacing   n(2.5)  n(5)   n(10)   n(20)   head window   snap-jitter  file
    8192    1.0965     17.6    75.2   368.2  1676.1   4.29mm k=48    0.389mm     248MB
    16384   0.7739     34.1   149.1   734.9  3355.3   4.28mm k=96    0.274mm     244MB
    32768   0.5468     67.2   297.3  1466.8  6711.0   4.27mm k=192   0.195mm     487MB
    64-ear spacing: 1.0674 / 0.7541 / 0.5328 mm  ->  head window 4.17mm, V0 as shipped

INDEX COUNTS SCALE AS N, NOT AS sqrt(N). The fitted exponent of n(r) against N over
8192->32768 is 0.99-1.00 for r >= 5mm and 0.97 at 2.5mm. This is the whole reason the
earlier 8192 test was undecidable: doubling K for a 4x point increase (which is what
"index counts scale as N^(1/2)" prescribes, and what run_famA_probe.sh's header says)
still halves the physical window. Holding the 4.29mm head window from 8192 to 32768 needs
k = 48 -> 192, and the measured window then lands at 4.29 / 4.28 / 4.27 mm.

THE AUTO BALL CAP IS NOT SAFE, AND ITS FAILURE GROWS WITH N -- the one thing in this
whole exercise that would have quietly biased the sweep AGAINST resolution. fam_kpconv
derives KMAX = ceil(2.8*pi*(R0/V0)^2) = 46 / 92 / 184 from the level-0 occupancy, on the
argument that r_l/v_l is constant so all levels match. Run its OWN ladder on 40 real ears
(MODE=ladder) and the deep levels are 2-3x denser than that design:

    N       worst ball by level (train density)      worst overall   auto KMAX
    8192    L0 35  L1 52  L2 54  L3 51                    54            46   TRUNCATES
    16384   L0 64  L1 111 L2 127 L3 110                  127            92   TRUNCATES
    32768   L0 106 L1 217 L2 252 L3 236                  252           184   TRUNCATES

So the SHIPPED 8192 config truncates 2.6% of its 10mm balls and 8.0% of its 20mm balls,
and at 16384 that becomes 9.3% / 12.1%. Set CFG_KMAX = 96 / 192 / 384 (still exactly ~N,
anchored on the measured worst ball x 1.235 for a -10% aug_scale x 1.05 for unsampled
ears) and the audit prints frac_truncated 0.000 at every level of every arm -- verified
on real ears at all three. It costs ~2.1x kpconv memory and step time, paid equally by
all three arms.

FP16 STORAGE IS FREE, MEASURED. Round-tripping the real fp32 8192 clouds through fp16
displaces a point by 5.1um mean / 21.9um max -- 0.5% of the 1.09mm spacing and 1.9% of
the 0.021mm the GT landmarks already sit off the surface. train_family.load_data does
torch.tensor(...).float(), so the network sees float32 either way; only the file halves.
GROUND TRUTH IS NOT QUANTISED -- build_hires_data.py copies `true`/`coarse` through at
float32 and only `clouds`/`nrm` become fp16.

RESOLUTION QUIETENS THE SNAP TARGET FOR FREE. The head returns the softmax-weighted mean
of its k nearest cloud points; the per-sample sd of that centroid across the 4 fresh
surface samples, measured at the k that holds the SAME 4.29mm window, is 0.389 / 0.274 /
0.194 mm at 8192 / 16384 / 32768 -- a clean 1/sqrt(N). TTA over 4 samples divides it by
2 again, so the sampling-noise floor of the output goes 0.194 -> 0.097mm. That is a
mechanism by which resolution can help that has nothing to do with the backbone, and it
is bounded: against a ~1.25mm error it is worth at most sqrt(1.25^2+0.194^2) -
sqrt(1.25^2+0.097^2) = 0.011mm. Resolution has to buy its gain somewhere else.

MODE=mem reports torch.cuda.max_memory_allocated() when a GPU is present and the process
PEAK WORKING SET otherwise. The CPU number is a PROXY, not a GPU measurement: tensor
shapes are identical but CUDA's SDPA takes a flash kernel where CPU takes the math
fallback (so ptv3's CPU peak OVERSTATES the GPU when no attention mask is needed), and
the caching allocator fragments where CPU frees eagerly. Run this file with MODE=mem on
the box before launching a sweep -- that is the only measured answer.

CPU-PROXY peak of ONE TRAINING forward+backward at B=1, sub_frac=0.625, with the FINAL
rescaled configs (CFG_KMAX 96/192/384), and the CPU time of the same call at B=2:

    N        kpconv GiB  fwd+bwd  x8192      ptv3 GiB  fwd+bwd  x8192
    8192       0.45       4.8 s    1.0        0.32      4.0 s    1.0
    16384      1.30      17.7 s    3.7        0.41      4.9 s    1.2
    32768      3.4-4.6   79.9 s   16.6        0.78      8.4 s    2.1

KPConv is QUADRATIC in N: it holds N points each with pi*r^2*N/A neighbours, and every
KPConv saves a (B, n, KMAX, cm) gather AND a (B, n, KMAX, NKP) kernel correlation for
backward. At B=16 that is ~7 / ~19 / ~60 GiB, so 32768 does NOT fit on a 48 GB A6000
alongside 1.1 GiB of resident cloud data; ACCUM=4 (micro-batch 4) brings it to ~18 GiB
and is gradient- AND randomness-EXACT (smoke test 4/4). ptv3 grows LINEARLY -- its
stage-0 attention stays on the memory-efficient SDPA path and its pooling grid is pinned
in millimetres -- and fits B=16 everywhere. The 32768 kpconv figure is the shakiest here:
three repeated probes gave 3.40 / 4.60 / 5.16 GiB, so the CPU RSS proxy carries ~+-30%
spread at that size. MODE=mem on the box replaces it with the CUDA allocator peak.

ENVIRONMENT
  MODE      smoke   geom | pool | ladder | mem | cfgcheck | smoke
  EARS      8       ears sampled per file. MODE=geom defaults to 64 and MODE=ladder to 40
                    when EARS is unset -- spacing needs the precision and the worst ball
                    is an extreme-value statistic (8192 eval worst: 42 at 8 ears, 59 at 12)
  PATCHES   256     comma-separated ptv3 patch per FILES entry (MODE=pool)
  SNAPK     48      k for the snap-jitter surrogate; pass each arm its own rescaled k
  FILES     scratch/screen_data_8192nrm.npz      comma-separated npz list (MODE=geom)
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
# The head gathers its k nearest cloud points, so k means a PHYSICAL window s*sqrt(k/pi).
# Comparing the snap jitter at a fixed k across resolutions compares different windows;
# pass each arm its own rescaled k (48 / 96 / 192) to compare the same one.
SNAPK = int(os.environ.get("SNAPK", "48"))


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
        # psapi.dll forwards GetProcessMemoryInfo to kernel32!K32GetProcessMemoryInfo on
        # current Windows and the forwarded name does not always resolve through
        # ctypes.windll; try both and REFUSE to report a silent zero.
        ok = 0
        for dll, nm in ((ctypes.windll.kernel32, "K32GetProcessMemoryInfo"),
                        (ctypes.windll.psapi, "GetProcessMemoryInfo")):
            fn = getattr(dll, nm, None)
            if fn is None:
                continue
            fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            fn.restype = wintypes.BOOL
            ok = fn(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
            if ok:
                break
        assert ok and c.PeakWorkingSetSize > 0, \
            "GetProcessMemoryInfo failed -- the CPU memory proxy would report 0, which " \
            "reads as 'it fits' and is worse than no number at all"
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


def ladder(files, v0s, r0=2.5, stages=3, subfrac=0.625, nears=None):
    """WHAT KMAX ACTUALLY HAS TO BE, measured on real ears at both densities.

    fam_kpconv derives KMAX = ceil(2.8*pi*(R0/V0)^2) on the argument that r_l/v_l is
    constant so every level has the same ball occupancy. On a REAL folded pinna that is
    not quite true: grid_subsample at a fine voxel keeps relatively more of the surface
    than the doubling predicts, so the deep levels are denser than designed and their
    balls run 2-3x the level-0 size. The shipped 8192 config already truncates ~8% of its
    20mm balls because of it. That is survivable as a constant, but it GROWS with N, and
    a sweep whose deepest receptive field is silently clipped more at 32768 than at 8192
    is not a clean resolution experiment. So: measure the worst ball per level, per arm,
    at the training AND the evaluation density, and set CFG_KMAX from that.
    """
    nears = NEARS if nears is None else nears     # EARS is the knob; the table in the
    for f, v0 in zip(files, v0s):                 # docstring came from EARS=40/40/20
        if not os.path.exists(f):
            print(f"\n### {f}  ABSENT -- skipped"); continue
        z = np.load(f)
        C = z["clouds"]
        E, N = C.shape[0], C.shape[2]
        auto = max(16, int(math.ceil(2.8 * math.pi * (r0 / v0) ** 2)))
        print(f"\n### {os.path.basename(f)}  N={N} V0={v0} R0={r0}  auto KMAX={auto}"
              f"  ({min(nears, E)} ears)")
        rs = np.random.RandomState(0)
        for tag, n in (("train", int(round(N * subfrac))), ("eval", N)):
            worst = [0] * (stages + 1)
            npts = [0] * (stages + 1)
            for e in np.linspace(0, E - 1, min(nears, E)).astype(int):
                P = C[e, 0].astype(np.float64)
                P = P if n == N else P[rs.choice(N, n, replace=False)]
                for l, _, r, m, med, mean, mx in kp_ladder(P, v0, r0, stages):
                    worst[l] = max(worst[l], mx); npts[l] = max(npts[l], m)
            need = max(worst)
            print(f"  {tag:5s} n={n:6d}  worst ball per level "
                  + " ".join(f"L{l}:{w}(of {npts[l]}pts)" for l, w in enumerate(worst))
                  + f"   -> KMAX must be >= {need}"
                  + ("  OK" if auto >= need else f"  AUTO {auto} TRUNCATES"))
        del z, C


def snap_jitter(C, q, k=48):
    """Surrogate for the head's output jitter across fresh surface samples.

    The offset/snap head returns a softmax-weighted mean of the k cloud points nearest a
    landmark, so how much that k-NN centroid MOVES between two independent samples of the
    same surface is the geometric part of `fresh_sample_pred_variance_mm`. TTA over n
    samples shrinks it by 1/sqrt(n); that is what choosing M costs.

    Returned as the PER-SAMPLE sd, i.e. the mean deviation from the M-sample mean divided
    by sqrt((M-1)/M) -- the raw deviation is biased low because the mean it is measured
    against contains the sample itself.
    """
    from scipy.spatial import cKDTree
    ctr = []
    for s in range(C.shape[0]):
        t = cKDTree(C[s])
        idx = t.query(q, k=k, workers=-1)[1]
        ctr.append(C[s][idx].mean(1))
    ctr = np.stack(ctr)
    M = len(ctr)
    return float(np.linalg.norm(ctr - ctr.mean(0), axis=-1).mean()) / math.sqrt((M - 1) / M)


def geom(files, nears=None):
    from scipy.spatial import cKDTree
    global NEARS
    NEARS = NEARS if nears is None else nears
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
                jit.append(snap_jitter(C[e].astype(np.float64), Q[e].astype(np.float64),
                                       SNAPK))
        # The crop area is an EAR property and it varies 8000..10418 mm^2, so the spacing
        # has a per-ear sd of ~4%. At EARS=8 the standard error is still 1.4% and the
        # linspace(0,339,8) subset happens to land 2.7% high: it gives 1.0965mm where 24
        # and 64 ears both give 1.067mm, and build_hires_data's own triangulated crop area
        # (median 9416 mm^2 over 60 ears) says 1.0715mm. That 2.7% is IDENTICAL at all
        # three arms (+2.68/+2.64/+2.66%), so it never threatened the sweep -- the arms
        # still hold the same window as each other to 0.04% -- but it is 2.7% off as an
        # absolute millimetre. Report the spread so nobody quotes 4 decimals off 8 ears.
        sd = float(np.std(nn)) * 2
        nn = float(np.mean(nn))
        area = N * (2 * nn) ** 2                      # E[NN] = s/2 for a random surface sample
        print(f"  mean-NN {nn:.4f}mm   2*meanNN {2*nn:.4f}mm   implied crop area "
              f"{area:.0f} mm^2   bbox extent {np.mean(ext, 0).round(1)}")
        print(f"  GRID-EQUIVALENT SPACING to use as kpconv V0: {2*nn:.4f} mm "
              f"+- {sd/math.sqrt(len(ears)):.4f} sem over {len(ears)} ears "
              f"(per-ear sd {sd:.4f}; sqrt(A/N) with the measured A=9416 mm^2 -> "
              f"{math.sqrt(9416/N):.4f} mm)")
        print(f"  fp16 round-trip displacement: mean {np.mean([a for a, _ in f16])*1000:.1f} um"
              f"   max {np.max([b for _, b in f16])*1000:.1f} um")
        if jit:
            j = float(np.mean(jit))
            print(f"  {SNAPK}-NN snap-target per-sample sd across the {M} fresh samples: "
                  f"{j:.4f} mm  (window {2*nn*math.sqrt(SNAPK/math.pi):.2f}mm)"
                  f"  -> TTA/{M}: {j/math.sqrt(M):.4f}  TTA/2: {j/math.sqrt(2):.4f}"
                  f"  TTA/1: {j:.4f}")
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


# ------------------------------------------------------------------ ptv3 pool budget
def pool_budget(files, patches, cands=(2, 3, 4, 6, 8, 12, 16), subfrac=0.625, nears=None):
    """Pick ptv3's POOLR from MEASURED occupancy instead of from a guess.

    ptv3 coarsens by a GRID (voxel * voxgrow^s) but writes the result into a fixed
    tensor budget ns[s] from counts(). When the grid leaves MORE occupied cells than
    there are slots, the uniform along-curve merge -- not the grid -- finishes the job,
    and the pooling stops being the millimetre ladder the config claims. The budget must
    therefore cover the occupancy at BOTH densities the harness runs at: sub_frac*N while
    training and the full N while evaluating.

    Over-provisioning is harmless to correctness (spare slots are masked padding) and
    costs only memory, so the choice is the LARGEST poolr that still covers both.
    """
    import fam_ptv3 as P3
    nears = NEARS if nears is None else nears
    for f, patch in zip(files, patches):
        if not os.path.exists(f):
            print(f"\n### {f}  ABSENT -- skipped"); continue
        z = np.load(f)
        C = z["clouds"]
        E, M, N = C.shape[:3]
        rs = np.random.RandomState(0)
        ears = np.linspace(0, E - 1, min(nears, E)).astype(int)
        vox = [0.85 * 2.5 ** s for s in range(1, 3)]
        occ = {}
        for tag, n in (("eval", N), ("train", int(round(N * subfrac)))):
            o = [[], []]
            for e in ears:
                P = C[e, 0].astype(np.float64)
                P = P if n == N else P[rs.choice(N, n, replace=False)]
                for i, v in enumerate(vox):
                    o[i].append(occupied_voxels(P, v))
            occ[tag] = (n, [int(np.mean(x)) for x in o], [int(np.max(x)) for x in o])
        print(f"\n### {os.path.basename(f)}  N={N} patch={patch}  occupied cells "
              f"@{vox[0]:.3f}/{vox[1]:.3f}mm")
        for tag in ("eval", "train"):
            n, mean, mx = occ[tag]
            print(f"  {tag:5s} n={n:6d}   mean {mean[0]:5d} / {mean[1]:4d}   "
                  f"WORST EAR {mx[0]:5d} / {mx[1]:4d}   <- the budget must clear the worst")
        ok = []
        for r in cands:
            fits, rows = True, []
            for tag in ("eval", "train"):
                n, _, mx = occ[tag]
                c = P3.Cfg({**P3.DEFAULTS, "patch": patch, "poolr": r, "stages": 3})
                ns = P3.counts(n, c)
                fits &= ns[1] >= mx[0] and ns[2] >= mx[1]
                rows.append(f"{tag} {ns[1]}/{ns[2]} ({mx[0]/ns[1]:.2f},{mx[1]/ns[2]:.2f})")
            print(f"  POOLR={r:<3d} {'OK ' if fits else 'over'}  " + "   ".join(rows))
            if fits:
                ok.append(r)
        print(f"  -> admissible POOLR {ok}; take the LARGEST = {max(ok) if ok else None}")
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
        # stage 0 has no padded slots, so the ONLY thing that can force an attention mask
        # there is n not being a whole number of patches. With a mask SDPA drops to a
        # materialised (B*npat, heads, P, P) score matrix -- at N=32768, patch=1024, B=16
        # that is ~5.6 GiB per block. Report the residue, do not assume it.
        pad0 = (-nsub) % c.patch
        head = (f"VOXEL={c.voxel} VOXGROW={c.voxgrow} POOLR={c.poolr} PATCH={c.patch} "
                f"STAGES={c.stages} slots {m.counts(nsub, c)} "
                f"voxels {[round(c.voxel * c.voxgrow ** s, 2) for s in range(c.stages)]}mm "
                f"| stage0 pad {pad0} -> "
                + ("NO mask, memory-efficient SDPA" if pad0 == 0 else
                   f"MASKED, materialised {c.heads}x{c.patch}^2 scores x "
                   f"{-(-nsub // c.patch)} patches x B"))
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
          f"{(pk - base if torch.cuda.is_available() else pk)/2**30:.2f} GiB"
          + ("" if torch.cuda.is_available() else
             f" (of which {(pk-base)/2**30:.2f} GiB above the pre-run baseline)")
          + f"  {dt:.1f}s")
    if fam == "ptv3" and net.enc.pool_stats:
        st = net.enc.pool_stats
        print(f"        grid pooling: " + "  ".join(
            f"{o:.0f} occupied -> {s} slots ({'MERGED, budget too small'if o > s else 'grid decides'})"
            for o, s in st))
        print("        (occupied < slots leaves PADDED slots, which force an attention "
              "mask and\n         cost the memory-efficient SDPA path at that stage; "
              "occupied > slots means the\n         uniform along-curve merge, not the "
              "millimetre grid, is doing the coarsening)")
    return out, npar


# ------------------------------------------------------- launch-config validation
def cfgcheck(fam, n, data, env):
    """Run ONE ear of the REAL data through the EXACT path train_family.py would take.

    Not a substitute for the training run -- it is the check that the environment block
    in run_res_sweep.sh actually constructs the family, that the family's own asserts
    (kpconv's use_nrm/NEEDS agreement, ptv3's npts-vs-DATA refusal) pass rather than
    fire, that the augmenter accepts the shapes, and that a gradient reaches every
    parameter. Everything that can fail in the first second of a 6-hour run.
    """
    import importlib
    keep = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        for m in ("train_family", "fam_kpconv", "fam_ptv3"):
            if m in sys.modules:          # NEEDS is read at IMPORT from the environment
                importlib.reload(sys.modules[m])
        TF = importlib.import_module("train_family")
        importlib.reload(TF)
        cls = TF.resolve_family(fam)
        cfg = {**TF.TRAIN_DEFAULTS, **getattr(cls, "DEFAULTS", {}), **TF.cfg_from_env()}
        d = TF.load_data(data, tuple(cls.NEEDS), "cpu")
        meta = dict(nl=85, contours=TF.CONTOURS, scale=30.0, npts=d["N"], fold=0,
                    dev="cpu", n_train_ears=272, artefacts={})
        model = cls(cfg, meta)
        gen = torch.Generator(device="cpu"); gen.manual_seed(0)
        b = TF.make_batch(d, [0], np.array([[0]]), 1, getattr(cls, "BATCH", None), meta)
        tg = d["true"][torch.tensor([0])]
        b, tg = TF.default_augment(b, tg, cfg, tuple(getattr(cls, "ROTATES", ("nrm",))), gen)
        b = TF._flatten_samples(b, 1)
        t0 = time.time()
        out = model(b)
        L = TF.default_loss(out, tg, model, b)
        L.backward()
        nog = [nm for nm, p in model.named_parameters() if p.grad is None]
        gn = sum(float(p.grad.norm()) ** 2 for p in model.parameters()
                 if p.grad is not None) ** .5
        print(f"  {fam:7s} N={n:6d} NEEDS={tuple(cls.NEEDS)} pts_after_aug="
              f"{tuple(b['pc'].shape)} params={sum(p.numel() for p in model.parameters()):,}"
              f" pred={tuple(out['pred'].shape)} loss={float(L):.3f} grad={gn:.3e}"
              f" ({time.time()-t0:.1f}s)")
        assert out["pred"].shape == (1, 85, 3), out["pred"].shape
        assert torch.isfinite(out["pred"]).all() and gn > 0 and not nog, nog
    finally:
        for k, v in keep.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


# ------------------------------------------------------------------ smoke test
def smoke():
    t0 = time.time()
    print("=" * 78)
    print("SMOKE 1/4 -- geometry measurement on a synthetic bundle")
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

    print("\nSMOKE 2/4 -- kpconv forward+backward, B=2")
    o1, p1 = mem("kpconv", 1024, 2, 1.0, 0,
                 dict(npts=1024, width=16, fdim=64, stages=2, r0=3.0, v0=1.5, npass=2))
    assert tuple(o1.shape) == (2, 85, 3), o1.shape

    print("\nSMOKE 3/4 -- ptv3 forward+backward, B=2")
    o2, p2 = mem("ptv3", 1024, 2, 1.0, 0,
                 dict(npts=1024, width=32, stages=2, depth=1, patch=256, headc=64,
                      voxel=1.09, voxgrow=2.5, poolr=4))
    assert tuple(o2.shape) == (2, 85, 3), o2.shape
    assert p1 > 0 and p2 > 0 and torch.isfinite(o1).all() and torch.isfinite(o2).all()

    print("\nSMOKE 4/4 -- train_family.py ACCUM is gradient-EXACT")
    # 32768-point KPConv does not fit 16 ears on 48GB, and shrinking bs would change the
    # optimisation and confound the sweep. ACCUM splits the batch instead. That is only
    # legitimate if the accumulated gradient EQUALS the full-batch one, which needs (a) no
    # BatchNorm anywhere -- every family here is LayerNorm-only -- and (b) each slice
    # reweighted by its share of the batch, since the losses are per-ear MEANS. Checked
    # entry-wise rather than asserted, and with a RAGGED tail (8 = 3+3+2) because an equal
    # split would not catch a missing reweight.
    import train_family as TF
    torch.manual_seed(1)
    meta = dict(nl=85, contours=TF.CONTOURS, scale=30.0, npts=256, fold=0, dev="cpu",
                n_train_ears=16, artefacts={})
    fm = TF.FakeFamily({**TF.TRAIN_DEFAULTS, **TF.FakeFamily.DEFAULTS, "dropout": 0.0}, meta)
    bb = {"pc": torch.randn(8, 256, 3) * 8, "coarse": torch.randn(8, 85, 3) * 8,
          "ear": torch.arange(8), "probe": torch.zeros(8, 1)}
    tg = torch.randn(8, 85, 3)
    fm.zero_grad(); TF.default_loss(fm(bb), tg, fm, bb).backward()
    ref = [p.grad.clone() for p in fm.parameters()]
    fm.zero_grad()
    for a in range(0, 8, 3):
        sl = {k: v[a:a + 3] for k, v in bb.items()}
        n = sl["ear"].shape[0]
        (TF.default_loss(fm(sl), tg[a:a + 3], fm, sl) * (n / 8)).backward()
    err = max(float((x - p.grad).abs().max()) / max(float(x.abs().max()), 1e-12)
              for x, p in zip(ref, fm.parameters()))
    print(f"  3 micro-batches of 3/3/2 vs one batch of 8: max relative entry error "
          f"{err:.2e} over {len(ref)} parameter tensors")
    assert err < 1e-5, f"ACCUM is not gradient-exact ({err:.2e})"
    # and the reweight is load-bearing: without it the ragged tail is over-weighted
    fm.zero_grad()
    for a in range(0, 8, 3):
        sl = {k: v[a:a + 3] for k, v in bb.items()}
        (TF.default_loss(fm(sl), tg[a:a + 3], fm, sl) / 3).backward()
    bad = max(float((x - p.grad).abs().max()) / max(float(x.abs().max()), 1e-12)
              for x, p in zip(ref, fm.parameters()))
    print(f"  same split with a naive 1/nslice weight instead: {bad:.2e} -- the reweight "
          f"is not cosmetic")
    assert bad > 1e-3

    # ...and the LIMIT of the exactness claim, measured rather than left implicit.
    # FakeFamily has no Dropout, so the two checks above and the end-to-end run below are
    # all blind to the one thing ACCUM really does perturb: dropout draws its mask per
    # FORWARD, so 4 micro-batches draw 4 masks where 1 batch drew 1. kpconv and ptv3 both
    # ship dropout=0.1, and kpconv-32768 is the only arm that runs ACCUM>1, so say the
    # size of it out loud instead of discovering it later.
    class _Drop(TF.FakeFamily):
        def __init__(self, cfg, meta):
            super().__init__(cfg, meta)
            self.dp = torch.nn.Dropout(float(cfg["dropout"]))

        def forward(self, b):
            o = super().forward(b)
            return {**o, "pred": self.dp(o["pred"])}

    for p in (0.0, 0.1):
        torch.manual_seed(1)
        fd = _Drop({**TF.TRAIN_DEFAULTS, **TF.FakeFamily.DEFAULTS, "dropout": p}, meta)
        fd.train()
        torch.manual_seed(7); fd.zero_grad()
        TF.default_loss(fd(bb), tg, fd, bb).backward()
        rf = [q.grad.clone() for q in fd.parameters()]
        torch.manual_seed(7); fd.zero_grad()
        for a in range(0, 8, 3):
            sl = {k: v[a:a + 3] for k, v in bb.items()}
            (TF.default_loss(fd(sl), tg[a:a + 3], fd, sl) * (sl["ear"].shape[0] / 8)).backward()
        e = max(float((x - q.grad).abs().max()) / max(float(x.abs().max()), 1e-12)
                for x, q in zip(rf, fd.parameters()))
        print(f"  with DROPOUT={p}: {e:.2e} -- "
              + ("exact" if e < 1e-5 else "a DIFFERENT dropout realisation, not the same run"))
        assert (e < 1e-5) == (p == 0.0), \
            f"dropout={p} exactness changed; the ACCUM caveat in run_res_sweep.sh is stale"

    # ...and the same thing END TO END through main(), because gradient equality is not
    # enough: ACCUM must also not perturb the RANDOM STREAM. train_family builds and
    # augments the whole batch and slices only the forward, so ACCUM=4 must reproduce
    # ACCUM=1's val MLE to float-summation error over a real (tiny) training run.
    import tempfile
    tmp2 = os.path.join(tempfile.gettempdir(), "res_sweep_accum")
    dp, tp, sp = TF.fake_bundle(tmp2)[:3]
    base = dict(FAMILY="fake", FOLD="0", SEED="0", EPOCHS="4", WORK=tmp2, DATA=dp,
                TRIS=tp, SSM=sp, TTA="2", EVAL_EVERY="4", ALIAS="0", CFG_BS="8",
                CFG_WIDTH="16", FULL_EVAL="0")
    keep = {k: os.environ.get(k) for k in list(base) + ["ACCUM", "TAG"]}
    got = {}
    try:
        for acc in ("1", "4"):
            os.environ.update({**base, "ACCUM": acc, "TAG": f"accum{acc}"})
            r = TF.main()
            got[acc] = (r["ordered_MLE_mm"], r["config"]["_micro_bs"])
    finally:
        for k, v in keep.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    d = abs(got["1"][0] - got["4"][0])
    print(f"  end-to-end: ACCUM=1 (micro-bs {got['1'][1]}) val {got['1'][0]:.4f} mm vs "
          f"ACCUM=4 (micro-bs {got['4'][1]}) val {got['4'][0]:.4f} mm -- delta {d:.1e} mm")
    assert d < 1e-3, f"ACCUM changes the training run by {d} mm, not just its memory"

    print(f"\nSMOKE PASS  ({time.time()-t0:.1f}s)")
    print("=" * 78)


if __name__ == "__main__":
    if MODE == "geom":
        # 8 ears leaves a 1.4% standard error on the one number the whole rescaling hangs
        # on, and the linspace(0,339,8) subset is 2.7% high. Default to 64; EARS overrides.
        geom(os.environ.get("FILES", "scratch/screen_data_8192nrm.npz").split(","),
             nears=int(os.environ["EARS"]) if "EARS" in os.environ else 64)
    elif MODE == "cfgcheck":
        # the exact per-arm blocks run_res_sweep.sh emits; keep the two in step
        ARM = {8192:  dict(v0=1.067, hk=48,  km=96,  ch=1024, patch=256,  poolr=2),
               16384: dict(v0=0.754, hk=96,  km=192, ch=512,  patch=512,  poolr=4),
               32768: dict(v0=0.533, hk=192, km=384, ch=256,  patch=1024, poolr=8)}
        for n in [int(x) for x in os.environ.get("NS", "8192,16384,32768").split(",")]:
            a, dp = ARM[n], f"scratch/screen_data_{n}nrm.npz"
            if not os.path.exists(dp):
                print(f"  {n}: {dp} ABSENT -- skipped"); continue
            cfgcheck("kpconv", n, dp, dict(
                USE_NRM=1, NPTS=n, CFG_NPTS=n, CFG_V0=a["v0"], CFG_R0=2.5,
                CFG_HEAD_K=a["hk"], CFG_KMAX=a["km"], CFG_NB_CHUNK=a["ch"],
                CFG_NB_STATS=1))
            cfgcheck("ptv3", n, dp, dict(
                USE_NRM=1, NPTS=n, CFG_NPTS=n, CFG_PATCH=a["patch"], CFG_K=a["hk"],
                CFG_POOLR=a["poolr"], CFG_VOXEL=0.85, CFG_VOXGROW=2.5))
    elif MODE == "ladder":
        # KMAX is a SAFETY cap read off a WORST case, so the default sample here is 40
        # ears rather than the 8 the other modes use: the worst ball is an extreme-value
        # statistic and 8 ears underestimate it (measured 8192 eval worst: 59 at 12 ears,
        # 42 at 8). EARS in the environment still overrides.
        fl = os.environ.get("FILES", "scratch/screen_data_8192nrm.npz").split(",")
        ladder(fl, [float(x) for x in os.environ.get("V0S", "1.096").split(",")],
               nears=int(os.environ["EARS"]) if "EARS" in os.environ else 40)
    elif MODE == "pool":
        fl = os.environ.get("FILES", "scratch/screen_data_8192nrm.npz").split(",")
        pool_budget(fl, [int(x) for x in os.environ.get("PATCHES", "256").split(",")])
    elif MODE == "mem":
        mem(os.environ.get("FAM", "kpconv"), int(os.environ.get("N", "8192")),
            int(os.environ.get("B", "2")), float(os.environ.get("SUBFRAC", "0.625")),
            int(os.environ.get("NOGRAD", "0")),
            json.loads(os.environ.get("CFG_JSON", "{}")))
    else:
        smoke()
