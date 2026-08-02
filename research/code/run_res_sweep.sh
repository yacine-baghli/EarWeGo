#!/bin/bash
# =============================================================================
# RESOLUTION SWEEP -- kpconv and ptv3 at 8192 / 16384 / 32768 points, fold 0.
#
# THE QUESTION: is 1.1776mm a RESOLUTION limit? Every model so far has run at
# 1.09-2.19mm point spacing while the mesh has ~0.67mm vertex spacing and the GT
# landmarks sit 0.021mm off that surface. 32768 area-weighted surface samples give
# 0.55mm spacing -- finer than the mesh.
#
# THE CONFOUND THIS SCRIPT EXISTS TO KILL. The earlier 8192-point DGCNN test was
# undecidable because K and GK were held FIXED while the spacing halved: "more points"
# silently also meant "a 2x smaller physical window". Here EVERY PARAMETER THAT DENOTES
# A DISTANCE IS HELD IN MILLIMETRES and every parameter that denotes an INDEX COUNT is
# rescaled so its millimetre meaning does not move.
#
# THE SCALING LAW IS N, NOT sqrt(N). On a surface, samples inside a ball of radius r
# number pi*r^2*N/A, so an index count holding a fixed mm radius scales as N. Spacing is
# what scales as N^-1/2. Measured on the real 340-ear clouds by res_sweep_prep.py
# MODE=geom (8 ears each, per-ear mean ball size):
#
#     N       spacing   n(2.5mm)  n(5mm)  n(10mm)  n(20mm)   head window   snap-jitter
#     8192    1.0965      17.6      75.2    368.2   1676.1   4.29mm k=48    0.389mm
#     16384   0.7739      34.1     149.1    734.9   3355.3   4.28mm k=96    0.274mm
#     32768   0.5468      67.2     297.3   1466.8   6711.0   4.27mm k=192   0.195mm
#
# The fitted exponent of n(r) against N is 0.99-1.00 for r >= 5mm and 0.97 at 2.5mm.
# It is NOT 0.5. The k column is what holds the head window at 4.29mm, and it lands within
# 0.01mm at every arm. "snap-jitter" is the per-sample sd of the k-NN centroid the
# offset/snap head returns, across the 4 fresh surface samples: resolution buys a 2x
# quieter snap target for free, before any learning, and TTA=4 divides it again by 2
# (0.194 -> 0.097mm at 32768). Bounded, though: against a ~1.25mm error that whole
# mechanism is worth at most 0.011mm, so resolution has to pay off somewhere else.
#
# KMAX IS SET BY HAND, AND THE AUTO VALUE WOULD HAVE CONFOUNDED THE SWEEP. fam_kpconv
# derives KMAX = ceil(2.8*pi*(R0/V0)^2) = 46 -> 92 -> 184 on the argument that r_l/v_l is
# constant so every level has the same ball occupancy. On a real folded pinna that is
# false: grid_subsample at a fine voxel keeps relatively more of the surface than the
# doubling predicts, so the DEEP levels are 2-3x denser than the design. Measured on 40
# real ears through fam_kpconv's own ladder (res_sweep_prep.py MODE=ladder), the worst
# ball at any level is 59 at 8192 and 127 at 16384 -- both ABOVE the auto cap. The shipped
# 8192 config therefore already truncates 2.6% of its 10mm balls and 8.0% of its 20mm
# balls, and that fraction GROWS with N (12.1% at 16384). A sweep whose deepest receptive
# field is silently clipped harder at 32768 than at 8192 is not a resolution experiment;
# it is a resolution experiment plus a shrinking deep window, biased against the thing
# under test.
#
#   CFG_KMAX = 96 / 192 / 384. Still exactly proportional to N, so it is the same
#   millimetre rescaling as everything else, but anchored on the MEASURED worst ball over
#   40 / 40 / 20 real ears. At the CORRECTED V0 (1.067/0.754/0.533 -- see arm_env) the
#   voxels are 2.7% finer and the deep levels correspondingly denser, so the worst ball
#   is 59 / 133 / 264 at training density and 60 / 140 / 274 at evaluation density.
#   Against the 1.235 density factor a -10% aug_scale adds and 1.05 for ears not
#   sampled, that leaves a 1.29 / 1.06 / 1.08 margin -- thin at the two upper arms but
#   positive, and the audit run on real ears at all three arms prints frac_truncated
#   0.000 at EVERY level. The run's own audit is in $WORK/<tag>.log: if it says anything
#   other than 0.000, stop and re-run MODE=ladder rather than reading the result.
#
# It is not free: kpconv memory and step time are both linear in KMAX, so this is a ~2.1x
# tax on every kpconv arm relative to the auto cap. It is paid equally by all three, so it
# does not touch the comparison -- only the bill. It is why the 32768 kpconv arm needs
# ACCUM=4.
#
# A SECOND THING THE MEASUREMENT EXPOSED, worth saying out loud because it shapes what
# this sweep can conclude: kpconv's voxel ladder is v0*2^l, so raising N does not only
# refine the INPUT, it refines every pooled level too (voxels 1.10/2.19/4.38/8.77mm at
# 8192 become 0.55/1.09/2.19/4.38mm at 32768). The conv RADII are held, so the receptive
# field ladder is unchanged, but the sweep tests "denser at every level", not "denser at
# the input only". Separating those would need a per-level voxel override fam_kpconv does
# not have. ptv3, by contrast, has its pooling grid pinned in millimetres, so ITS arms
# differ at stage 0 only -- which makes the two families a useful pair here.
#
# WHAT IS HELD, ARM TO ARM
#     kpconv   conv radii 2.5 / 5 / 10 / 20 mm            (already millimetres)
#              nkp kp_rho kp_sigma stages width bottle
#     ptv3     voxel 0.85mm, voxgrow 2.5 -> pooling grids 2.125mm and 5.31mm
#              attention window s*sqrt(patch) = 17.5mm
#              stages depth width heads mlp bits
#     both     lr bs wd sub_frac aug_* epochs fold seed, and M=4 fresh samples
#
# WHAT MOVES, AND ONLY BECAUSE THE MILLIMETRES DEMAND IT
#     kpconv   V0 = the MEASURED grid-equivalent spacing (a property of the data, not a
#              knob). KMAX is then set BY HAND to 96/192/384 rather than left to the
#              auto rule, for the reason given above; it is still exactly ~N.
#              HEAD_K 48 -> 96 -> 192 holds the 4.29mm head window.
#              NB_CHUNK only caps a transient; it changes no result.
#     ptv3     PATCH 256 -> 512 -> 1024 holds the 17.5mm attention window.
#              K 48 -> 96 -> 192 holds the head window, as for kpconv.
#              POOLR 2 -> 4 -> 8 makes the SLOT BUDGET cover the MEASURED occupied-cell
#              count of the 2.125mm and 5.31mm grids at BOTH the training density
#              (sub_frac*N) and the evaluation density (N) -- i.e. so the GRID does the
#              coarsening rather than the uniform along-curve merge. Occupancy is a
#              property of the EAR and grows only 2434 -> 2679 -> 2837 cells at 2.125mm
#              over a 4x point increase, so this ratio ladder lands the three arms on
#              IDENTICAL budgets: eval slots [N, 4096, 2048/1024/1024], train slots
#              [.625N, 2560/2560/3072, 1280/1024/1024]. The pooled stages are then
#              physically the same across arms and only stage 0 changes -- which is
#              exactly the variable under test.
#              MEASURED worst-ear utilisation of the stage-1 budget: 0.89 / 1.05 / 0.96
#              training, 0.63 / 0.70 / 0.75 evaluating.
#              The 16384 arm is 5% over on its worst ear, so ~5% of that ear's coarse
#              voxels get merged with a curve-neighbour. Taken deliberately: the strictly
#              largest admissible value there is POOLR=3, but that would give it 3584
#              training slots against the other arms' 2560/3072 and break the very
#              matching this ladder exists for. res_sweep_prep.py MODE=pool prints the
#              full admissibility table -- re-run it if any of these numbers is edited.
#
# THE 8192 ARM IS RE-RUN, NOT TAKEN FROM HISTORY. The shipped 8192 members (kpconv
# 1.2516, ptv3 1.2417) ran with V0 at its 1.0 default against a real 1.0965mm spacing,
# with head_k/k left at 48 without ever asking what 48 meant in mm, and with ptv3's
# POOLR=4 giving 2048 stage-1 slots for a measured worst-ear 2595 occupied cells at
# evaluation and 1280 slots for 2272 at training -- 1.27x and 1.77x over budget, so the
# merge and not the grid was doing most of the coarsening. Comparing a rescaled 16384 run
# against those numbers would mix "more resolution" with "a corrected convention". The
# sweep is read off the THREE NUMBERS THIS SCRIPT PRODUCES; 1.2516/1.2417 are history,
# not the control. Expect the re-run 8192 numbers to differ from them, in either
# direction, and do not read that difference as a resolution effect.
#
# EFFECTIVE BATCH IS HELD AT 16 EVERYWHERE. KPConv's neighbour tensors scale as N*KMAX,
# and both are proportional to N, so its cost is QUADRATIC. CPU-proxy peak of one training
# forward+backward at B=1, and the CPU time of that same call at B=2:
#
#              kpconv                        ptv3
#     N        GiB/ear   fwd+bwd  x8192      GiB/ear  fwd+bwd  x8192
#     8192      0.45      4.8 s    1.0        0.32     4.0 s    1.0
#     16384     1.30     17.7 s    3.7        0.41     4.9 s    1.2
#     32768     3.4-4.6  79.9 s   16.6        0.78     8.4 s    2.1
#
# So kpconv at B=16 needs ~7 / ~19 / ~60 GiB and does NOT fit at 32768 on a 48 GB card
# alongside 1.1 GiB of resident cloud data; ACCUM=4 (micro-batch 4) brings it to ~18 GiB.
# ptv3 stays LINEAR -- 12 GiB at B=16 even at 32768 -- because its stage-0 attention stays
# on the memory-efficient SDPA path and its pooled stages are pinned in millimetres.
# The 32768 kpconv row is the least trustworthy number here: three repeated probes gave
# 3.40, 4.60 and 5.16 GiB, i.e. the CPU RSS proxy has ~+-30% spread at that size.
# Every family here is LayerNorm-only, and train_family builds and AUGMENTS the whole
# batch and slices only the forward, so ACCUM leaves the gradient and the AUGMENTATION
# stream alone: res_sweep_prep.py's smoke checks the gradient entry-wise (4.3e-07
# relative at dropout=0), checks that dropping the per-slice reweight breaks it
# (3.5e-01), and runs the trainer end to end at ACCUM=1 and ACCUM=4 for an identical val
# MLE (delta 0.0e+00 mm). Shrinking bs instead would change the optimisation.
#
# THE ONE THING ACCUM DOES CHANGE, measured (smoke 4/4, dropout row): DROPOUT masks are
# drawn per slice, so at the shipped dropout=0.1 the ACCUM=4 gradient differs from the
# ACCUM=1 one by O(1) relative -- a different realisation of the same distribution, not
# a bias. Only kpconv-32768 runs ACCUM>1, so exactly one of the six cells carries an
# extra seed-sized nuisance on top of the seed noise every cell already has. That is
# inside the >=0.08mm read-out threshold below, but it is NOT the "byte-identical run"
# the rest of this paragraph would otherwise imply. Set CFG_DROPOUT=0 on every arm if
# you need the arms bit-comparable.
#
# THOSE ARE CPU-RSS PROXY NUMBERS, NOT CUDA. PREFLIGHT 1 below re-measures them with
# torch.cuda.max_memory_allocated on the box before anything trains. Trust that, not this.
#
# ONE LANDMINE, IF ANYONE EDITS THE NUMBERS. ptv3 keeps SDPA on the memory-efficient path
# at stage 0 only while the point count is an exact multiple of `patch`; otherwise
# Block.forward pads, builds an additive mask, and falls back to a materialised
# (B*npat, heads, patch, patch) score matrix -- at 32768 with patch=1024 and B=16 that is
# ~5.6 GiB PER BLOCK and the arm will OOM. It works here only because sub_frac=0.625 and
# patch=N/32 make n/patch exactly 20 while training and 32 while evaluating, for all three
# arms. Change sub_frac, or set patch to anything other than N/32, and re-check.
#
# COST. Nobody in this repo has recorded a GPU wall clock for a 1200-epoch famA run, so
# the ABSOLUTE cost of this sweep is not known in advance and is not guessed here.
# PREFLIGHT 2 measures it with a 12-epoch probe (1% of a run) and prints a projection for
# each of the six cells BEFORE any of them starts, so the decision to spend is taken on a
# measurement. The RELATIVE cost is measured (table above): on CPU, one kpconv step costs
# 1.0 / 3.7 / 16.6 units across the arms and one ptv3 step 1.0 / 1.2 / 2.1, and the KMAX
# correction adds ~2.1x to every kpconv arm on top of that. Expect kpconv-32768 to be the
# single dominant line item, plausibly 15-35x the kpconv-8192 arm once GPU parallelism
# compresses the ratio somewhat -- but that compression is a guess and the probe is not.
# If the projection is unaffordable, DROP THE 16384 ARM rather than shortening the runs:
# 8192-vs-32768 is the largest contrast, and EPOCHS must stay equal across arms or the
# comparison is between two schedules rather than two resolutions. Dropping kpconv-32768
# and keeping ptv3-32768 is the other cheap option, and ptv3 is arguably the better probe
# of resolution anyway because its pooled stages are pinned in millimetres.
#
#   bash run_res_sweep.sh                    # preflight, then all 6 runs
#   PREFLIGHT_ONLY=1 bash run_res_sweep.sh   # memory + timing probes only -- DO THIS FIRST
#   ARMS="8192 32768" bash run_res_sweep.sh  # the cheap two-arm version
#   ARMS="32768" FAMS="kpconv" bash run_res_sweep.sh
#
# FILES THIS NEEDS ON THE BOX
#   screen_data_16384nrm.npz (244 MB) / screen_data_32768nrm.npz (~487 MB)   fp16, M=4
#   train_family.py  <- RE-UPLOAD, it gained ACCUM
#   res_sweep_prep.py, fam_kpconv.py, fam_ptv3.py
# =============================================================================
set -u
cd "${EAR:-/home/ubuntu/ear}" || exit 1

WORK=${WORK:-$(pwd)}
ARMS=${ARMS:-"8192 16384 32768"}
FAMS=${FAMS:-"kpconv ptv3"}
FOLD=${FOLD:-0}
SEED=${SEED:-0}
EPOCHS=${EPOCHS:-1200}
PREFLIGHT=${PREFLIGHT:-1}
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}
PROBE_EPOCHS=${PROBE_EPOCHS:-12}

# ---- per-arm parameters. Everything here is DERIVED, not tuned; see the header.
arm_env() {                       # $1 = family, $2 = N   ->  echoes the env assignments
  # S was first measured over EARS=8 and came out 1.096 / 0.774 / 0.547. The crop area is
  # an ear property spanning 8000..10418 mm^2, so 8 ears leave a 1.4% standard error and
  # that particular subset sits 2.7% high: 24 and 64 ears both give 1.067 / 0.754 / 0.533,
  # and build_hires_data's own triangulated crop area (median 9416 mm^2) agrees. The bias
  # was IDENTICAL at all three arms (+2.68/+2.64/+2.66%), so it never touched the
  # comparison -- only the absolute millimetre label. Corrected here, and the neighbour
  # audit re-run at these values prints frac_truncated 0.000 at every level of every arm.
  case "$2" in
    8192)  S=1.067;  HK=48;  CH=1024; KMAX=96;  PATCH=256;  POOLR=2; KACC=1; PACC=1 ;;
    16384) S=0.754;  HK=96;  CH=512;  KMAX=192; PATCH=512;  POOLR=4; KACC=1; PACC=1 ;;
    32768) S=0.533;  HK=192; CH=256;  KMAX=384; PATCH=1024; POOLR=8; KACC=4; PACC=1 ;;
    *) echo "unknown arm $2" >&2; return 1 ;;
  esac
  if [ "$1" = kpconv ]; then
    echo "CFG_V0=$S CFG_R0=2.5 CFG_HEAD_K=$HK CFG_KMAX=$KMAX CFG_NB_CHUNK=$CH ACCUM=$KACC"
  else
    echo "CFG_PATCH=$PATCH CFG_K=$HK CFG_POOLR=$POOLR CFG_VOXEL=0.85 CFG_VOXGROW=2.5 ACCUM=$PACC"
  fi
}

data_for() { echo "$WORK/screen_data_$1nrm.npz"; }

# The box is FLAT: every other run_*.sh here calls `python3 -u train_family.py` from
# $EAR, i.e. uploads land in /home/ubuntu/ear/ with no research/code/ tree. Hard-coding
# research/code/res_sweep_prep.py made BOTH preflight-1 probes die with "can't open
# file", and the `|| echo FAILED (OOM?)` below would have reported that as an OOM --
# the exact opposite of what it means. Resolve it, and refuse to start if it is absent
# rather than let the preflight silently degrade into six fake OOMs.
PREP=${PREP:-}
if [ -z "$PREP" ]; then
  for p in res_sweep_prep.py research/code/res_sweep_prep.py; do
    [ -f "$p" ] && { PREP=$p; break; }
  done
fi
[ -n "$PREP" ] && [ -f "$PREP" ] || {
  echo "res_sweep_prep.py not found under $(pwd) -- upload it (PREP=<path> overrides)" >&2
  exit 1
}

for N in $ARMS; do
  D=$(data_for "$N")
  [ -f "$D" ] || { echo "MISSING $D -- build it with"; \
    echo "  NPTS=$N M=4 DTYPE=fp16 NSHARD=4 SHARD=k python research/code/build_hires_data.py"; \
    exit 1; }
done

# ---- preflight 1: MEASURED peak memory, per family per arm, before anything trains.
# The batch sizes below are the ones the runs will actually use; if a probe OOMs, raise
# ACCUM for that arm in arm_env() rather than lowering bs.
if [ "$PREFLIGHT" = 1 ]; then
  echo "=== PREFLIGHT 1/2: measured peak memory (train forward+backward, sub_frac=0.625)"
  for N in $ARMS; do
    for F in $FAMS; do
      eval "$(arm_env "$F" "$N")"
      MB=$(( 16 / ${ACCUM:-1} ))
      if [ "$F" = kpconv ]; then
        CJ="{\"npts\":$N,\"v0\":$CFG_V0,\"r0\":2.5,\"head_k\":$CFG_HEAD_K,\"kmax\":$CFG_KMAX,\"nb_chunk\":$CFG_NB_CHUNK,\"use_nrm\":1}"
      else
        CJ="{\"npts\":$N,\"patch\":$CFG_PATCH,\"k\":$CFG_K,\"poolr\":$CFG_POOLR,\"voxel\":0.85,\"voxgrow\":2.5,\"use_nrm\":1}"
      fi
      MODE=mem FAM=$F N=$N B=$MB SUBFRAC=0.625 NOGRAD=0 CFG_JSON="$CJ" \
        python3 -u "$PREP" || echo "  ^^ $F $N B=$MB FAILED (OOM?)"
      MODE=mem FAM=$F N=$N B=1 SUBFRAC=1.0 NOGRAD=1 CFG_JSON="$CJ" \
        python3 -u "$PREP" || echo "  ^^ $F $N eval FAILED"
    done
  done

  # ---- preflight 2: wall clock. PROBE_EPOCHS epochs, then multiply.
  echo "=== PREFLIGHT 2/2: timing probe ($PROBE_EPOCHS epochs) -> projected $EPOCHS-epoch cost"
  for N in $ARMS; do
    for F in $FAMS; do
      T="res_probe_${F}_n${N}"
      env $(arm_env "$F" "$N") WORK=$WORK DATA=$(data_for "$N") FULL_EVAL=0 ALIAS=0 \
        USE_NRM=1 NPTS=$N CFG_NPTS=$N FAMILY=$F FOLD=$FOLD SEED=$SEED \
        EPOCHS=$PROBE_EPOCHS EVAL_EVERY=$PROBE_EPOCHS TTA=1 TAG=$T \
        python3 -u train_family.py 2>&1 | tail -4
      python3 - "$WORK/$T.json" "$EPOCHS" "$PROBE_EPOCHS" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
f = int(sys.argv[2]) / int(sys.argv[3])
print(f"  PROJECTED {r['variant']} {r['config']['_data'].split('_')[-1]}: "
      f"{r['runtime_s']*f/3600:.1f} h for {sys.argv[2]} epochs "
      f"(probe {r['runtime_s']:.0f}s, accum {r['config']['_accum']}, "
      f"micro-bs {r['config']['_micro_bs']})")
PY
    done
  done
  [ "$PREFLIGHT_ONLY" = 1 ] && { echo RES_SWEEP_PREFLIGHT_DONE; exit 0; }
fi

# ---- the sweep itself
for N in $ARMS; do
  for F in $FAMS; do
    T=res_${F}_n${N}_f${FOLD}_s${SEED}
    echo "=== RES SWEEP $F N=$N fold $FOLD seed $SEED  -> $WORK/$T.log ==="
    # the FULL log is kept: the kpconv neighbour audit (frac_truncated per layer) and
    # ptv3's occupied->slots line are printed ONCE at the start of a run and are the only
    # evidence that the rescaled parameters did what the header claims. `tail` would eat
    # them. Only the lines worth reading live go to stdout.
    env $(arm_env "$F" "$N") WORK=$WORK DATA=$(data_for "$N") FULL_EVAL=0 ALIAS=0 \
      USE_NRM=1 NPTS=$N CFG_NPTS=$N FAMILY=$F VARIANT=res_${F}_$N \
      FOLD=$FOLD SEED=$SEED EPOCHS=$EPOCHS TAG=$T \
      python3 -u train_family.py 2>&1 | tee "$WORK/$T.log" \
      | grep -E 'FOLD |params \||^ *[0-9] +|^(same|strided)_|occupied|raw MLE|pipeline:|Error|Traceback|assert'
  done
done

echo "=== SWEEP SUMMARY (raw val MLE, fold $FOLD -- NOT pooled OOF, NOT full-pipeline)"
python3 - "$WORK" "$FOLD" "$SEED" "$ARMS" "$FAMS" <<'PY'
import json, os, sys
work, fold, seed = sys.argv[1], sys.argv[2], sys.argv[3]
for f in sys.argv[5].split():
    row = []
    for n in sys.argv[4].split():
        p = f"{work}/res_{f}_n{n}_f{fold}_s{seed}.json"
        if not os.path.exists(p):
            row.append(f"{n}: --"); continue
        r = json.load(open(p))
        row.append(f"{n}: {r['ordered_MLE_mm']:.4f} ({r['runtime_s']/3600:.1f}h)")
    print(f"  {f:8s} " + "   ".join(row))
print("  one fold and one seed resolves ~0.04mm at best -- read a >=0.08mm move as real,")
print("  anything smaller needs the other four folds before it means anything.")
PY
echo RES_SWEEP_DONE
