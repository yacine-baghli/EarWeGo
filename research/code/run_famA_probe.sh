#!/bin/bash
# Family A viability probes at 8192 points with corrected normals, full length, fold 0.
# Triage before committing 5x3 blocks -- not a decision mechanism. Neighbourhood sizes
# are scaled to hold the PHYSICAL window: spacing sqrt(A/N) is 1.09mm at 8192 vs 2.19mm
# at 2048, so index counts double where the code does not derive them from mm already.
cd /home/ubuntu/ear
D=/home/ubuntu/ear/screen_data_8192nrm.npz
echo "=== famA kpconv 8192 ==="
WORK=/home/ubuntu/ear DATA=$D FULL_EVAL=0 USE_NRM=1 NPTS=8192 CFG_NPTS=8192 \
  FAMILY=kpconv FOLD=0 SEED=0 EPOCHS=1200 TAG=famA_kpconv_f0 \
  python3 -u train_family.py 2>&1 | tail -3
echo "=== famA ptv3 8192 ==="
WORK=/home/ubuntu/ear DATA=$D FULL_EVAL=0 USE_NRM=1 NPTS=8192 CFG_NPTS=8192 \
  CFG_POOLR=4 CFG_VOXEL=0.85 FAMILY=ptv3 FOLD=0 SEED=0 EPOCHS=1200 TAG=famA_ptv3_f0 \
  python3 -u train_family.py 2>&1 | tail -3
echo "=== famA pointnext 8192 ==="
WORK=/home/ubuntu/ear DATA=$D FULL_EVAL=0 USE_NRM=1 NPTS=8192 CFG_NPTS=8192 \
  FAMILY=pointnext FOLD=0 SEED=0 EPOCHS=1200 TAG=famA_pointnext_f0 \
  python3 -u train_family.py 2>&1 | tail -3
echo FAMA_PROBE_DONE
