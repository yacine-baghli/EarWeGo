#!/bin/bash
# kpconv + ptv3 on folds 1-4 (fold 0 exists) so cross-family ensembling can be measured
# over all 340 ears with nested-OOF weights. One seed each for now: this establishes
# whether the ensemble gain is real, not an adoption decision. pointnext is dropped --
# it is the most decorrelated (0.773) but too weak (1.4022) to help, and the slowest.
cd /home/ubuntu/ear
D=/home/ubuntu/ear/screen_data_8192nrm.npz
for f in 1 2 3 4; do
  echo "=== kpconv FOLD=$f ==="
  WORK=/home/ubuntu/ear DATA=$D FULL_EVAL=0 USE_NRM=1 NPTS=8192 CFG_NPTS=8192 \
    FAMILY=kpconv FOLD=$f SEED=0 EPOCHS=1200 TAG=famA_kpconv_f$f \
    python3 -u train_family.py 2>&1 | tail -2
  echo "=== ptv3 FOLD=$f ==="
  WORK=/home/ubuntu/ear DATA=$D FULL_EVAL=0 USE_NRM=1 NPTS=8192 CFG_NPTS=8192 \
    CFG_POOLR=4 CFG_VOXEL=0.85 FAMILY=ptv3 FOLD=$f SEED=0 EPOCHS=1200 \
    TAG=famA_ptv3_f$f python3 -u train_family.py 2>&1 | tail -2
done
echo FAMA_CV_DONE
