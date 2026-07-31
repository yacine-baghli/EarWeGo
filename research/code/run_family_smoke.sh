#!/bin/bash
# Integration smoke at the DATA's own resolution (2048), only to prove the modules run
# on the real box. The families are designed for 8k+; real viability probes use
# screen_data_8192nrm.npz. NOT a decision mechanism.
cd /home/ubuntu/ear
echo "=== SMOKE pointnext ==="
WORK=/home/ubuntu/ear DATA=/home/ubuntu/ear/screen_data_2048nrm.npz FULL_EVAL=0 \
  USE_NRM=1 CFG_NPTS=2048 FAMILY=pointnext FOLD=0 SEED=0 EPOCHS=40 EVAL_EVERY=20 \
  TAG=smoke_pointnext timeout 1800 python3 -u train_family.py 2>&1 | tail -4
echo "=== SMOKE ptv3 ==="
WORK=/home/ubuntu/ear DATA=/home/ubuntu/ear/screen_data_2048nrm.npz FULL_EVAL=0 \
  USE_NRM=1 CFG_NPTS=2048 CFG_POOLR=2 CFG_VOXEL=1.3 FAMILY=ptv3 FOLD=0 SEED=0 \
  EPOCHS=40 EVAL_EVERY=20 TAG=smoke_ptv3 timeout 1800 python3 -u train_family.py 2>&1 | tail -4
echo FAMILY_SMOKE_DONE
