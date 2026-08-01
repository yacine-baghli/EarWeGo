#!/bin/bash
# Extra seeds for the two cross-family members. dgcnn contributes a 3-seed mean while
# kpconv and ptv3 contribute one seed each, so their members are noisier than they need
# to be. Seed-ensembling was worth -0.044mm on dgcnn; this is the reliable increment
# while the profile/endpoint work is built.
cd /home/ubuntu/ear
D=/home/ubuntu/ear/screen_data_8192nrm.npz
for f in 0 1 2 3 4; do
  echo "=== kpconv SEED=1 FOLD=$f ==="
  WORK=/home/ubuntu/ear DATA=$D FULL_EVAL=0 USE_NRM=1 NPTS=8192 CFG_NPTS=8192 \
    FAMILY=kpconv FOLD=$f SEED=1 EPOCHS=1200 TAG=famA_kpconv_s1_f$f \
    python3 -u train_family.py 2>&1 | grep -E "raw MLE|Error|Traceback"
  echo "=== ptv3 SEED=1 FOLD=$f ==="
  WORK=/home/ubuntu/ear DATA=$D FULL_EVAL=0 USE_NRM=1 NPTS=8192 CFG_NPTS=8192 \
    CFG_POOLR=4 CFG_VOXEL=0.85 FAMILY=ptv3 FOLD=$f SEED=1 EPOCHS=1200 \
    TAG=famA_ptv3_s1_f$f python3 -u train_family.py 2>&1 | grep -E "raw MLE|Error|Traceback"
done
echo FAMA_SEEDS_DONE
