#!/bin/bash
# Family C viability probe: ONE full-length run before any hyperparameter search.
# If the family cannot approach the 1.32mm pooled-OOF baseline on a single fold, a
# 19-hour 5x3 block and a nested search would both be premature.
cd /home/ubuntu/ear
for spec in "t2s dgcnn" "s2t dgcnn"; do
  set -- $spec
  echo "=== famC DIRECTION=$1 ENCODER=$2 FOLD=0 ==="
  WORK=/home/ubuntu/ear DATA=/home/ubuntu/ear/screen_data_2048.npz \
    ARTEFACTS=/home/ubuntu/ear/template_f0.npz FULL_EVAL=0 \
    CFG_DIRECTION=$1 CFG_ENCODER=$2 TAG=famC_$1_$2_f0 \
    FAMILY=template FOLD=0 SEED=0 EPOCHS=1200 python3 -u train_family.py 2>&1 | tail -4
done
echo FAMC_PROBE_DONE
