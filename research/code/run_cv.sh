#!/bin/bash
# 5-fold GroupKFold CV (grouped by subject). Each fold trains on 136 subjects and
# predicts the held-out 34 -> out-of-fold predictions covering all 340 ears exactly
# once. Gives a reliable estimate (340 ears vs 60) plus 5 split-diverse models.
cd /home/ubuntu/ear
for f in 0 1 2 3 4; do
  echo "===FOLD $f START==="
  FOLD=$f SEED=$f EQUI=0 \
    DATA=/home/ubuntu/ear/deep_dataset.npz \
    OUT=/home/ubuntu/ear/gpu_cv_f$f.npz \
    python3 -u gpu_train.py > cv_f$f.log 2>&1
  tail -2 cv_f$f.log
done
echo "===CV DONE==="
