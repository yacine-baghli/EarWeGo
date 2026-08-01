#!/bin/bash
# Family F, second attempt. The first used NCTRL=16, which is 53-64% of the degrees of
# freedom of the larger contours -- a band that costs ~0.95mm of pure REPRESENTATION
# error before any learning (measured: research/results/curve_floor.json). That, not the
# phase idea, is what produced 1.81mm. NCTRL=32 exceeds every contour's landmark count
# (max 30), so the curve can represent the targets exactly and the only remaining
# structural constraint is the one this family exists to impose: monotone ordering.
cd /home/ubuntu/ear
for nc in 32 24; do
  echo "=== famF NCTRL=$nc FOLD=0 ==="
  WORK=/home/ubuntu/ear DATA=/home/ubuntu/ear/screen_data_2048nrm.npz FULL_EVAL=0 \
    USE_NRM=1 CFG_NCTRL=$nc CFG_W_CURVE=0.0 FAMILY=phase FOLD=0 SEED=0 EPOCHS=1200 \
    TAG=famF_nc${nc}_f0 python3 -u train_family.py 2>&1 | tail -3
done
echo FAMF2_DONE
