#!/bin/bash
# Family F viability probe: explicit curve + monotone phase, fold 0, full length.
# Two configs differing only in the phase-invariant curve-term weight, since the whole
# design rests on the two heads not fighting: W_CURVE=0 collapses to ordered MSE on a
# curve-constrained output, W_CURVE=0.3 lets geometry improve independently of phase.
cd /home/ubuntu/ear
for wc in 0.0 0.3; do
  echo "=== famF W_CURVE=$wc FOLD=0 ==="
  WORK=/home/ubuntu/ear DATA=/home/ubuntu/ear/screen_data_2048nrm.npz FULL_EVAL=0 \
    USE_NRM=1 CFG_W_CURVE=$wc FAMILY=phase FOLD=0 SEED=0 EPOCHS=1200 \
    TAG=famF_wc${wc}_f0 python3 -u train_family.py 2>&1 | tail -3
done
echo FAMF_PROBE_DONE
