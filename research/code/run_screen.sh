#!/bin/bash
# Screening on ONE representative fold (0). One change per run.
# base x2 seeds FIRST: without training noise nothing else is interpretable.
cd /home/ubuntu/ear
for spec in "base 0" "base 1" "untied4 0" "untied6 0" "fusion2 0"; do
  set -- $spec
  echo "=== VARIANT=$1 SEED=$2 ==="
  VARIANT=$1 SEED=$2 FOLD=0 EPOCHS=1200 python3 -u gpu_screen.py 2>&1 | tail -3
done
echo SCREEN_DONE
