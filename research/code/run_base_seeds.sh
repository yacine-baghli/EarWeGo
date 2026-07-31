#!/bin/bash
# Reference baseline under the upgraded protocol: 5 folds x 3 seeds.
# base s0 f0-f4 and base s1 f0 already exist, so fill in the remaining 9 runs.
cd /home/ubuntu/ear
for spec in "1 1" "1 2" "1 3" "1 4" "2 0" "2 1" "2 2" "2 3" "2 4"; do
  set -- $spec
  echo "=== base SEED=$1 FOLD=$2 ==="
  VARIANT=base SEED=$1 FOLD=$2 EPOCHS=1200 python3 -u gpu_screen.py 2>&1 | tail -2
done
echo BASE_SEEDS_DONE
