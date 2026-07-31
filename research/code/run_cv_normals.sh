#!/bin/bash
# Adjudicate `normals` against `base` over ALL folds. Paired: the same fold split feeds
# both models, so every one of the 340 ears is compared model-to-model on the same
# held-out partition. Fold 0 is already done for both.
cd /home/ubuntu/ear
for f in 1 2 3 4; do
  echo "=== base FOLD=$f ==="
  VARIANT=base SEED=0 FOLD=$f EPOCHS=1200 python3 -u gpu_screen.py 2>&1 | tail -2
  echo "=== normals FOLD=$f ==="
  DATA=/home/ubuntu/ear/screen_data_2048nrm.npz VARIANT=normals SEED=0 FOLD=$f \
    EPOCHS=1200 python3 -u gpu_screen.py 2>&1 | tail -2
done
echo CV_DONE
