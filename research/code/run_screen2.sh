#!/bin/bash
cd /home/ubuntu/ear
echo "=== VARIANT=untied6 SEED=0 ==="
VARIANT=untied6 SEED=0 FOLD=0 EPOCHS=1200 python3 -u gpu_screen.py 2>&1 | tail -3
echo "=== VARIANT=pts4096 SEED=0 ==="
VARIANT=pts4096 SEED=0 FOLD=0 EPOCHS=1200 python3 -u gpu_screen.py 2>&1 | tail -3
echo "=== VARIANT=normals SEED=0 ==="
DATA=/home/ubuntu/ear/screen_data_2048nrm.npz VARIANT=normals SEED=0 FOLD=0 EPOCHS=1200 python3 -u gpu_screen.py 2>&1 | tail -3
echo "=== VARIANT=chamfer SEED=0 ==="
VARIANT=chamfer SEED=0 FOLD=0 EPOCHS=1200 python3 -u gpu_screen.py 2>&1 | tail -3
echo SCREEN2_DONE
