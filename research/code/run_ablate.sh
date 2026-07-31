#!/bin/bash
# Cheap OOF inference ablations: sampler (A), neighbourhood + temperature (B).
cd /home/ubuntu/ear
run () {  # tag clouds gk k temp
  TAG=$1 CLOUDS=/home/ubuntu/ear/$2 GK_OV=$3 K_OV=$4 TEMP=$5 \
    python3 -u gpu_ablate.py 2>&1 | grep -E '^\['
}
run base   all_multisample.npz 20 48 1.0
run area   clouds_area.npz     20 48 1.0
run K32    all_multisample.npz 20 32 1.0
run K64    all_multisample.npz 20 64 1.0
run K96    all_multisample.npz 20 96 1.0
run GK12   all_multisample.npz 12 48 1.0
run GK32   all_multisample.npz 32 48 1.0
run T07    all_multisample.npz 20 48 0.7
run T13    all_multisample.npz 20 48 1.3
run T17    all_multisample.npz 20 48 1.7
echo ABLATE_DONE
