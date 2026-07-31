#!/bin/bash
# Corrected-normals re-run under the upgraded protocol: 5 folds x 3 seeds.
# The previous normals result is void -- it trained on inward right-ear normals.
# This also sanity-checks the corrected normals input that families A/B/C all consume.
cd /home/ubuntu/ear
while pgrep -f "[r]un_base_seeds.sh" > /dev/null; do sleep 60; done   # queue behind the baseline
echo "QUEUE released: baseline finished, starting corrected normals"
for s in 0 1 2; do
  for f in 0 1 2 3 4; do
    echo "=== normalsfix SEED=$s FOLD=$f ==="
    DATA=/home/ubuntu/ear/screen_data_2048nrm.npz VARIANT=normals OUTTAG=normalsfix \
      SEED=$s FOLD=$f EPOCHS=1200 python3 -u gpu_screen.py 2>&1 | tail -2
  done
done
echo NORMALSFIX_DONE
