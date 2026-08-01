#!/bin/bash
# Profile-decoder viability probe, fold 0, full length. One control and three arms, each
# differing from the control in ONE respect, because the control IS the shipped baseline:
# with PROFILE_CONTOURS="" fam_profile.py is the 813,232-param base model parameter for
# parameter, so `base` here is the reference the arms are read against on the same fold,
# the same seed and the same data -- not against a number from another run.
#   base      PROFILE_CONTOURS=none    free XYZ everywhere (expect ~1.2652 at USE_NRM=1)
#             ("none", not "": train_family._autotype json.loads an empty CFG_* value)
#   fixed     PROFILE_CONTOURS=2,3     inner_helix + sup._antihelix placed at the
#                                      TRAINING-FOLD mean profile. Zero extra parameters.
#   learned   + PROFILE_MODE=learned   a per-ear deviation bounded by the measured sd
#   all4      PROFILE_CONTOURS=0,1,2,3 the NEGATIVE control. profile_apply.py measures
#                                      +0.2320mm for this post hoc; if it comes out fine
#                                      here, the placement is not doing what it claims.
# ARTEFACTS carries the per-fold profile from research/code/build_profile.py; without it
# the module falls back to the uniform profile and says so in its first line of output.
cd /home/ubuntu/ear
COMMON="WORK=/home/ubuntu/ear DATA=/home/ubuntu/ear/screen_data_2048nrm.npz FULL_EVAL=0 \
USE_NRM=1 ARTEFACTS=/home/ubuntu/ear/profile_f0.npz FAMILY=profile FOLD=0 SEED=0 EPOCHS=1200"
for arm in 'base:CFG_PROFILE_CONTOURS=none' 'fixed:CFG_PROFILE_CONTOURS=2,3' \
           'learned:CFG_PROFILE_CONTOURS=2,3 CFG_PROFILE_MODE=learned' \
           'all4:CFG_PROFILE_CONTOURS=0,1,2,3'; do
  nm=${arm%%:*}; env_=${arm#*:}
  echo "=== profile $nm FOLD=0 ==="
  env $COMMON $env_ TAG=famG_${nm}_f0 python3 -u train_family.py 2>&1 | tail -4
done
echo PROFILE_PROBE_DONE
