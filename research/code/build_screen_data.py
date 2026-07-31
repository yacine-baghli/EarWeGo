"""
Screening data for gpu_screen.py.

screen_data_2048.npz is a straight repack of all_multisample.npz (the M=4 independent
surface samples already used by the shipped fresh-sample TTA), so the `base` screening
variant reproduces the trained baseline exactly -- one change per run means the DATA
must not change either.

    python research/code/build_screen_data.py

Fields: clouds (E,M,N,3) coarse (E,85,3) true (E,85,3) R (E,3,3) c0 (E,3) split (E,)
Coordinates are in the per-ear canonical frame; nothing here is published.
"""
import numpy as np
d = np.load("scratch/all_multisample.npz")
np.savez("scratch/screen_data_2048.npz", clouds=d["clouds"], coarse=d["coarse"],
         true=d["true"], R=d["R"], c0=d["c0"], split=d["split"])
print("wrote scratch/screen_data_2048.npz", d["clouds"].shape)
