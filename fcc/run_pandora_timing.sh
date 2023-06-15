#!/bin/bash

for iseed in 1 2 3; do
    for nptcl in 64 128 256 512; do
        SLURM_JOB_ID=$iseed ./run_sim_gun_np.sh $iseed pi- $nptcl &> gun_np_${nptcl}_${iseed}.txt &
    done
    wait
done
