#!/bin/bash

SLURM_JOB_ID=1 ./run_sim_gun_np.sh 1 pi- 100 &> gun_np_100_1.txt
#for iseed in 6 7; do
#    for nptcl in 25 50 100 200; do
#        SLURM_JOB_ID=$iseed ./run_sim_gun_np.sh $iseed pi- $nptcl &> gun_np_${nptcl}_${iseed}.txt
#    done
#done
