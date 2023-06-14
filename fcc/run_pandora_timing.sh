#!/bin/bash

for nptcl in 10 20 30 40 50 60 70 80 90 100 150 200 250 300 500 1000; do
    SLURM_JOB_ID=1 ./run_sim_gun_np.sh 1 pi- $nptcl &> gun_np_${nptcl}_1.txt
    SLURM_JOB_ID=1 ./run_sim_gun_np.sh 2 pi- $nptcl &> gun_np_${nptcl}_2.txt
    SLURM_JOB_ID=1 ./run_sim_gun_np.sh 3 pi- $nptcl &> gun_np_${nptcl}_3.txt
done
