#!/bin/bash

USE_CUDA=False
SUBSCRIPT=scripts/tallinn/cmssw-el8-gpu.sh

END=`wc -l scripts/tallinn/ttbar_pu.txt | cut -f1 -d' '`
for ifile in $(seq 1 6); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh True mlpfpu scripts/tallinn/ttbar_pu.txt TTbar_PU $ifile
    # sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False mlpfpu scripts/tallinn/ttbar_pu.txt TTbar_PU $ifile
    # sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False pf scripts/tallinn/ttbar_pu.txt TTbar_PU $ifile
done
