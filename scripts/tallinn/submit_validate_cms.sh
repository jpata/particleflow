#!/bin/bash

END=`wc -l scripts/cmssw/qcd_pu.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch scripts/tallinn/cmssw-el8.sh scripts/cmssw/validation_job.sh mlpf scripts/cmssw/qcd_pu.txt QCD_PU $ifile
    sbatch scripts/tallinn/cmssw-el8.sh scripts/cmssw/validation_job.sh pf scripts/cmssw/qcd_pu.txt QCD_PU $ifile
done

END=`wc -l scripts/cmssw/ttbar_pu.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch scripts/tallinn/cmssw-el8.sh scripts/cmssw/validation_job.sh mlpf scripts/cmssw/ttbar_pu.txt TTbar_PU $ifile
    sbatch scripts/tallinn/cmssw-el8.sh scripts/cmssw/validation_job.sh pf scripts/cmssw/ttbar_pu.txt TTbar_PU $ifile
done
