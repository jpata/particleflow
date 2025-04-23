#!/bin/bash

END=`wc -l scripts/cmssw/jetmet0.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job_data.sh mlpf scripts/tallinn/jetmet0.txt JetMET0 $ifile
    sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job_data.sh pf scripts/tallinn/jetmet0.txt JetMET0 $ifile
done

END=`wc -l scripts/cmssw/qcd_pu.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job.sh mlpf scripts/tallinn/qcd_pu.txt QCD_PU $ifile
    sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job.sh pf scripts/tallinn/qcd_pu.txt QCD_PU $ifile
done

END=`wc -l scripts/cmssw/ttbar_pu.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job.sh mlpf scripts/tallinn/ttbar_pu.txt TTbar_PU $ifile
    sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job.sh pf scripts/tallinn/ttbar_pu.txt TTbar_PU $ifile
done

END=`wc -l scripts/cmssw/qcd_nopu.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job.sh mlpf scripts/tallinn/qcd_nopu.txt QCD_noPU $ifile
    sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job.sh pf scripts/tallinn/qcd_nopu.txt QCD_noPU $ifile
done

END=`wc -l scripts/cmssw/ttbar_nopu.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job.sh mlpf scripts/tallinn/ttbar_nopu.txt TTbar_noPU $ifile
    sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job.sh pf scripts/tallinn/ttbar_nopu.txt TTbar_noPU $ifile
done
