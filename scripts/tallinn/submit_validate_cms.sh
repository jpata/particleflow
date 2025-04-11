#!/bin/bash

# END=`wc -l scripts/cmssw/jetmet0.txt | cut -f1 -d' '`
# for ifile in $(seq 1 $END); do
#     sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job_data.sh mlpf scripts/cmssw/jetmet0.txt JetMET0 $ifile
#     sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job_data.sh pf scripts/cmssw/jetmet0.txt JetMET0 $ifile
# done
#
# END=`wc -l scripts/cmssw/qcd_pu.txt | cut -f1 -d' '`
# for ifile in $(seq 1 $END); do
#     sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job.sh mlpf scripts/cmssw/qcd_pu_local.txt QCD_PU $ifile
#     sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job.sh pf scripts/cmssw/qcd_pu_local.txt QCD_PU $ifile
# done

# END=`wc -l scripts/cmssw/ttbar_pu.txt | cut -f1 -d' '`
# for ifile in $(seq 1 1); do
#     sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job.sh mlpf scripts/cmssw/ttbar_pu_local.txt TTbar_PU $ifile
#     sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job.sh pf scripts/cmssw/ttbar_pu_local.txt TTbar_PU $ifile
# done

# END=`wc -l scripts/cmssw/singleele.txt | cut -f1 -d' '`
# for ifile in $(seq 1 $END); do
#     sbatch scripts/tallinn/cmssw-el8.sh scripts/cmssw/validation_job.sh mlpf scripts/cmssw/singleele.txt SingleEle $ifile
#     sbatch scripts/tallinn/cmssw-el8.sh scripts/cmssw/validation_job.sh pf scripts/cmssw/singleele.txt SingleEle $ifile
# done
#
# END=`wc -l scripts/cmssw/singlegamma.txt | cut -f1 -d' '`
# for ifile in $(seq 1 $END); do
#     sbatch scripts/tallinn/cmssw-el8.sh scripts/cmssw/validation_job.sh mlpf scripts/cmssw/singlegamma.txt SingleGamma $ifile
#     sbatch scripts/tallinn/cmssw-el8.sh scripts/cmssw/validation_job.sh pf scripts/cmssw/singlegamma.txt SingleGamma $ifile
# done
#
# END=`wc -l scripts/cmssw/singlepi.txt | cut -f1 -d' '`
# for ifile in $(seq 1 $END); do
#     sbatch scripts/tallinn/cmssw-el8.sh scripts/cmssw/validation_job.sh mlpf scripts/cmssw/singlepi.txt SinglePi $ifile
#     sbatch scripts/tallinn/cmssw-el8.sh scripts/cmssw/validation_job.sh pf scripts/cmssw/singlepi.txt SinglePi $ifile
# done
#
# END=`wc -l scripts/cmssw/qcd_nopu.txt | cut -f1 -d' '`
# for ifile in $(seq 1 $END); do
#     sbatch scripts/tallinn/cmssw-el8.sh scripts/cmssw/validation_job.sh mlpf scripts/cmssw/qcd_nopu.txt QCD_noPU $ifile
#     sbatch scripts/tallinn/cmssw-el8.sh scripts/cmssw/validation_job.sh pf scripts/cmssw/qcd_nopu.txt QCD_noPU $ifile
# done
#
# END=`wc -l scripts/cmssw/ttbar_nopu.txt | cut -f1 -d' '`
# for ifile in $(seq 1 $END); do
#     sbatch scripts/tallinn/cmssw-el8.sh scripts/cmssw/validation_job.sh mlpf scripts/cmssw/ttbar_nopu.txt TTbar_noPU $ifile
#     sbatch scripts/tallinn/cmssw-el8.sh scripts/cmssw/validation_job.sh pf scripts/cmssw/ttbar_nopu.txt TTbar_noPU $ifile
# done
