#!/bin/bash

# END=`wc -l scripts/tallinn/jetmet0.txt | cut -f1 -d' '`
# for ifile in $(seq 1 $END); do
#     sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job_data.sh mlpfpu scripts/tallinn/jetmet0.txt JetMET0 $ifile
#     sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job_data.sh pf scripts/tallinn/jetmet0.txt JetMET0 $ifile
# done
#
# END=`wc -l scripts/tallinn/qcd_pu.txt | cut -f1 -d' '`
# for ifile in $(seq 1 $END); do
#     sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job.sh mlpfpu scripts/tallinn/qcd_pu.txt QCD_PU $ifile
#     # sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job.sh mlpf scripts/tallinn/qcd_pu.txt QCD_PU $ifile
#     sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job.sh pf scripts/tallinn/qcd_pu.txt QCD_PU $ifile
# done
#
# END=`wc -l scripts/tallinn/ttbar_pu.txt | cut -f1 -d' '`
# for ifile in $(seq 1 $END); do
#     sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job.sh mlpfpu scripts/tallinn/ttbar_pu.txt TTbar_PU_mlpfpu $ifile
#     # sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job.sh mlpf scripts/tallinn/ttbar_pu.txt TTbar_PU $ifile
#     sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job.sh pf scripts/tallinn/ttbar_pu.txt TTbar_PU $ifile
# done
#
# END=`wc -l scripts/tallinn/qcd_nopu.txt | cut -f1 -d' '`
# for ifile in $(seq 1 $END); do
#     sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job.sh mlpfpu scripts/tallinn/qcd_nopu.txt QCD_noPU $ifile
#     sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job.sh pf scripts/tallinn/qcd_nopu.txt QCD_noPU $ifile
# done
#
#
# END=`wc -l scripts/tallinn/ttbar_nopu.txt | cut -f1 -d' '`
# for ifile in $(seq 1 $END); do
#     sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job.sh mlpfpu scripts/tallinn/ttbar_nopu.txt TTbar_noPU $ifile
#     sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job.sh pf scripts/tallinn/ttbar_nopu.txt TTbar_noPU $ifile
# done

END=`wc -l scripts/tallinn/muon0.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job_data.sh mlpfpu scripts/tallinn/muon0.txt Muon0 $ifile
    sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job_data.sh pf scripts/tallinn/muon0.txt Muon0 $ifile
done

END=`wc -l scripts/tallinn/zmm.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job_data.sh mlpfpu scripts/tallinn/zmm.txt Zmm $ifile
    sbatch scripts/tallinn/cmssw-el8-gpu.sh scripts/cmssw/validation_job_data.sh pf scripts/tallinn/zmm.txt Zmm $ifile
done
