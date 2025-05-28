#!/bin/bash

USE_CUDA=False
SUBSCRIPT=scripts/tallinn/cmssw-el8.sh

END=`wc -l scripts/tallinn/jetmet0.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job_data.sh ${USE_CUDA} mlpfpu scripts/tallinn/jetmet0.txt JetMET0 $ifile
    # sbatch $SUBSCRIPT scripts/cmssw/validation_job_data.sh ${USE_CUDA} pf scripts/tallinn/jetmet0.txt JetMET0 $ifile
done

END=`wc -l scripts/tallinn/qcd_pu.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh ${USE_CUDA} mlpfpu scripts/tallinn/qcd_pu.txt QCD_PU $ifile
    # sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh ${USE_CUDA} pf scripts/tallinn/qcd_pu.txt QCD_PU $ifile
done

END=`wc -l scripts/tallinn/ttbar_pu.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh ${USE_CUDA} mlpfpu scripts/tallinn/ttbar_pu.txt TTbar_PU $ifile
    # sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh ${USE_CUDA} pf scripts/tallinn/ttbar_pu.txt TTbar_PU $ifile
done

END=`wc -l scripts/tallinn/qcd_nopu.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh ${USE_CUDA} mlpfpu scripts/tallinn/qcd_nopu.txt QCD_noPU $ifile
    # sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh ${USE_CUDA} pf scripts/tallinn/qcd_nopu.txt QCD_noPU $ifile
done


END=`wc -l scripts/tallinn/ttbar_nopu.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh ${USE_CUDA} mlpfpu scripts/tallinn/ttbar_nopu.txt TTbar_noPU $ifile
    # sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh ${USE_CUDA} pf scripts/tallinn/ttbar_nopu.txt TTbar_noPU $ifile
done

# END=`wc -l scripts/tallinn/muon0.txt | cut -f1 -d' '`
# for ifile in $(seq 1 $END); do
#     sbatch $SUBSCRIPT scripts/cmssw/validation_job_data.sh ${USE_CUDA} mlpfpu scripts/tallinn/muon0.txt Muon0 $ifile
#     sbatch $SUBSCRIPT scripts/cmssw/validation_job_data.sh ${USE_CUDA} pf scripts/tallinn/muon0.txt Muon0 $ifile
# done
#
# END=`wc -l scripts/tallinn/zmm.txt | cut -f1 -d' '`
# for ifile in $(seq 1 $END); do
#     sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh ${USE_CUDA} mlpfpu scripts/tallinn/zmm.txt Zmm $ifile
#     sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh ${USE_CUDA} pf scripts/tallinn/zmm.txt Zmm $ifile
# done
