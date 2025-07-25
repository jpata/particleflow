#!/bin/bash

SUBSCRIPT=scripts/tallinn/cmssw-el8.sh

END=`wc -l scripts/tallinn/jetmet0.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job_data.sh False mlpfpu scripts/tallinn/jetmet0.txt JetMET0 $ifile
    sbatch $SUBSCRIPT scripts/cmssw/validation_job_data.sh False pf scripts/tallinn/jetmet0.txt JetMET0 $ifile
done

END=`wc -l scripts/tallinn/photonjet_pu_13p6.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False mlpfpu scripts/tallinn/photonjet_pu_13p6.txt PhotonJet_PU_13p6 $ifile
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False pf scripts/tallinn/photonjet_pu_13p6.txt PhotonJet_PU_13p6 $ifile
done

END=`wc -l scripts/tallinn/photonjet_nopu_13p6.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False mlpfpu scripts/tallinn/photonjet_nopu_13p6.txt PhotonJet_noPU_13p6 $ifile
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False pf scripts/tallinn/photonjet_nopu_13p6.txt PhotonJet_noPU_13p6 $ifile
done

END=`wc -l scripts/tallinn/ttbar_pu_13p6.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False mlpfpu scripts/tallinn/ttbar_pu_13p6.txt TTbar_PU_13p6 $ifile
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False pf scripts/tallinn/ttbar_pu_13p6.txt TTbar_PU_13p6 $ifile
done

END=`wc -l scripts/tallinn/ttbar_nopu_13p6.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False mlpfpu scripts/tallinn/ttbar_nopu_13p6.txt TTbar_noPU_13p6 $ifile
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False pf scripts/tallinn/ttbar_nopu_13p6.txt TTbar_noPU_13p6 $ifile
done

END=`wc -l scripts/tallinn/qcd_pu_13p6.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False mlpfpu scripts/tallinn/qcd_pu_13p6.txt QCD_PU_13p6 $ifile
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False pf scripts/tallinn/qcd_pu_13p6.txt QCD_PU_13p6 $ifile
done

END=`wc -l scripts/tallinn/qcd_nopu_13p6.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False mlpfpu scripts/tallinn/qcd_nopu_13p6.txt QCD_noPU_13p6 $ifile
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False pf scripts/tallinn/qcd_nopu_13p6.txt QCD_noPU_13p6 $ifile
done
