#!/bin/bash

SUBSCRIPT=scripts/tallinn/cmssw-el8.sh

END=`wc -l scripts/tallinn/jetmet0.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job_data.sh False mlpfpu scripts/tallinn/jetmet0.txt JetMET0 $ifile
    sbatch $SUBSCRIPT scripts/cmssw/validation_job_data.sh False pf scripts/tallinn/jetmet0.txt JetMET0 $ifile
done

END=`wc -l scripts/tallinn/qcd_pu.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False mlpfpu scripts/tallinn/qcd_pu.txt QCD_PU $ifile
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False pf scripts/tallinn/qcd_pu.txt QCD_PU $ifile
done

END=`wc -l scripts/tallinn/ttbar_pu.txt | cut -f1 -d' '`
for ifile in $(seq 1 6); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False mlpfpu scripts/tallinn/ttbar_pu.txt TTbar_PU $ifile
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False pf scripts/tallinn/ttbar_pu.txt TTbar_PU $ifile
done

END=`wc -l scripts/tallinn/qcd_nopu.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False mlpfpu scripts/tallinn/qcd_nopu.txt QCD_noPU $ifile
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False pf scripts/tallinn/qcd_nopu.txt QCD_noPU $ifile
done

END=`wc -l scripts/tallinn/ttbar_nopu.txt | cut -f1 -d' '`
for ifile in $(seq 1 1); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False mlpfpu scripts/tallinn/ttbar_nopu.txt TTbar_noPU_cpu $ifile
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False pf scripts/tallinn/ttbar_nopu.txt TTbar_noPU $ifile
done

END=`wc -l scripts/tallinn/zmm_nopu.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False mlpfpu scripts/tallinn/zmm_nopu.txt Zmm_nopu $ifile
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False pf scripts/tallinn/zmm_nopu.txt Zmm_nopu $ifile
done

END=`wc -l scripts/tallinn/zee_nopu.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False mlpfpu scripts/tallinn/zee_nopu.txt Zee_nopu $ifile
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False pf scripts/tallinn/zee_nopu.txt Zee_nopu $ifile
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
for ifile in $(seq 2050 $END); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False mlpfpu scripts/tallinn/qcd_pu_13p6.txt QCD_PU_13p6 $ifile
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False pf scripts/tallinn/qcd_pu_13p6.txt QCD_PU_13p6 $ifile
done

END=`wc -l scripts/tallinn/qcd_nopu_13p6.txt | cut -f1 -d' '`
for ifile in $(seq 1 $END); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False mlpfpu scripts/tallinn/qcd_nopu_13p6.txt QCD_noPU_13p6 $ifile
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False pf scripts/tallinn/qcd_nopu_13p6.txt QCD_noPU_13p6 $ifile
done
