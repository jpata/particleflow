#!/bin/bash

SUBSCRIPT=scripts/tallinn/cmssw-el8.sh
SITE=T2_EE_Estonia

# END=`wc -l scripts/tallinn/jetmet0.txt | cut -f1 -d' '`
# for ifile in $(seq 1 $END); do
#     sbatch $SUBSCRIPT scripts/cmssw/validation_job_data.sh False mlpf scripts/tallinn/jetmet0.txt JetMET0 $ifile $SITE
#     sbatch $SUBSCRIPT scripts/cmssw/validation_job_data.sh False pf scripts/tallinn/jetmet0.txt JetMET0 $ifile $SITE
# done
#
# END=`wc -l scripts/tallinn/photonjet_pu_13p6.txt | cut -f1 -d' '`
# for ifile in $(seq 1 $END); do
#     sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False mlpf scripts/tallinn/photonjet_pu_13p6.txt PhotonJet_PU_13p6 $ifile $SITE
#     sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False pf scripts/tallinn/photonjet_pu_13p6.txt PhotonJet_PU_13p6 $ifile $SITE
# done
#
# END=`wc -l scripts/tallinn/photonjet_nopu_13p6.txt | cut -f1 -d' '`
# for ifile in $(seq 1 $END); do
#     sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False mlpf scripts/tallinn/photonjet_nopu_13p6.txt PhotonJet_noPU_13p6 $ifile $SITE
#     sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False pf scripts/tallinn/photonjet_nopu_13p6.txt PhotonJet_noPU_13p6 $ifile $SITE
# done
#
# END=`wc -l scripts/tallinn/ttbar_pu.txt | cut -f1 -d' '`
# for ifile in $(seq 1 $END); do
#     sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False mlpf scripts/tallinn/ttbar_pu.txt TTbar_PU $ifile $SITE
#     sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False pf scripts/tallinn/ttbar_pu.txt TTbar_PU $ifile $SITE
# done
#
# END=`wc -l scripts/tallinn/ttbar_pu_13p6.txt | cut -f1 -d' '`
# for ifile in $(seq 1 $END); do
#     sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False mlpf scripts/tallinn/ttbar_pu_13p6.txt TTbar_PU_13p6 $ifile $SITE
#     sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False pf scripts/tallinn/ttbar_pu_13p6.txt TTbar_PU_13p6 $ifile $SITE
# done
#
# END=`wc -l scripts/tallinn/ttbar_nopu_13p6.txt | cut -f1 -d' '`
# for ifile in $(seq 1 $END); do
#     sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False mlpf scripts/tallinn/ttbar_nopu_13p6.txt TTbar_noPU_13p6 $ifile $SITE
#     sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False pf scripts/tallinn/ttbar_nopu_13p6.txt TTbar_noPU_13p6 $ifile $SITE
# done

END=`wc -l scripts/tallinn/qcd_pu.txt | cut -f1 -d' '`
for ifile in $(seq 10 $END); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False mlpf scripts/tallinn/qcd_pu.txt QCD_PU $ifile $SITE
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False pf scripts/tallinn/qcd_pu.txt QCD_PU $ifile $SITE
done

END=`wc -l scripts/tallinn/qcd_pu_13p6.txt | cut -f1 -d' '`
for ifile in $(seq 1 1000); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False mlpf scripts/tallinn/qcd_pu_13p6.txt QCD_PU_13p6 $ifile $SITE
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False pf scripts/tallinn/qcd_pu_13p6.txt QCD_PU_13p6 $ifile $SITE
done

# END=`wc -l scripts/tallinn/qcd_nopu_13p6.txt | cut -f1 -d' '`
# for ifile in $(seq 1 $END); do
#     sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False mlpf scripts/tallinn/qcd_nopu_13p6.txt QCD_noPU_13p6 $ifile $SITE
#     sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False pf scripts/tallinn/qcd_nopu_13p6.txt QCD_noPU_13p6 $ifile $SITE
# done

END=`wc -l scripts/tallinn/qcd_pu_13p6_v3.txt | cut -f1 -d' '`
for ifile in $(seq 1 1000); do
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False mlpf scripts/tallinn/qcd_pu_13p6_v3.txt QCD_PU_13p6_v3 $ifile $SITE
    sbatch $SUBSCRIPT scripts/cmssw/validation_job.sh False pf scripts/tallinn/qcd_pu_13p6_v3.txt QCD_PU_13p6_v3 $ifile $SITE
done
