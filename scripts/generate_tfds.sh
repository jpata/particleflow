#!/bin/bash

export KERAS_BACKEND=tensorflow
export PYTHONPATH="mlpf:$PYTHONPATH"

# T2_EE_Estonia
export IMG=/home/software/singularity/pytorch.simg:2024-08-18
export CMD="singularity exec -B /local -B /scratch/persistent $IMG tfds build"

# Desktop
# export MANUAL_DIR=/media/joosep/data/cms/v3_1/
# export DATA_DIR=/home/joosep/tensorflow_datasets
# export IMG=/home/joosep/HEP-KBFI/singularity/pytorch.simg
# export CMD="singularity exec -B /media/joosep/data --env PYTHONPATH=$PYTHONPATH $IMG tfds build "

# CMS
# export DATA_DIR=/scratch/persistent/joosep/tensorflow_datasets2
# export MANUAL_DIR=/local/joosep/mlpf/cms/20240823_simcluster
# for i in `seq 1 10`; do
#     $CMD mlpf/heptfds/cms_pf/ttbar --config $i --data_dir $DATA_DIR --manual_dir $MANUAL_DIR/pu55to75 --overwrite &> logs/tfds_ttbar.log$i &
#     $CMD mlpf/heptfds/cms_pf/qcd --config $i --data_dir $DATA_DIR --manual_dir $MANUAL_DIR/pu55to75 --overwrite &> logs/tfds_qcd.log$i &
#     $CMD mlpf/heptfds/cms_pf/ztt --config $i --data_dir $DATA_DIR --manual_dir $MANUAL_DIR/pu55to75 --overwrite &> logs/tfds_ztt.log$i &
# done
# $CMD mlpf/heptfds/cms_pf/singleele --data_dir $DATA_DIR --manual_dir $MANUAL_DIR/nopu --overwrite --config 1 &> logs/tfds_singleele.log &
## run 13p6 TeV validation sample
# for i in `seq 1 10`; do
#   $CMD mlpf/heptfds/cms_pf/qcd_13p6 --config $i --data_dir $DATA_DIR --manual_dir /local/joosep/mlpf/cms/20250630_cmssw_15_0_5_f8ae2f/pu55to75_val --overwrite &> logs/tfds_qcd.log$i &
# done
# wait

# CLIC cluster-based
# export DATA_DIR=/scratch/persistent/joosep/tensorflow_datasets
# export MANUAL_DIR=/local/joosep/mlpf/clic_edm4hep/
# $CMD mlpf/heptfds/clic_pf_edm4hep/qq --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_qq.log &
# $CMD mlpf/heptfds/clic_pf_edm4hep/ttbar --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_ttbar.log &
# $CMD mlpf/heptfds/clic_pf_edm4hep/ww_fullhad --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_ww_fullhad.log &
# $CMD mlpf/heptfds/clic_pf_edm4hep/zh --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_zh.log &
# $CMD mlpf/heptfds/clic_pf_edm4hep/z --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_z.log &
# $CMD mlpf/heptfds/clic_pf_edm4hep/ttbar_pu10 --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_ttbar_pu10.log &
# wait

# CLD cluster-based
# export DATA_DIR=/scratch/persistent/joosep/tensorflow_datasets
# export MANUAL_DIR=/local/joosep/mlpf/cld_edm4hep/2024_05_full
# $CMD mlpf/heptfds/cld_pf_edm4hep/ttbar --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_ttbar.log &
# wait

# CLIC hit-based
# export MANUAL_DIR=/local/joosep/mlpf/clic_edm4hep_hits/
# export DATA_DIR=/local/joosep/mlpf/tensorflow_datasets/clic/hits/
# $CMD mlpf/heptfds/clic_pf_edm4hep_hits/qq --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_qq_hits.log &
# $CMD mlpf/heptfds/clic_pf_edm4hep_hits/ttbar --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_ttbar_hits.log &
# $CMD mlpf/heptfds/clic_pf_edm4hep_hits/qq_10k --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_qq_hits_10k.log &
# $CMD mlpf/heptfds/clic_pf_edm4hep_hits/ttbar_10k --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_ttbar_hits_10k.log &
# $CMD mlpf/heptfds/clic_pf_edm4hep_hits/single_kaon0L --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_single_kaon0L_hits.log &
# $CMD mlpf/heptfds/clic_pf_edm4hep_hits/single_ele --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_single_ele_hits.log &
# $CMD mlpf/heptfds/clic_pf_edm4hep_hits/single_pi0 --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_single_pi0_hits.log &
# $CMD mlpf/heptfds/clic_pf_edm4hep_hits/single_pi --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_single_pi_hits.log &
# $CMD mlpf/heptfds/clic_pf_edm4hep_hits/single_neutron --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_single_neutron_hits.log &
# $CMD mlpf/heptfds/clic_pf_edm4hep_hits/single_gamma --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_single_gamma_hits.log &
# $CMD mlpf/heptfds/clic_pf_edm4hep_hits/single_mu --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_single_mu_hits.log &
# wait
