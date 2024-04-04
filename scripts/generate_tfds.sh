#!/bin/bash

# Tallinn
export MANUAL_DIR=/local/joosep/mlpf/cms/v3
export DATA_DIR=/local/joosep/mlpf/cms/v3/tensorflow_datasets
export IMG=/home/software/singularity/pytorch.simg:2024-03-11
export PYTHONPATH=mlpf
export KERAS_BACKEND=tensorflow
export CMD="singularity exec -B /local -B /scratch/persistent $IMG tfds build "

# Desktop
# IMG=/home/joosep/HEP-KBFI/singularity/tf-2.13.0.simg
# DATA_DIR=/home/joosep/tensorflow_datasets
# export PYTHONPATH="mlpf:$PYTHONPATH"
# CMD="singularity exec -B /media/joosep/data --env PYTHONPATH=$PYTHONPATH $IMG tfds build "

# CMS
# export DATA_DIR=/scratch/persistent/joosep/tensorflow_datasets
# $CMD mlpf/heptfds/cms_pf/ttbar --data_dir $DATA_DIR --manual_dir $MANUAL_DIR/pu55to75 --overwrite &> logs/tfds_ttbar.log &
# $CMD mlpf/heptfds/cms_pf/qcd --data_dir $DATA_DIR --manual_dir $MANUAL_DIR/pu55to75 --overwrite &> logs/tfds_qcd.log &
# $CMD mlpf/heptfds/cms_pf/ztt --data_dir $DATA_DIR --manual_dir $MANUAL_DIR/pu55to75 --overwrite &> logs/tfds_ztt.log &
# $CMD mlpf/heptfds/cms_pf/qcd_high_pt --data_dir $DATA_DIR --manual_dir $MANUAL_DIR/pu55to75 --overwrite &> logs/tfds_qcd_high_pt.log &
# $CMD mlpf/heptfds/cms_pf/smst1tttt --data_dir $DATA_DIR --manual_dir $MANUAL_DIR/pu55to75 --overwrite &> logs/tfds_smst1tttt.log &
# $CMD mlpf/heptfds/cms_pf/vbf --data_dir $DATA_DIR --manual_dir $MANUAL_DIR/pu55to75 --overwrite &> logs/tfds_vbf.log &
# $CMD mlpf/heptfds/cms_pf/singlepi0 --data_dir $DATA_DIR --manual_dir $MANUAL_DIR/nopu --overwrite &> logs/tfds_singlepi0.log &
# $CMD mlpf/heptfds/cms_pf/singleneutron --data_dir $DATA_DIR --manual_dir $MANUAL_DIR/nopu --overwrite &> logs/tfds_singleneutron.log &
# $CMD mlpf/heptfds/cms_pf/singleele --data_dir $DATA_DIR --manual_dir $MANUAL_DIR/nopu --overwrite &> logs/tfds_singleele.log &
# $CMD mlpf/heptfds/cms_pf/singlegamma --data_dir $DATA_DIR --manual_dir $MANUAL_DIR/nopu --overwrite &> logs/tfds_singlegamma.log &
# $CMD mlpf/heptfds/cms_pf/singlemu --data_dir $DATA_DIR --manual_dir $MANUAL_DIR/nopu --overwrite &> logs/tfds_singlemu.log &
# $CMD mlpf/heptfds/cms_pf/singlepi --data_dir $DATA_DIR --manual_dir $MANUAL_DIR/nopu --overwrite &> logs/tfds_singlepi.log &
# $CMD mlpf/heptfds/cms_pf/singleproton --data_dir $DATA_DIR --manual_dir $MANUAL_DIR/nopu --overwrite &> logs/tfds_singleproton.log &
# $CMD mlpf/heptfds/cms_pf/singletau --data_dir $DATA_DIR --manual_dir $MANUAL_DIR/nopu --overwrite &> logs/tfds_singletau.log &
# $CMD mlpf/heptfds/cms_pf/multiparticlegun --data_dir $DATA_DIR --manual_dir $MANUAL_DIR/nopu --overwrite &> logs/tfds_multiparticlegun.log &
# wait

# CLIC cluster-based
# export MANUAL_DIR=/local/joosep/mlpf/clic_edm4hep/
# export MANUAL_DIR=/media/joosep/data/mlpf/clic_edm4hep_2023_02_27/
# $CMD mlpf/heptfds/clic_pf_edm4hep/qq --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_qq.log &
# $CMD mlpf/heptfds/clic_pf_edm4hep/ttbar --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_ttbar.log &
# $CMD mlpf/heptfds/clic_pf_edm4hep/zh --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_zh.log &
# $CMD mlpf/heptfds/clic_pf_edm4hep/ttbar_pu10 --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_ttbar_pu10.log &
# $CMD mlpf/heptfds/clic_pf_edm4hep/ww_fullhad --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_ww_fullhad.log &
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

# Delphes
# export MANUAL_DIR=/local/joosep/mlpf/delphes/
# $CMD mlpf/heptfds/delphes_pf/delphes_ttbar_pf --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_delphes_ttbar.log &
# $CMD mlpf/heptfds/delphes_pf/delphes_qcd_pf --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_delphes_qcd.log &
# wait
