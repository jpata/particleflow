#!/bin/bash

# Tallinn
export MANUAL_DIR=/local/joosep/mlpf/cms/v2
export DATA_DIR=/scratch-persistent/joosep/tensorflow_datasets
export IMG=/home/software/singularity/tf-2.11.0.simg
export PYTHONPATH=`pwd`/mlpf
export CMD="singularity exec -B /local -B /scratch-persistent --env PYTHONPATH=$PYTHONPATH $IMG tfds build "

# Desktop
# IMG=/home/joosep/HEP-KBFI/singularity/tf-2.10.0.simg
# MANUAL_DIR=data/
# DATA_DIR=/home/joosep/tensorflow_datasets
# export PYTHONPATH="mlpf:$PYTHONPATH"
# CMD="singularity exec --env PYTHONPATH=$PYTHONPATH $IMG tfds build "

# CMS
$CMD mlpf/heptfds/cms_pf/ttbar --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_ttbar.log &
$CMD mlpf/heptfds/cms_pf/qcd --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_qcd.log &
$CMD mlpf/heptfds/cms_pf/ztt --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_ztt.log &
$CMD mlpf/heptfds/cms_pf/qcd_high_pt --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_qcd_high_pt.log &
$CMD mlpf/heptfds/cms_pf/singlepi0 --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_singlepi0.log &
$CMD mlpf/heptfds/cms_pf/singleneutron --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_singleneutron.log &
$CMD mlpf/heptfds/cms_pf/singleele --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_singleele.log &
$CMD mlpf/heptfds/cms_pf/singlegamma --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_singlegamma.log &
$CMD mlpf/heptfds/cms_pf/singlemu --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_singlemu.log &
$CMD mlpf/heptfds/cms_pf/singlepi --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_singlepi.log &
$CMD mlpf/heptfds/cms_pf/singleproton --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_singleproton.log &
$CMD mlpf/heptfds/cms_pf/singletau --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_singletau.log &
wait

# CLIC
export MANUAL_DIR=/local/joosep/mlpf/clic_edm4hep_2023_02_21
$CMD mlpf/heptfds/clic_pf_edm4hep/qq --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_qq.log &
$CMD mlpf/heptfds/clic_pf_edm4hep/ttbar --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_ttbar.log &
wait

# Delphes
$CMD mlpf/heptfds/delphes_pf/delphes_pf &> logs/tfds_delphes.log &
wait
