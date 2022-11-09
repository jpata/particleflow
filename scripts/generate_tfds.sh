#!/bin/bash

MANUAL_DIR=/local/joosep/mlpf/gen/v2
DATA_DIR=/scratch-persistent/joosep/tensorflow_datasets
IMG=/home/software/singularity/tf-2.10.0.simg

singularity exec -B /local -B /scratch-persistent $IMG tfds build mlpf/heptfds/cms_pf/ttbar --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_ttbar.log &
singularity exec -B /local -B /scratch-persistent $IMG tfds build mlpf/heptfds/cms_pf/qcd --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_qcd.log &
singularity exec -B /local -B /scratch-persistent $IMG tfds build mlpf/heptfds/cms_pf/ztt --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_ztt.log &
singularity exec -B /local -B /scratch-persistent $IMG tfds build mlpf/heptfds/cms_pf/qcd_high_pt --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_qcd_high_pt.log &
singularity exec -B /local -B /scratch-persistent $IMG tfds build mlpf/heptfds/cms_pf/singlepi0 --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_singlepi0.log &
singularity exec -B /local -B /scratch-persistent $IMG tfds build mlpf/heptfds/cms_pf/singleneutron --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_singleneutron.log &
singularity exec -B /local -B /scratch-persistent $IMG tfds build mlpf/heptfds/cms_pf/singleele --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_singleele.log &
singularity exec -B /local -B /scratch-persistent $IMG tfds build mlpf/heptfds/cms_pf/singlegamma --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/tfds_singlegamma.log &
wait
