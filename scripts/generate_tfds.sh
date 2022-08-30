#!/bin/bash

MANUAL_DIR=/hdfs/local/joosep/mlpf/gen/v2
DATA_DIR=/scratch-persistent/joosep/tensorflow_datasets
export PYTHONPATH=hep_tfds
IMG=/home/software/singularity/tf-2.9.0.simg

singularity exec --env PYTHONPATH=$PYTHONPATH -B /hdfs -B /scratch-persistent $IMG tfds build hep_tfds/heptfds/cms_pf/ttbar --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/ttbar.log &
singularity exec --env PYTHONPATH=$PYTHONPATH -B /hdfs -B /scratch-persistent $IMG tfds build hep_tfds/heptfds/cms_pf/qcd --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/qcd.log &
singularity exec --env PYTHONPATH=$PYTHONPATH -B /hdfs -B /scratch-persistent $IMG tfds build hep_tfds/heptfds/cms_pf/ztt --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/ztt.log &
singularity exec --env PYTHONPATH=$PYTHONPATH -B /hdfs -B /scratch-persistent $IMG tfds build hep_tfds/heptfds/cms_pf/qcd_high_pt --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite &> logs/qcd_high_pt.log &
wait
