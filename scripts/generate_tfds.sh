#!/bin/bash

MANUAL_DIR=/hdfs/local/joosep/mlpf/gen/v2
DATA_DIR=/scratch-persistent/joosep/tensorflow_datasets
export PYTHONPATH=hep_tfds
IMG=/home/software/singularity/tf-2.8.0.simg

#singularity exec -B /hdfs $IMG tfds build hep_tfds/heptfds/cms_pf/singlemu --manual_dir $MANUAL_DIR --overwrite
#singularity exec -B /hdfs $IMG tfds build hep_tfds/heptfds/cms_pf/singleele --manual_dir $MANUAL_DIR --overwrite
#singularity exec -B /hdfs $IMG tfds build hep_tfds/heptfds/cms_pf/singlepi --manual_dir $MANUAL_DIR --overwrite
#singularity exec -B /hdfs $IMG tfds build hep_tfds/heptfds/cms_pf/singlepi0 --manual_dir $MANUAL_DIR --overwrite
#singularity exec -B /hdfs $IMG tfds build hep_tfds/heptfds/cms_pf/singletau --manual_dir $MANUAL_DIR --overwrite
#singularity exec -B /hdfs $IMG tfds build hep_tfds/heptfds/cms_pf/singlegamma --manual_dir $MANUAL_DIR --overwrite
#singularity exec -B /hdfs $IMG tfds build hep_tfds/heptfds/cms_pf/ztt --manual_dir $MANUAL_DIR --overwrite
singularity exec --env PYTHONPATH=$PYTHONPATH -B /hdfs -B /scratch-persistent $IMG tfds build hep_tfds/heptfds/cms_pf/ttbar --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite
singularity exec --env PYTHONPATH=$PYTHONPATH -B /hdfs -B /scratch-persistent $IMG tfds build hep_tfds/heptfds/cms_pf/qcd --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite
singularity exec --env PYTHONPATH=$PYTHONPATH -B /hdfs -B /scratch-persistent $IMG tfds build hep_tfds/heptfds/cms_pf/ztt --data_dir $DATA_DIR --manual_dir $MANUAL_DIR --overwrite
