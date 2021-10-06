#!/bin/bash

MANUAL_DIR=/hdfs/local/joosep/mlpf/gen/
export PYTHONPATH=hep_tfds
IMG=/home/software/singularity/tf26.simg:latest

singularity exec -B /hdfs $IMG tfds build hep_tfds/heptfds/cms_pf/singlemu --manual_dir $MANUAL_DIR --overwrite
singularity exec -B /hdfs $IMG tfds build hep_tfds/heptfds/cms_pf/singleele --manual_dir $MANUAL_DIR --overwrite
singularity exec -B /hdfs $IMG tfds build hep_tfds/heptfds/cms_pf/singlepi --manual_dir $MANUAL_DIR --overwrite
singularity exec -B /hdfs $IMG tfds build hep_tfds/heptfds/cms_pf/singlepi0 --manual_dir $MANUAL_DIR --overwrite
singularity exec -B /hdfs $IMG tfds build hep_tfds/heptfds/cms_pf/singletau --manual_dir $MANUAL_DIR --overwrite
singularity exec -B /hdfs $IMG tfds build hep_tfds/heptfds/cms_pf/singlegamma --manual_dir $MANUAL_DIR --overwrite
singularity exec -B /hdfs $IMG tfds build hep_tfds/heptfds/cms_pf/ztt --manual_dir $MANUAL_DIR --overwrite
singularity exec -B /hdfs $IMG tfds build hep_tfds/heptfds/cms_pf/ttbar --manual_dir $MANUAL_DIR --overwrite
