#!/bin/bash
set -e
export TFDS_DATA_DIR=`pwd`/tensorflow_datasets
export PWD=`pwd`
export PYTHONPATH=`pwd`
export KERAS_BACKEND=torch

# Quick unit tests
python -m pytest tests/test_dataloader.py
python -m pytest tests/test_dataloader_behavior.py
python -m pytest tests/test_endless_interleaved_iterator.py
python -m pytest tests/test_resumable_sampler.py
python -m pytest tests/test_interleaved_iterator.py
python -m pytest tests/test_lr_schedule.py

#create data directories
rm -Rf local_test_data/TTbar_13p6TeV_TuneCUETP8M1_cfi
mkdir -p local_test_data/TTbar_13p6TeV_TuneCUETP8M1_cfi/root
cd local_test_data/TTbar_13p6TeV_TuneCUETP8M1_cfi/root

#Only CMS-internal use is permitted by CMS rules! Do not use these MC simulation files otherwise!
wget --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/20240823_simcluster/pu55to75/TTbar_14TeV_TuneCUETP8M1_cfi/root/pfntuple_100000.root
wget --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/20240823_simcluster/pu55to75/TTbar_14TeV_TuneCUETP8M1_cfi/root/pfntuple_100001.root

cd ../../..

#Create the ntuples using postprocessing2.py
for file in `\ls -1 local_test_data/TTbar_13p6TeV_TuneCUETP8M1_cfi/root/*.root`; do
  python mlpf/data/cms/postprocessing2.py \
    --input $file \
    --outpath local_test_data/TTbar_13p6TeV_TuneCUETP8M1_cfi
done
find local_test_data

#create the tensorflow dataset for the last split config only
tfds build mlpf/heptfds/cms_pf/ttbar --config 10 --manual_dir ./local_test_data

mkdir -p experiments

# --------------------------------------------------------------------------------------------
# Test 1: Initial training using the 'train' sub-command
# --------------------------------------------------------------------------------------------
python mlpf/pipeline.py \
  --config parameters/pytorch/pyg-cms.yaml \
  --data-dir ./tensorflow_datasets/ \
  --prefix MLPF_test_ \
  --pipeline \
  train \
  --num-steps 2 \
  --checkpoint-freq 2 \
  --gpus 0 \
  --make-plots \
  --conv-type attention \
  --dtype float32 \
  --attention-type math \
  --num-convs 1

ls experiments/MLPF_test_*/checkpoints/*

# Capture the experiment directory created by the first run for the next steps
export EXP_DIR=$(ls -d experiments/MLPF_test_*/)

# --------------------------------------------------------------------------------------------
# Test 2: Fine-tuning from a checkpoint in a NEW directory
# --experiment-dir is omitted, so a new one is created.
# --------------------------------------------------------------------------------------------
python mlpf/pipeline.py \
  --config parameters/pytorch/pyg-cms.yaml \
  --data-dir ./tensorflow_datasets/ \
  --prefix MLPF_test_ \
  --pipeline \
  train \
  --num-steps 4 \
  --checkpoint-freq 2 \
  --gpus 0 \
  --make-plots \
  --conv-type attention \
  --dtype float32 \
  --attention-type math \
  --num-convs 1 \
  --load ${EXP_DIR}/checkpoints/checkpoint-02.pth

ls experiments/MLPF_test_*/checkpoints/*

## --------------------------------------------------------------------------------------------
## Test 3: Ray Train training using the 'ray-train' sub-command
## --------------------------------------------------------------------------------------------
#python mlpf/pipeline.py \
#  --config parameters/pytorch/pyg-cms.yaml \
#  --data-dir ${PWD}/tensorflow_datasets/ \
#  --experiments-dir ${PWD}/experiments \
#  --prefix MLPF_test_ \
#  --pipeline \
#  ray-train \
#  --num-steps 4 \
#  --ray-cpus 2 \
#  --ray-local \
#  --conv-type attention \
#  --dtype float32 \
#  --attention-type math \
#  --num-convs 1
