#!/bin/bash
set -e
export TFDS_DATA_DIR=`pwd`/tensorflow_datasets
export PWD=`pwd`
export PYTHONPATH=`pwd`
export KERAS_BACKEND=torch

# START comment block
# Commented out for now, do not enable these so that the test is quick!
# # Quick unit tests
# python -m pytest tests/test_dataloader.py
# python -m pytest tests/test_dataloader_behavior.py
# python -m pytest tests/test_endless_interleaved_iterator.py
# python -m pytest tests/test_resumable_sampler.py
# python -m pytest tests/test_interleaved_iterator.py
# python -m pytest tests/test_lr_schedule.py
# python -m pytest tests/test_config_overrides.py

# #create data directories
# rm -Rf local_test_data/TTbar_13p6TeV_TuneCUETP8M1_cfi
# mkdir -p local_test_data/TTbar_13p6TeV_TuneCUETP8M1_cfi/root
# cd local_test_data/TTbar_13p6TeV_TuneCUETP8M1_cfi/root

# #Only CMS-internal use is permitted by CMS rules! Do not use these MC simulation files otherwise!
# wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/20240823_simcluster/pu55to75/TTbar_14TeV_TuneCUETP8M1_cfi/root/pfntuple_100000.root
# wget -q --no-check-certificate -nc https://jpata.web.cern.ch/jpata/mlpf/cms/20240823_simcluster/pu55to75/TTbar_14TeV_TuneCUETP8M1_cfi/root/pfntuple_100001.root

# cd ../../..

# #Create the ntuples using postprocessing2.py
# for file in `\ls -1 local_test_data/TTbar_13p6TeV_TuneCUETP8M1_cfi/root/*.root`; do
#   python mlpf/data/cms/postprocessing2.py \
#     --input $file \
#     --outpath local_test_data/TTbar_13p6TeV_TuneCUETP8M1_cfi
# done
# find local_test_data

# #create the tensorflow dataset for the last split config only
# tfds build mlpf/heptfds/cms_pf/ttbar --config 10 --manual_dir ./local_test_data

# mkdir -p experiments
# END comment block

# --------------------------------------------------------------------------------------------
# Test 1: Initial training using the 'train' sub-command
# --------------------------------------------------------------------------------------------
python mlpf/pipeline.py \
  --spec-file particleflow_spec.yaml \
  --model-name pyg-cms-v1 \
  --production cms_run3 \
  --data-dir ./tensorflow_datasets/ \
  --prefix MLPF_test_ \
  --pipeline \
  train \
  --num-steps 2 \
  --checkpoint-freq 1 \
  --gpus 0 \
  --make-plots \
  --conv-type attention \
  --dtype float32 \
  --attention-type math \
  --num-convs 1 \
  --num_workers 1 --prefetch_factor 1

ls experiments/MLPF_test_*/checkpoints/*

# Capture the experiment directory created by the first run for the next steps
export EXP_DIR_1=$(ls -d experiments/MLPF_test_*/ | tail -n 1)

# --------------------------------------------------------------------------------------------
# Test 2: Fine-tuning from a checkpoint in a NEW directory
# --experiment-dir is omitted, so a new one is created.
# --------------------------------------------------------------------------------------------
python mlpf/pipeline.py \
  --spec-file particleflow_spec.yaml \
  --model-name pyg-cms-v1 \
  --production cms_run3 \
  --data-dir ./tensorflow_datasets/ \
  --prefix MLPF_test_ \
  --pipeline \
  train \
  --num-steps 4 \
  --checkpoint-freq 1 \
  --gpus 0 \
  --make-plots \
  --conv-type attention \
  --dtype float32 \
  --attention-type math \
  --num-convs 1 \
  --load ${EXP_DIR_1}/checkpoints/checkpoint-02.pth \
  --num_workers 1 --prefetch_factor 1

ls experiments/MLPF_test_*/checkpoints/*

# Capture the latest experiment directory (from Test 2)
export EXP_DIR_2=$(ls -d experiments/MLPF_test_*/ | tail -n 1)

# --------------------------------------------------------------------------------------------
# Test 3: ONNX export and validation
# --------------------------------------------------------------------------------------------
python scripts/cms-validate-onnx.py \
  --checkpoint ${EXP_DIR_2}/checkpoints/checkpoint-04.pth \
  --model-kwargs ${EXP_DIR_2}/model_kwargs.pkl \
  --dataset cms_pf_ttbar \
  --data-dir ./tensorflow_datasets/ \
  --num-events 2 \
  --outdir ./onnx_validation_cms --device cpu

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
