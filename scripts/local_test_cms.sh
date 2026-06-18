#!/bin/bash
set -e
export TFDS_DATA_DIR=`pwd`/tensorflow_datasets
export PWD=`pwd`
export PYTHONPATH=`pwd`
export KERAS_BACKEND=torch

# Quick unit tests
rm -Rf .pytest_cache
uv run python -m pytest --cache-clear tests

# 1. Fetch test data
./scripts/fetch_test_data_cms.sh

#create the tensorflow dataset for the last split config only
uv run tfds build mlpf/heptfds/cms_pf/ttbar --config 10 --manual_dir ./local_test_data

mkdir -p experiments

# --------------------------------------------------------------------------------------------
# Test 1: Initial training using the 'train' sub-command
# --------------------------------------------------------------------------------------------
uv run python mlpf/pipeline.py \
  --spec-file particleflow_spec.yaml \
  --model-name pyg-cms-v1 \
  --production cms_run3 \
  --data-dir ./tensorflow_datasets/ \
  --prefix MLPF_test_ \
  --pipeline \
  train \
  --num_steps 2 \
  --checkpoint_freq 1 \
  --gpus 0 \
  --make-plots \
  --conv_type attention \
  --dtype float32 \
  --model.attention.attention_type math \
  --model.attention.num_convs 1 \
  --ntrain 10 --ntest 10 --nvalid 10 \
  --num_workers 1 --prefetch_factor 1

ls experiments/MLPF_test_*/checkpoints/*

# Capture the experiment directory created by the first run for the next steps
export EXP_DIR_1=$(ls -d experiments/MLPF_test_*/ | tail -n 1)

# --------------------------------------------------------------------------------------------
# Test 2: Fine-tuning from a checkpoint in a NEW directory
# --experiment-dir is omitted, so a new one is created.
# --------------------------------------------------------------------------------------------
uv run python mlpf/pipeline.py \
  --spec-file particleflow_spec.yaml \
  --model-name pyg-cms-v1 \
  --production cms_run3 \
  --data-dir ./tensorflow_datasets/ \
  --prefix MLPF_test_ \
  --pipeline \
  train \
  --num_steps 4 \
  --checkpoint_freq 1 \
  --gpus 0 \
  --make-plots \
  --conv_type attention \
  --dtype float32 \
  --model.attention.attention_type math \
  --model.attention.num_convs 1 \
  --load ${EXP_DIR_1}/checkpoints/checkpoint-02.pth \
  --ntrain 10 --ntest 10 --nvalid 10 \
  --num_workers 1 --prefetch_factor 1

ls experiments/MLPF_test_*/checkpoints/*

# Capture the latest experiment directory (from Test 2)
export EXP_DIR_2=$(ls -d experiments/MLPF_test_*/ | tail -n 1)

# --------------------------------------------------------------------------------------------
# Test 3: ONNX export and validation
# --------------------------------------------------------------------------------------------
uv run --project envs/ort-cpu python scripts/cms-validate-onnx.py \
  --checkpoint ${EXP_DIR_2}/checkpoints/checkpoint-04.pth \
  --model-kwargs ${EXP_DIR_2}/model_kwargs.pkl \
  --dataset cms_pf_ttbar/10 \
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
#  --num_steps 4 \
#  --ray-cpus 2 \
#  --ray-local \
#  --conv-type attention \
#  --dtype float32 \
#  --attention-type math \
#  --num-convs 1
