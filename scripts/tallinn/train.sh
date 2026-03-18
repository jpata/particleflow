#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:l40:1
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out
export PF_SITE=tallinn
DATA_DIR=$(python3 scripts/get_param.py particleflow_spec.yaml productions.cld.workspace_dir)/tfds/
./scripts/tallinn/wrapper.sh python mlpf/pipeline.py --spec-file particleflow_spec.yaml --model-name pyg-cld-v1 --production cld --data-dir $DATA_DIR train
