#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:l40:1
#SBATCH --mem-per-gpu 40G
#SBATCH -o logs/slurm-%x-%j-%N.out

export PF_SITE=local
export RAY_TRAIN_ENABLE_V2_MIGRATION_WARNINGS=0

# Determine the data directory from the spec file
DATA_DIR=$(python3 scripts/get_param.py particleflow_spec.yaml productions.cld.workspace_dir)/tfds/

# Launch training using Ray Train
# --ray-local runs Ray on the local node without needing a pre-existing cluster
# --ray-gpus 1 specifies the number of workers (each with 1 GPU if --ray-gpus > 0)
# --ray-cpus 8 specifies the total number of CPUs to allocate for the training workers
./scripts/local/wrapper.sh python mlpf/pipeline.py \
    --spec-file particleflow_spec.yaml \
    --model-name pyg-cld-v1 \
    --production cld \
    --data-dir $DATA_DIR \
    ray-train \
    --ray-local \
    --ray-gpus 1 \
    --ray-cpus 8 \
    --gpu_batch_multiplier 256
