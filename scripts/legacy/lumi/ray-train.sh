#!/bin/bash

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
module load aws-ofi-rccl

unset ROCR_VISIBLE_DEVICES
$WITH_CONDA
source particleflow-env/bin/activate
pip list installed
rocm-smi --showdriverversion
python3 mlpf/pipeline.py \
    --data-dir $TFDS_DATA_DIR --config parameters/pytorch/pyg-cms.yaml \
    --ray-train --gpus $num_gpus --ray-gpus $num_gpus_task --ray-cpus $((SLURM_CPUS_PER_TASK*SLURM_JOB_NUM_NODES)) \
    --experiments-dir /scratch/project_465001293/joosep/particleflow/experiments-ray --comet \
    --gpu-batch-multiplier 12 --num-workers 4 --prefetch-factor 10 --checkpoint-freq 1 --conv-type attention --dtype bfloat16 --optimizer lamb --lr 0.002
