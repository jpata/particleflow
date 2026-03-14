#!/bin/bash

module use /appl/local/containers/ai-modules
module load singularity-AI-bindings
module load aws-ofi-rccl

unset ROCR_VISIBLE_DEVICES
$WITH_CONDA
source particleflow-env/bin/activate
pip list installed
rocm-smi --showdriverversion
ray start --address $ip_head --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus $num_gpus_task --block --redis-password $redis_password
