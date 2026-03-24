#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:l40:1
#SBATCH --mem-per-gpu 40G
#SBATCH --cpus-per-gpu 4
#SBATCH -o logs/slurm-%x-%j-%N.out
set -e
export PF_SITE=tallinn

DATA_DIR=$(pixi run python3 scripts/get_param.py particleflow_spec.yaml productions.cld.workspace_dir)/tfds/
./scripts/tallinn/wrapper.sh python mlpf/pipeline.py --spec-file particleflow_spec.yaml --model-name pyg-cld-hits-v1 --production cld --data-dir $DATA_DIR --prefix plots_ test --load experiments/pyg-cld-hits-v1_cld_20260323_132139_857098/checkpoints/checkpoint-50000.pth --make_plots True --ntest 1000
./scripts/tallinn/wrapper.sh python mlpf/pipeline.py --spec-file particleflow_spec.yaml --model-name pyg-cld-v1 --production cld --data-dir $DATA_DIR --prefix plots_ test --load experiments/pyg-cld-v1_cld_20260323_131651_522592/checkpoints/checkpoint-50000.pth --make_plots True --ntest 1000
