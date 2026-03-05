#!/bin/bash
set -e

WORKFLOW=cms_2025_main
MODEL=pyg-cms-v1

# Extract the container image from the spec file
IMG=$(./scripts/get_pytorch_image.sh)
APPTAINER_ARGS=" -B /local -B /cvmfs -B /scratch/local"

#Generate the training datasets
apptainer exec -B /local --env PYTHONPATH=`pwd` \
    $IMG \
    python3 mlpf/produce_snakemake.py \
    --production $WORKFLOW \
    --steps gen,post,tfds
./scripts/tallinn/kbfi-slurm-container -m snakemake --executor slurm \
    --profile tallinn \
    -s snakemake_jobs/$WORKFLOW/Snakefile \
    --jobs unlimited \
    --use-apptainer \
    --apptainer-args $APPTAINER_ARGS

#Train the model on GPU
apptainer exec -B /local --env PYTHONPATH=`pwd` \
    $IMG \
    python3 mlpf/produce_snakemake.py \
    --production $WORKFLOW \
    --steps train \
    --model $MODEL
./scripts/tallinn/kbfi-slurm-container -m snakemake --executor slurm \
    --profile tallinn \
    -s snakemake_jobs/$WORKFLOW/Snakefile \
    --jobs unlimited \
    --use-apptainer \
    --apptainer-args $APPTAINER_ARGS

#Run MC validation jobs (MLPF inference on generated validation datasets)
apptainer exec -B /local --env PYTHONPATH=`pwd` \
    $IMG \
    python3 mlpf/produce_snakemake.py \
    --production $WORKFLOW \
    --steps val
./scripts/tallinn/kbfi-slurm-container -m snakemake --executor slurm \
    --profile tallinn \
    -s snakemake_jobs/$WORKFLOW/Snakefile \
    --jobs unlimited \
    --use-apptainer \
    --apptainer-args $APPTAINER_ARGS

#Run data validation jobs (MLPF inference on existing data files)
apptainer exec -B /local --env PYTHONPATH=`pwd` \
    $IMG \
    python3 mlpf/produce_snakemake.py \
    --production $WORKFLOW \
    --steps val_data
./scripts/tallinn/kbfi-slurm-container -m snakemake --executor slurm \
    --profile tallinn \
    -s snakemake_jobs/$WORKFLOW/Snakefile \
    --jobs unlimited \
    --use-apptainer \
    --apptainer-args $APPTAINER_ARGS

#Run the validation plots
apptainer exec -B /local --env PYTHONPATH=`pwd` \
    $IMG \
    python3 mlpf/produce_validation_snakemake.py \
    --config validation_cms.yaml
./scripts/tallinn/kbfi-slurm-container -m snakemake --executor slurm \
    --profile tallinn \
    -s snakemake_jobs/validation_$WORKFLOW/Snakefile \
    --jobs unlimited \
    --use-apptainer \
    --apptainer-args $APPTAINER_ARGS
