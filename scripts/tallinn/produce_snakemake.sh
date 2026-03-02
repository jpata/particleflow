#!/bin/bash
set -e

WORKFLOW=cms_2025_main
MODEL=pyg-cms-v1

#WORKFLOW=cld_2025_edm4hep
#MODEL=pyg-cld-v1

#WORKFLOW=clic_2025_edm4hep
#MODEL=pyg-clic-v1

# Extract the container image from the spec file
IMG=$(./scripts/get_pytorch_image.sh)

#singularity exec -B /local --env PYTHONPATH=`pwd` \
#    $IMG \
#    python3 mlpf/produce_snakemake.py \
#    --production $WORKFLOW \
#    --steps gen,post,tfds
#./scripts/tallinn/kbfi-slurm-container -m snakemake --executor slurm \
#    --profile tallinn \
#    -s snakemake_jobs/$WORKFLOW/Snakefile \
#    --jobs unlimited \
#    --use-apptainer \
#    --apptainer-args " -B /local -B /cvmfs -B /scratch/local"
#
#singularity exec -B /local --env PYTHONPATH=`pwd` \
#    $IMG \
#    python3 mlpf/produce_snakemake.py \
#    --production $WORKFLOW \
#    --steps train \
#    --model $MODEL
#./scripts/tallinn/kbfi-slurm-container -m snakemake --executor slurm \
#    --profile tallinn \
#    -s snakemake_jobs/$WORKFLOW/Snakefile \
#    --jobs unlimited \
#    --use-apptainer \
#    --apptainer-args " -B /local -B /cvmfs -B /scratch/local --nv"

singularity exec -B /local --env PYTHONPATH=`pwd` \
    $IMG \
    python3 mlpf/produce_snakemake.py \
    --production $WORKFLOW \
    --steps val_data
./scripts/tallinn/kbfi-slurm-container -m snakemake --executor slurm \
    --profile tallinn \
    -s snakemake_jobs/$WORKFLOW/Snakefile \
    --jobs unlimited \
    --use-apptainer \
    --apptainer-args " -B /local -B /cvmfs -B /scratch/local"
