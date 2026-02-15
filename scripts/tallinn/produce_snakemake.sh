#!/bin/bash
set -e

WORKFLOW=cms_2025_main
MODEL=pyg-cms-v1

#WORKFLOW=cld_2025_edm4hep
#MODEL=pyg-cld-v1

#WORKFLOW=clic_2025_edm4hep
#MODEL=pyg-clic-v1

singularity exec -B /local --env PYTHONPATH=`pwd` /home/software/singularity/pytorch.simg\:2026-02-04 python3 mlpf/produce_snakemake.py --production $WORKFLOW --steps gen,post,tfds
./scripts/tallinn/container-python -m snakemake --executor slurm --profile tallinn -s snakemake_jobs/$WORKFLOW/Snakefile --jobs unlimited --use-apptainer --apptainer-args " -B /local -B /cvmfs -B /scratch/local"

singularity exec -B /local --env PYTHONPATH=`pwd` /home/software/singularity/pytorch.simg\:2026-02-04 python3 mlpf/produce_snakemake.py --production $WORKFLOW --steps train --model $MODEL
./scripts/tallinn/container-python -m snakemake --executor slurm --profile tallinn -s snakemake_jobs/$WORKFLOW/Snakefile --jobs unlimited --use-apptainer --apptainer-args " -B /local -B /cvmfs -B /scratch/local --nv"
