#!/bin/bash

singularity exec -B /local /home/software/singularity/pytorch.simg\:2026-01-20 python3 mlpf/data/cms/produce_snakemake.py --production cld_2025_edm4hep
./scripts/tallinn/container-python -m snakemake --executor slurm --profile tallinn -s snakemake_jobs/cld_2025_edm4hep/Snakefile --jobs unlimited --use-apptainer --apptainer-args " -B /local -B /cvmfs -B /scratch/local"
