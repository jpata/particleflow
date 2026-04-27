#!/bin/bash
sbatch scripts/tallinn/train_cld.sh
sbatch scripts/tallinn/train_cld_hits.sh
sbatch scripts/tallinn/train_clic.sh
sbatch scripts/tallinn/train_clic_hits.sh
