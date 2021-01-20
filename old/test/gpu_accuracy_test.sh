#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 1


#Hello World
#singularity exec -B /scratch -B /home --nv /home/software/singularity/base.simg:2020-08-13 \
 # python3 hello_world.py

singularity exec -B /home --nv /home/software/singularity/base.simg:latest python3 accuarcy_test.py --kwargs_path /home/aadi/praktika/model_init_data/model_kwargs.pkl --weights_path /home/aadi/praktika/model_init_data/weights.pth --test_bin /home/aadi/praktika/binned_data/bin_4k --l1m 1.0 --l2m 1.0 --target_type cand --device gpu


