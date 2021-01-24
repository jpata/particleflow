#!/bin/bash
#SBATCH -p gpu
#SBATCH --gpus 1


#Hello World
#singularity exec -B /scratch -B /home --nv /home/software/singularity/base.simg:2020-08-13 \
 # python3 hello_world.py


singularity exec -B /home --nv /home/software/singularity/base.simg:latest python3 model_time_eval.py --kwargs_path /home/aadi/praktika/model_init_data/model_kwargs.pkl --weights_path /home/aadi/praktika/model_init_data/weights.pth --bins_dir_path /home/aadi/praktika/binned_data --save_path /home/aadi/praktika/pickle_lists_final4 --repetitions 30 --sample_size 1000 --device gpu




~                           
