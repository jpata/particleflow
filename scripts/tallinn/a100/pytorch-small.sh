#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:mig:1
#SBATCH --mem-per-gpu 60G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-07-03
cd ~/particleflow

env

# singularity exec -B /scratch/persistent --nv \
#     --env PYTHONPATH=hep_tfds \
#     --env KERAS_BACKEND=torch \
#     $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
#     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
#     --train --conv-type attention --attention-type flash --gpu-batch-multiplier 5 --num-workers 1 --prefetch-factor 50 --dtype bfloat16 --ntrain 1000 --nvalid 1000 --num-epochs 50

WEIGHTS=experiments/pyg-cms_20240705_102527_068348/checkpoints/checkpoint-44-25.959111.pth
# singularity exec -B /scratch/persistent --nv \
#      --env PYTHONPATH=hep_tfds \
#      --env KERAS_BACKEND=torch \
#      $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 0 \
#      --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
#      --export-onnx --conv-type attention --attention-type math --gpu-batch-multiplier 10 --num-workers 1 --prefetch-factor 10 --load $WEIGHTS --dtype float32
#

singularity exec -B /scratch/persistent --nv \
     --env PYTHONPATH=hep_tfds \
     --env KERAS_BACKEND=torch \
     $IMG  python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
     --test --make-plots --conv-type attention --gpu-batch-multiplier 10 --load $WEIGHTS --ntrain 100000 --nvalid 100000 --ntest 100000 #--test-datasets cms_pf_ttbar --ntest 50000 &> logs/eval_cms_pf_ttbar.txt

# singularity exec -B /scratch/persistent --nv \
#      --env PYTHONPATH=hep_tfds \
#      --env KERAS_BACKEND=torch \
#      $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
#      --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
#      --test --make-plots --conv-type attention --gpu-batch-multiplier 10 --num-workers 8 --prefetch-factor 10 --load $WEIGHTS --test-datasets cms_pf_qcd --ntest 50000 &> logs/eval_cms_pf_qcd.txt
#
# singularity exec -B /scratch/persistent --nv \
#      --env PYTHONPATH=hep_tfds \
#      --env KERAS_BACKEND=torch \
#      $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
#      --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
#      --test --make-plots --conv-type attention --gpu-batch-multiplier 10 --num-workers 1 --prefetch-factor 10 --load $WEIGHTS --test-datasets cms_pf_ztt --ntest 50000 &> logs/eval_cms_pf_ztt.txt
