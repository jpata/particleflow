#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem-per-gpu 20G
#SBATCH -o logs/slurm-%x-%j-%N.out

IMG=/home/software/singularity/pytorch.simg:2024-04-30
cd ~/particleflow

env

WEIGHTS=experiments/pyg-cms_20240430_094836_751206/checkpoints/checkpoint-25-17.631161.pth

singularity exec -B /scratch/persistent --nv \
     --env PYTHONPATH=hep_tfds \
     --env KERAS_BACKEND=torch \
     $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 0 \
     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
     --export-onnx --conv-type attention --attention-type math --gpu-batch-multiplier 10 --num-workers 1 --prefetch-factor 10 --load $WEIGHTS --dtype float32

singularity exec -B /scratch/persistent --nv \
     --env PYTHONPATH=hep_tfds \
     --env KERAS_BACKEND=torch \
     $IMG  python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
     --test --make-plots --conv-type attention --gpu-batch-multiplier 10 --num-workers 8 --prefetch-factor 10 --load $WEIGHTS --test-datasets cms_pf_ttbar --ntest 50000 &> logs/eval_cms_pf_ttbar.txt

singularity exec -B /scratch/persistent --nv \
     --env PYTHONPATH=hep_tfds \
     --env KERAS_BACKEND=torch \
     $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
     --test --make-plots --conv-type attention --gpu-batch-multiplier 10 --num-workers 8 --prefetch-factor 10 --load $WEIGHTS --test-datasets cms_pf_qcd --ntest 50000 &> logs/eval_cms_pf_qcd.txt

singularity exec -B /scratch/persistent --nv \
     --env PYTHONPATH=hep_tfds \
     --env KERAS_BACKEND=torch \
     $IMG python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
     --data-dir /scratch/persistent/joosep/tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
     --test --make-plots --conv-type attention --gpu-batch-multiplier 10 --num-workers 1 --prefetch-factor 10 --load $WEIGHTS --test-datasets cms_pf_ztt --ntest 50000 &> logs/eval_cms_pf_ztt.txt
