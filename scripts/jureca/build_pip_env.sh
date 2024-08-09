#!/bin/bash

# 2023-12-14
# Author: E. Wulff


module --force purge
ml Stages/2024 GCC/12.3.0 Python/3.11.3
ml CUDA/12 cuDNN/8.9.5.29-CUDA-12 NCCL/default-CUDA-12 Apptainer-Tools/2024

jutil env activate -p jureap57

python3 -m venv ray_tune_env

source ray_tune_env/bin/activate

pip3 install --upgrade pip
pip3 install numpy<1.25
pip3 install pandas<1.6.0dev0
pip3 install scikit-learn
pip3 install matplotlib
pip3 install tqdm
pip3 install autopep8
pip3 install mplhep
pip3 install awkward
pip3 install fastjet
pip3 install comet-ml
pip3 install tensorflow_datasets==4.9.3
pip3 install torch torchvision
pip3 install hls4ml[profiling]
pip3 install torch_geometric
pip3 install ray[data,train,tune,serve]
pip3 install async_timeout
pip3 install numba
pip3 install hyperopt
pip3 install causal-conv1d==1.0.2
pip3 install mamba-ssm
pip3 install comet-ml

deactivate
