
# Installation

```
#get the code
git clone https://github.com/jpata/particleflow.git
cd particleflow
```

## CLIC cluster-based training

```
#Download the training datasets, about 5GB
rsync -r --progress lxplus.cern.ch:/eos/user/j/jpata/mlpf/tensorflow_datasets/clic/clusters/clic_* ~/tensorflow_datasets/

#Run the training, multi-GPU support on the same machine is available, specify explicitly the GPUs you want to use
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 mlpf/pipeline.py train -c parameters/clic.yaml
```
