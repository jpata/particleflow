
# Installation

```
#get the code
git clone https://github.com/jpata/particleflow.git
cd particleflow
```

## CMS training

```
#Download the training datasets, about 400GB (access in only granted upon request to CMS members)
rsync -r --progress lxplus.cern.ch:/eos/user/j/jpata/mlpf/cms/tensorflow_datasets/cms_* ~/tensorflow_datasets/

#Run the training, multi-GPU support on the same machine is available, specify explicitly the GPUs you want to use
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 mlpf/pipeline.py train -c parameters/cms-gen.yaml
```

## CLIC training

```
#Download the training datasets, about 5GB
rsync -r --progress lxplus.cern.ch:/eos/user/j/jpata/mlpf/cms/tensorflow_datasets/clic_* ~/tensorflow_datasets/

#Run the training, multi-GPU support on the same machine is available, specify explicitly the GPUs you want to use
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 mlpf/pipeline.py train -c parameters/clic.yaml
```


## Delphes training

```
#Download the training datasets, about 30GB
rsync -r --progress lxplus.cern.ch:/eos/user/j/jpata/mlpf/cms/tensorflow_datasets/delphes_* ~/tensorflow_datasets/

#Run the training, multi-GPU support on the same machine is available, specify explicitly the GPUs you want to use
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 mlpf/pipeline.py train -c parameters/delphes.yaml
```
