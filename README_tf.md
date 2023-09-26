
# Installation

```
#get the code
git clone https://github.com/jpata/particleflow.git
cd particleflow
git checkout clic_recipe_v1.6
```

## CLIC cluster-based training

```
#Download the training datasets, about 50GB
rsync -r --progress lxplus.cern.ch:/eos/user/j/jpata/mlpf/tensorflow_datasets/clic/clusters/clic_* ~/tensorflow_datasets/

#To speed up the first run, if batch sizes are unchanged, you can reuse the dataset number of steps and normalization caches
#Note: if batch size or number of GPUs is changed, the cache folder must be deleted to allow it to be regenerated, which requires two iterations over the dataset
wget https://hep.kbfi.ee/~joosep/mlpf_clic_dataset_cache.tar.gz
tar xf mlpf_clic_dataset_cache.tar.gz

#Run the training from scratch
python3 mlpf/pipeline.py train --config parameters/clic.yaml

#Run the training from a previous checkpoint, loading the model weights and optimizer state
python3 mlpf/pipeline.py train --config parameters/clic.yaml --weights models/mlpf-clic-2023-results/clusters_best_tuned_gnn_clic_v130/weights/weights-96-5.346523.hdf5

#Run the evaluation for a given training directory, loading the best weight file in the directory
python3 mlpf/pipeline.py train --config parameters/clic.yaml --train-dir experiments/clic/*

#Run the evaluation for the checkpoint training directory, loading the best weight file in the directory
mv models/mlpf-clic-2023-results/clusters_best_tuned_gnn_clic_v130/evaluation/epoch_96 models/mlpf-clic-2023-results/clusters_best_tuned_gnn_clic_v130/evaluation/epoch_96_old
python3 mlpf/pipeline.py train --config parameters/clic.yaml --train-dir models/mlpf-clic-2023-results/clusters_best_tuned_gnn_clic_v130/
```
