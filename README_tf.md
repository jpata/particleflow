
# Installation

```
git clone https://github.com/jpata/particleflow.git
cd particleflow
git checkout v1.6
```

# Dependencies

Install the required python packages, e.g.
```
pip install -r requirements.txt
```

or use the singularity/apptainer environment for CUDA GPUs:
```
apptainer shell --nv https://hep.kbfi.ee/~joosep/tf-2.13.0.simg
```

# CLIC cluster-based training

```
#Download the training datasets, about 50GB
rsync -r --progress lxplus.cern.ch:/eos/user/j/jpata/mlpf/tensorflow_datasets/clic/clusters/clic_* ~/tensorflow_datasets/

#Run the training from scratch, tuning the batch size to be suitable for an 8GB GPU
python3 mlpf/pipeline.py train --config parameters/clic.yaml --batch-multiplier 0.5

#Download and run the training from a checkpoint (optional)
wget https://huggingface.co/jpata/particleflow/resolve/main/weights-96-5.346523.hdf5
wget https://huggingface.co/jpata/particleflow/resolve/main/opt-96-5.346523.pkl
python3 mlpf/pipeline.py train --config parameters/clic.yaml --weights weights-96-5.346523.hdf5 --batch-multiplier 0.5

#Run the evaluation for a given training directory, loading the best weight file in the directory
python3 mlpf/pipeline.py evaluate --train-dir experiments/clic-REPLACEME

#Run the plots for a given training directory, using the evaluation outputs
python3 mlpf/pipeline.py plots --train-dir experiments/clic-REPLACEME
```
