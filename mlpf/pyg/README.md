# Setup

Have conda installed.
```bash
conda env create -f environment.yml
conda activate mlpf
```

# Downloading the datasets

## CMS dataset

To download and process the full CMS dataset:
```bash
./get_data_cms.sh
```
This script will download and process the data under a directory called `data/cms` under `particleflow`.

## DELPHES dataset
The dataset is available from zenodo: https://doi.org/10.5281/zenodo.4452283.

To download and process the full DELPHES dataset:
```bash
./get_data_delphes.sh
```
This script will download and process the data under a directory called `data/delphes` under `particleflow`.

## CLIC dataset

To download and process the full CLIC dataset:
```bash
./get_data_clic.sh
```
This script will download and process the data under a directory called `data/clic_edm4hep_2023_02_27` under `particleflow`.


# Supervised training

The training script for either CMS, DELPHES or CLIC dataset is the same.

For example:
```bash
cd ../
python -u pyg_pipeline.py --dataset=${dataset} --data_path=${data_path} --outpath=${outpath} --prefix=${model_prefix}
```
where:
- dataset: `CMS` or `DELPHES` or `CLIC`.
- data_path: path to the samples (e.g. `../data/cms/`)
- outpath: path to store the experiment (by default: `../experiments`).
- prefix: the name of the model which will be the name of the directory that holds the results.

Adding the arguments `--make_predictions --make_plots` will run inference and make plots for evaluation.

Adding the argument `--load` will load a model instead of training from scratch.


# Self supervised training

The idea is to pre-train a model similar in architecture to MLPF but using unlabeled data. This model is based on the VICReg model: https://arxiv.org/abs/2105.04906.

It is useful to note that the ssl pipeline relies heavily on the data split mode chosen to train VICReg and MLPF.
Currently, two data split modes are supported:
1. `domain_adaptation`: where VICReg is trained exclusively on the QCD sample. MLPF is trained exclusively on the TTbar sample and further tested on both the QCD and TTbar samples.
2. `mix`: where both VICReg and MLPF are trained on a mix of all samples. VICReg is trained on 90% of the data. MLPF uses the remaining 10%.

To specify either data split mode, pass it to the --data_split_mode argument.

**Note:** only CLIC dataset is supported for ssl trainings.

To run a training of a VICReg model:
```bash
cd ../
python ssl_pipeline.py --data_split_mode mix --prefix_VICReg VICReg_test
```

To train an mlpf via an ssl approach using the pre-trained VICReg model:
```bash
cd ../
python ssl_pipeline.py --data_split_mode mix --prefix_VICReg VICReg_test --load_VICReg --prefix MLPF_test --train_mlpf --ssl
```
You can also add the argument `--native` to train a native (supervised) version of mlpf for comparisons with ssl.

To evaluate the MLPF model(s):
```bash
cd ../
python ssl_evaluate.py --data_split_mode mix --prefix_VICReg VICReg_test --load_VICReg --prefix MLPF_test --ssl --native
```