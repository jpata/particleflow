# Setup

Have conda installed.
```bash
conda env create -f environment.yml
conda activate mlpf
```

# Training

### DELPHES training
The dataset is available from zenodo: https://doi.org/10.5281/zenodo.4452283.

To download and process the full DELPHES dataset:
```bash
./get_data_delphes.sh
```

This script will download and process the data under a directory called `data/delphes` under `particleflow`.

To perform a quick training on the dataset:
```bash
cd ../
python -u pyg_pipeline.py --data delphes --dataset=<path_to_delphes_data> --dataset_qcd=<path_to_delphes_data>
```

To load a pretrained model which is stored in a directory under `particleflow/experiments` for evaluation:
```bash
cd ../
python -u pyg_pipeline.py --data delphes --load --load_model=<model_directory> --load_epoch=<epoch_to_load> --dataset=<path_to_delphes_data> --dataset_qcd=<path_to_delphes_data>
```

### CMS training

To download and process the full CMS dataset:
```bash
./get_data_cms.sh
```
This script will download and process the data under a directory called `data/cms` under `particleflow`.

To perform a quick training on the dataset:
```bash
cd ../
python -u pyg_pipeline.py --data cms --dataset=<path_to_cms_data> --dataset_qcd=<path_to_cms_data>
```

To load a pretrained model which is stored in a directory under `particleflow/experiments` for evaluation:
```bash
cd ../
python -u pyg_pipeline.py --data cms --load --load_model=<model_directory> --load_epoch=<epoch_to_load> --dataset=<path_to_cms_data> --dataset_qcd=<path_to_cms_data>
```

### XAI and LRP studies on MLPF

You must have a pre-trained model under `particleflow/experiments`:
```bash
cd ../
python -u lrp_mlpf_pipeline.py --run_lrp --make_rmaps --load_model=<your_model> --load_epoch=<your_epoch>
```
