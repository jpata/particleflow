### DELPHES training
The dataset is available from zenodo: https://doi.org/10.5281/zenodo.4452283.

To download and process the full DELPHES dataset:
```bash
./get_data_delphes
```

This script will download and process the data under a directory called data/delphes under /particleflow.

To perform a quick training on the dataset:
```bash
cd ../
python -u pyg_pipeline.py --data delphes --overwrite --target='gen'
```

### CMS training

To download and process the full CMS dataset:
```bash
./get_data_cms
```
This script will download and process the data under a directory called data/cms under /particleflow.

To perform a quick training on the dataset:
```bash
cd ../
python -u pyg_pipeline.py --data cms --overwrite --target='gen' --dataset=<path_to_data_cms> --dataset_qcd=<path_to_data_cms>
```

### Instructions to run LRP

You must have a pre-trained model under particleflow/experiments):
```bash
cd ../
python -u lrp_pipeline.py --run_lrp --make_rmaps --load_model=<your_model> --load_epoch=<your_epoch>
```
