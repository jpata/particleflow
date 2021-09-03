Short instructions to do a quick training on delphes data:
```bash
cd ../..
./scripts/local_test_delphes_pytorch.sh
```

### Delphes dataset
The dataset is available from zenodo: https://doi.org/10.5281/zenodo.4452283.

Instructions to download and process the full Delphes dataset:
```bash
cd ../../scripts/
./get_all_data_delphes.sh
```

This script will download and process the data under a directory called "test_tmp_delphes/" in particleflow. There are will be two subdirectories under test_tmp_delphes/ (1) data/: which contains the data (2) experiments/: which will contain any trained model


Instructions to explain using LRP (you must have an already trained model in test_tmp_delphes/experiments):
```bash
cd LRP/
python -u main_reg.py --LRP_load_model=<your_model> --LRP_load_epoch=<your_epoch>
```
