# Setup

Have conda installed.
```bash
conda env create -f environment.yml
conda activate mlpf
```

# Semi-supervised training on CLIC

To download and process the full CLIC dataset:
```bash
cd clic/
./get_data_clic.sh
```
This script will download and process the data under a directory called `data/clic` under `particleflow`.

To run a training of VICReg:
```bash
cd ../
python ssl_pipeline.py --model_prefix_VICReg VICReg_test
```

To train mlpf via an ssl approach using the pre-trained VICReg model:
```bash
cd ../
python ssl_pipeline.py --model_prefix_VICReg VICReg_test --load_VICReg --model_prefix_mlpf MLPF_test --train_mlpf
```
