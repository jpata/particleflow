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
This script will download and process the data under `particleflow/data/clic`.

It is useful to note that the ssl pipeline relies heavily on the data split mode chosen to train VICReg and MLPF.
Currently, two data split modes are supported:
1. `domain_adaptation`: where VICReg is trained exclusively on the QCD sample. MLPF is trained exclusively on the TTbar sample and further tested on both the QCD and TTbar samples.
2. `mix`: where both VICReg and MLPF are trained on a mix of all samples. VICReg is trained on 90% of the data. MLPF uses the remaining 10%.

To specefy either data split mode, pass it to the --data_split_mode argument.

To run a training of a VICReg model:
```bash
cd ../
python ssl_pipeline.py --data_split_mode mix --prefix_VICReg VICReg_test
```

To train an mlpf via an ssl approach using the pre-trained VICReg model:
```bash
cd ../
python ssl_pipeline.py --data_split_mode mix --prefix_VICReg VICReg_test --load_VICReg --prefix_mlpf MLPF_test --train_mlpf --ssl
```
You can also add the argument `--native` to train a native (supervised) version of mlpf for comparisons with ssl.
