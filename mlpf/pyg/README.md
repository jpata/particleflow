# Setup

This image https://gitlab.nrp-nautilus.io/fmokhtar/particleflow contains the necessary libraries to run MLPF with the pytorch backend.

You can pull the image using docker e.g. `docker pull gitlab-registry.nrp-nautilus.io/fmokhtar/particleflow:latest`.

Alternatively, you can refer to the singularity image https://github.com/HEP-KBFI/singularity/blob/master/specs/pytorch.singularity.

# Retrieving the datasets

The current pytorch backend shares the same dataset format as the tensorflow backend which uses `tensorflow_datasets` (more information can be found here `../heptfds`)

# Supervised training or testing

First make sure to update the config yaml e.g. `../../parameters/pyg-cms-test.yaml` to your desired model parameter configuration and choice of physics samples for training and testing.

After that, the entry point to launch training or testing for either CMS, DELPHES or CLIC is the same. From the main repo run,

```bash
python -u mlpf/pyg_pipeline.py --dataset=${} --data_dir=${} --prefix=${} --gpus=${} --ntrain 10 --nvalid 10 --ntest 10
```
where:
- `--dataset`: choices are `cms` or `delphes` or `clic`
- `--data_dir`: path to the tensorflow_datasets (e.g. `../data/tensorflow_datasets/`)
- `--prefix`: path pointing to the model directory (note: a unique hash will be appended to avoid overwrite)
- `--gpus`: to use CPU set to empty string ""; else to use gpus provide e.g. "0,1"
- `ntrain`, `nvalid`, `ntest`: specefies number of events (per sample) that will be used

Adding the arguments:
- `--load` will load a pre-trained model
- `--train` will run a training (may train a loaded model if `--load` is provided)
- `--test` will run inference and save the predictions as `.parquets`
- `--make-plots` will use the predictions stored after running with `--test` to make plots for evaluation
- `--export-onnx` will export the model to ONNX

You can also pass your own config yaml to `--config` (by default: `../../parameters/pyg_config.yaml`).
