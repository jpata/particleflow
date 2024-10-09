## Validation data

```
gfal-copy -r root://xrootd.hep.kbfi.ee:1094//store/user/jpata/mlpf/results/ ./
```

See below for the steps to reproduce these samples.

The resulting plots can be found at:
```
https://jpata.web.cern.ch/jpata/mlpf/results/acat2022_20221004_model40M_revalidation20240523/
https://jpata.web.cern.ch/jpata/mlpf/results/acat2022_20221004_model40M_revalidation_CMSSW14_20240527/
```

## Code setup

The following should work on lxplus.
```
#ensure proxy is set
voms-proxy-init -voms cms -valid 192:00
voms-proxy-info

#Initialize EL8
cmssw-el8

export SCRAM_ARCH=el8_amd64_gcc12
cmsrel CMSSW_14_1_0_pre3
cd CMSSW_14_1_0_pre3/src
cmsenv
git cms-init

#set the directories we want to check out
echo "/Configuration/Generator/" >> .git/info/sparse-checkout
echo "/IOMC/ParticleGuns/" >>  .git/info/sparse-checkout
echo "/RecoParticleFlow/PFProducer/" >> .git/info/sparse-checkout
echo "/Validation/RecoParticleFlow/" >> .git/info/sparse-checkout

#checkout the CMSSW code
git remote add jpata https://github.com/jpata/cmssw.git
git fetch -a jpata
git checkout pfanalysis_caloparticle_CMSSW_14_1_0_pre3_acat2022

#compile
scram b -j4

#download the latest MLPF model
mkdir -p RecoParticleFlow/PFProducer/data/mlpf/
wget https://huggingface.co/jpata/particleflow/resolve/main/cms/2022_10_04_gnnlsh_model40M_acat2022/dev.onnx?download=true -O RecoParticleFlow/PFProducer/data/mlpf/dev.onnx

# must be b786aa6de49b51f703c87533a66326d6
md5sum RecoParticleFlow/PFProducer/data/mlpf/dev.onnx
```

## Running MLPF in CMSSW

### PF validation
To test MLPF on higher statistics, it's not practical to redo full reconstruction before the particle flow step.
We can follow a similar logic as the PF validation, where only the relevant PF sequences are rerun.

We use the following datasets for this:
```
/RelValQCD_FlatPt_15_3000HS_14/CMSSW_14_1_0_pre3-PU_140X_mcRun3_2024_realistic_v8_STD_2024_PU-v2/GEN-SIM-DIGI-RAW
/RelValTTbar_14TeV/CMSSW_14_1_0_pre3-PU_140X_mcRun3_2024_realistic_v8_STD_2024_PU-v2/GEN-SIM-DIGI-RAW
/RelValQQToHToTauTau_14TeV/CMSSW_14_1_0_pre3-PU_140X_mcRun3_2024_realistic_v8_STD_2024_PU-v2/GEN-SIM-DIGI-RAW
/RelValSingleEFlatPt2To100/CMSSW_14_1_0_pre3-PU_140X_mcRun3_2024_realistic_v8_STD_2024_PU-v2/GEN-SIM-DIGI-RAW
/RelValSingleGammaFlatPt8To150/CMSSW_14_1_0_pre3-PU_140X_mcRun3_2024_realistic_v8_STD_2024_PU-v2/GEN-SIM-DIGI-RAW
/RelValSinglePiFlatPt0p7To10/CMSSW_14_1_0_pre3-PU_140X_mcRun3_2024_realistic_v8_STD_2024_PU-v2/GEN-SIM-DIGI-RAW
```

#### MINIAOD with PF and MLPF
The PF validation workflows can be run using the scripts in
```
cd particleflow

#the number 1 signifies the row index (filename) in the input file to process
./scripts/cmssw/validation_job.sh mlpf scripts/cmssw/qcd_pu.txt QCD_PU 1
./scripts/cmssw/validation_job.sh pf scripts/cmssw/qcd_pu.txt QCD_PU 1
```

The MINIAOD output will be in `$CMSSW_BASE/out/QCD_PU_mlpf` and `$CMSSW_BASE/out/QCD_PU_pf`.

## Generating MLPF training samples

If you want to regenerate ML training samples from scratch with CMSSW, check the scripts
```
mlpf/data_cms/genjob_nopu.sh
mlpf/data_cms/genjob_pu55to75.sh
```

## pytorch training

Copy the datasets from EOS (about 500GB):
```
rsync -r --progress lxplus.cern.ch:/eos/user/j/jpata/mlpf/tensorflow_datasets/cms ./tensorflow_datasets
```

Download the pytorch distribution:
```
wget https://jpata.web.cern.ch/jpata/pytorch.simg
```

On a machine with a single GPU, the following is a quick test of the training workflow
```
singularity exec --env CUDA_VISIBLE_DEVICES=0 -B /scratch/persistent --nv \
    --env PYTHONPATH=hep_tfds \
    --env KERAS_BACKEND=torch \
    pytorch.simg python3.10 mlpf/pyg_pipeline.py --dataset cms --gpus 1 \
    --data-dir ./tensorflow_datasets --config parameters/pytorch/pyg-cms.yaml \
    --train --test --make-plots --conv-type attention --num-epochs 10 --gpu-batch-multiplier 1 \
    --num-workers 4 --prefetch-factor 100 --checkpoint-freq 1 --ntrain 1000 --ntest 1000 --nvalid 1000
```
