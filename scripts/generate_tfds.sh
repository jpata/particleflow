#!/bin/bash
rm -Rf data/Single* data/TTbar*

mkdir -p data/SingleElectronFlatPt1To100_pythia8_cfi/raw
cp -R mlpf/data/SingleElectronFlatPt1To100_pythia8_cfi/*/*.pkl data/SingleElectronFlatPt1To100_pythia8_cfi/raw/
PYTHONPATH=hep_tfds CUDA_VISIBLE_DEVICES=0 singularity exec --nv ~/HEP-KBFI/singularity/tf26.simg tfds build hep_tfds/heptfds/cms_pf/singleele --manual_dir data --overwrite

mkdir -p data/SinglePiFlatPt0p7To10_cfi/raw
cp -R mlpf/data/SinglePiFlatPt0p7To10_cfi/*/*.pkl data/SinglePiFlatPt0p7To10_cfi/raw/
PYTHONPATH=hep_tfds CUDA_VISIBLE_DEVICES=0 singularity exec --nv ~/HEP-KBFI/singularity/tf26.simg tfds build hep_tfds/heptfds/cms_pf/singlepi --manual_dir data --overwrite

mkdir -p data/SingleTauFlatPt2To150_cfi/raw
cp -R mlpf/data/SingleTauFlatPt2To150_cfi/*/*.pkl data/SingleTauFlatPt2To150_cfi/raw/
PYTHONPATH=hep_tfds CUDA_VISIBLE_DEVICES=0 singularity exec --nv ~/HEP-KBFI/singularity/tf26.simg tfds build hep_tfds/heptfds/cms_pf/singletau --manual_dir data --overwrite

mkdir -p data/TTbar_14TeV_TuneCUETP8M1_cfi/raw
cp -R mlpf/data/TTbar_14TeV_TuneCUETP8M1_cfi/*/*.pkl data/TTbar_14TeV_TuneCUETP8M1_cfi/raw/
PYTHONPATH=hep_tfds CUDA_VISIBLE_DEVICES=0 singularity exec --nv ~/HEP-KBFI/singularity/tf26.simg tfds build hep_tfds/heptfds/cms_pf/ttbar --manual_dir data --overwrite
