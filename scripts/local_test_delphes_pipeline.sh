#!/bin/bash
set -e

mkdir -p data/pythia8_ttbar/raw
mkdir -p data/pythia8_qcd/val

#download a test input file (you can also download everything from Zenodo at 10.5281/zenodo.4559324)
wget -q --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_ttbar_0_0.pkl.bz2
mv tev14_pythia8_ttbar_0_0.pkl.bz2 data/pythia8_ttbar/raw/
wget -q --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_qcd_10_0.pkl.bz2
mv tev14_pythia8_qcd_10_0.pkl.bz2 data/pythia8_qcd/val/

mkdir -p experiments

echo "Cloning hep_tfds."
git clone https://github.com/erwulff/hep_tfds.git
echo "Installing hep_tfds."
cd hep_tfds
sudo python3 setup.py install
cd ..

#Run a simple training on a few events
rm -Rf data/pythia8_ttbar/tfr
python3 mlpf/pipeline.py data -c parameters/delphes.yaml

cd hep_tfds
echo "Building TFRecords files."
tfds build heptfds/delphes_pf --overwrite --manual_dir data/ --data_dir data/
cd ../
sudo rm -rf hep_tfds
echo "Removed hep_tfds repo."

#Run a simple training on a few events
python3 mlpf/pipeline.py train -c parameters/delphes.yaml --nepochs 2 --ntrain 5 --ntest 5

ls ./experiments/delphes_*/weights/

#Generate the pred.npz file of predictions
python3 mlpf/pipeline.py evaluate -c parameters/delphes.yaml -t ./experiments/delphes_*
