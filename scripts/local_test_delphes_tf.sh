#!/bin/bash
set -e

mkdir -p local_test_data/pythia8_ttbar
mkdir -p local_test_data/pythia8_ttbar/val
cd local_test_data/pythia8_ttbar

#download a test input file (you can also download everything from Zenodo at 10.5281/zenodo.4559324)
wget -q --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_ttbar_0_0.pkl.bz2
wget -q --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_ttbar_0_1.pkl.bz2
mv tev14_pythia8_ttbar_0_1.pkl.bz2 val/

cd ../..

mkdir -p experiments
rm -Rf experiments/test-*

echo "Cloning hep_tfds."
git clone https://github.com/erwulff/hep_tfds.git
echo "Installing hep_tfds."
cd hep_tfds
sudo python3 setup.py install
cd ..
rm -rf hep_tfds
echo "Removed hep_tfds repo."

#Run a simple training on a few events
rm -Rf local_test_data/pythia8_ttbar/tfr
python3 mlpf/launcher.py --model-spec parameters/test-delphes.yaml --action data

#Run a simple training on a few events
python3 mlpf/launcher.py --model-spec parameters/test-delphes.yaml --action train

#Generate the pred.npz file of predictions
python3 mlpf/launcher.py --model-spec parameters/test-delphes.yaml --action eval --weights ./experiments/test-*/weights/weights-01-*.hdf5

#Generate the timing file
python3 mlpf/launcher.py --model-spec parameters/test-delphes.yaml --action time --weights ./experiments/test-*/weights/weights-01-*.hdf5

