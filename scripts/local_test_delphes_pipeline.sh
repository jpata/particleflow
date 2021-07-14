#!/bin/bash
set -e

mkdir -p local_test_data/pythia8_ttbar
mkdir -p local_test_data/pythia8_ttbar/val
mkdir -p local_test_data/pythia8_ttbar/raw
mkdir -p local_test_data/pythia8_qcd
mkdir -p local_test_data/pythia8_qcd/val
cd local_test_data/pythia8_ttbar

#download a test input file (you can also download everything from Zenodo at 10.5281/zenodo.4559324)
wget -q --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_ttbar_0_0.pkl.bz2
wget -q --no-check-certificate -nc https://zenodo.org/record/4559324/files/tev14_pythia8_ttbar_0_1.pkl.bz2
cp tev14_pythia8_ttbar_0_0.pkl.bz2 raw/
cp tev14_pythia8_ttbar_0_1.pkl.bz2 val/
mv tev14_pythia8_ttbar_0_1.pkl.bz2 ../pythia8_qcd/val/

cd ../..

# Create downloads folder to trick tfds that dataset is already downloaded
mkdir local_test_data/downloads
mkdir local_test_data/downloads/delphes_pf
cp -r local_test_data/pythia8_* local_test_data/downloads/delphes_pf

mkdir -p experiments
rm -Rf experiments/test-*

#Run a simple training on a few events
rm -Rf local_test_data/pythia8_ttbar/tfr
python3 mlpf/launcher.py --model-spec parameters/test-delphes.yaml --action data

echo "Cloning hep_tfds."
git clone https://github.com/erwulff/hep_tfds.git
echo "Installing hep_tfds."
cd hep_tfds
python3 setup.py install
echo "Building TFRecords files."
tfds build heptfds/delphes_pf --overwrite --data_dir ../local_test_data/
cd ../
rm -rf hep_tfds
echo "Removed hep_tfds repo."

#Run a simple training on a few events
python3 mlpf/pipeline.py train -c parameters/test-delphes.yaml -p test-delphes-

#Generate the pred.npz file of predictions
python3 mlpf/pipeline.py evaluate -c parameters/test-delphes.yaml -t ./experiments/test-delphes-*

#Generate the timing file
python3 mlpf/launcher.py --model-spec parameters/test-delphes.yaml --action time --weights ./experiments/test-delphes-*/weights/weights-01-*.hdf5

rm -Rf local_test_data
