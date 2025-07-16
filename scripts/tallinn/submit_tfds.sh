#!/bin/bash

SUB=scripts/tallinn/generate_tfds.sh

# export DATA_DIR=/local/joosep/mlpf/tensorflow_datasets/cms/2.7.1/
# export MANUAL_DIR=/local/joosep/mlpf/cms/20250508_cmssw_15_0_5_d3c6d1/
# for i in `seq 1 10`; do
#     sbatch $SUB cms_pf/qcd_nopu $i nopu
#     sbatch $SUB cms_pf/ttbar_nopu $i nopu
#     sbatch $SUB cms_pf/ztt_nopu $i nopu
#     sbatch $SUB cms_pf/qcd $i pu55to75
#     sbatch $SUB cms_pf/ttbar $i pu55to75
#      sbatch $SUB cms_pf/ztt $i pu55to75
# done

# export DATA_DIR=/local/joosep/mlpf/tensorflow_datasets/clic
# export MANUAL_DIR=/local/joosep/mlpf/clic_edm4hep/
# for i in `seq 1 10`; do
#     sbatch $SUB clic_pf_edm4hep/ttbar $i
#     sbatch $SUB clic_pf_edm4hep/qq $i
#     sbatch $SUB clic_pf_edm4hep/ww_fullhad $i
# done
