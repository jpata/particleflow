#!/bin/bash

SUB=scripts/tallinn/generate_tfds.sh

for i in `seq 1 10`; do
    sbatch $SUB qcd_nopu $i nopu
    sbatch $SUB ttbar_nopu $i nopu
    sbatch $SUB ztt_nopu $i nopu
    sbatch $SUB qcd $i pu55to75
    sbatch $SUB ttbar $i pu55to75
    sbatch $SUB ztt $i pu55to75
done
