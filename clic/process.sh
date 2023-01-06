#!/bin/bash

IMG_HEPSIM=/home/software/singularity/centos7hepsim.img
IMG_MODERN=/home/software/singularity/tf-2.10.0.simg

infile=$1
filename="${infile%.*}"
echo $filename

singularity exec -B /local -B /scratch $IMG_HEPSIM ./process_hepsim.sh $filename.slcio
singularity exec -B /local -B /scratch $IMG_MODERN python3 convert_to_parquet.py $filename.pkl
rm $filename.pkl
